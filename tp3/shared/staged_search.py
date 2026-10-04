"""Shared search rules and execution of independent development runs."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .experiments import (config_identity, data_hashes, recorded_config, run,
                          sha256_file, validate_config, write_json)

REQUIRED = {'schema_version', 'protocol', 'base', 'primary_seed', 'ranking', 'near_tie',
            'improvement_rule', 'stages'}

# ---------------------------------------------------------------- pure rules

def rank_key(row):
    return (-row['accuracy'], row['loss'], row['parameter_count'], row['config_id'])


def rank(rows):
    """Completed seed rows by the pre-registered order; failed runs never rank."""
    return sorted((r for r in rows if r['status'] == 'completed'), key=rank_key)


def group_summary(rows, seeds):
    """Mean/std per configuration over exactly the given seeds.

    Missing or duplicated seeds are a protocol error. A group with a failed
    seed is kept as 'failed' and excluded from the ranking.
    """
    groups = {}
    for row in rows:
        groups.setdefault(row['config_id'], []).append(row)
    if not groups:
        raise ValueError('No runs to group.')
    summary = []
    for config_id, group in groups.items():
        if sorted(r['seed'] for r in group) != sorted(seeds):
            raise ValueError(f'Group {config_id} requires exactly the seeds {sorted(seeds)}.')
        if len({(r['dataset_sha256'], r['split_sha256']) for r in group}) != 1:
            raise ValueError('A seed group requires a common dataset and partition.')
        entry = {'config_id': config_id, 'label': group[0]['label'], 'seeds': sorted(seeds),
                 'run_ids': [r['run_id'] for r in sorted(group, key=lambda r: r['seed'])],
                 'parameter_count': group[0]['parameter_count'], 'config': group[0]['config']}
        if any(r['status'] != 'completed' for r in group):
            entry['status'] = 'failed'
            summary.append(entry)
            continue
        accuracy = [r['accuracy'] for r in group]
        loss = [r['loss'] for r in group]
        entry.update(status='completed', accuracy_mean=float(np.mean(accuracy)),
                     accuracy_std=float(np.std(accuracy, ddof=1)) if len(group) > 1 else None,
                     loss_mean=float(np.mean(loss)),
                     loss_std=float(np.std(loss, ddof=1)) if len(group) > 1 else None,
                     chosen_epochs=[r['chosen_epoch'] for r in sorted(group, key=lambda r: r['seed'])],
                     all_converged=all(r['converged'] for r in group),
                     duration_seconds_mean=float(np.mean([r['duration_seconds'] for r in group])))
        summary.append(entry)
    return rank_groups(summary)


def rank_groups(summary):
    """Stage-7 order: mean accuracy, mean cross-entropy, parameters, config_id."""
    completed = sorted((s for s in summary if s['status'] == 'completed'), key=lambda s: (
        -s['accuracy_mean'], s['loss_mean'], s['parameter_count'], s['config_id']))
    return completed + sorted((s for s in summary if s['status'] != 'completed'),
                              key=lambda s: s['config_id'])


def pooled_sigma(a, b):
    """Pooled sample standard deviation (ddof=1) of two seed groups."""
    return math.sqrt((a['accuracy_std'] ** 2 + b['accuracy_std'] ** 2) / 2)


def compare_groups(better, other, multiplier):
    sigma = pooled_sigma(better, other)
    difference = better['accuracy_mean'] - other['accuracy_mean']
    return {'better': better['config_id'], 'other': other['config_id'],
            'difference': difference, 'sigma': sigma, 'threshold': multiplier * sigma,
            'significant': difference > multiplier * sigma}


def near_tie_pair(ranked, threshold):
    """The two best seed-42 rows when they differ by less than threshold."""
    if len(ranked) >= 2 and ranked[0]['accuracy'] - ranked[1]['accuracy'] < threshold:
        return ranked[:2]
    return None


def resolve_near_tie(summary, incumbent_id, multiplier):
    """Improvement rule on 3-seed groups; returns (winner config_id, evidence)."""
    completed = [s for s in summary if s['status'] == 'completed']
    if not completed:
        raise ValueError('Both near-tie groups failed.')
    if len(completed) == 1:
        return completed[0]['config_id'], {'reason': 'only one group completed every seed'}
    leader, other = completed[:2]
    comparison = compare_groups(leader, other, multiplier)
    if comparison['significant']:
        winner, reason = leader['config_id'], 'mean advantage exceeds the sigma threshold'
    elif incumbent_id in (leader['config_id'], other['config_id']):
        winner, reason = incumbent_id, 'not distinguishable; incumbent kept'
    else:
        winner, reason = leader['config_id'], 'not distinguishable; 3-seed ranking'
    return winner, {'reason': reason, 'comparison': comparison}


def choose_activation(rows, threshold, preference):
    """Best learning rate per activation; tie (< threshold) chooses preference."""
    best = {}
    for row in rank(rows):
        best.setdefault(row['activation'], row)
    if not best:
        raise ValueError('No completed stage-0 runs.')
    ordered = sorted(best.values(), key=rank_key)
    decision = {'best_per_activation': {a: r['run_id'] for a, r in best.items()}}
    if len(ordered) == 1:
        return ordered[0], {**decision, 'reason': 'only one activation completed'}
    difference = ordered[0]['accuracy'] - ordered[1]['accuracy']
    tie = difference < threshold and preference in best
    winner = best[preference] if tie else ordered[0]
    return winner, {**decision, 'difference': difference, 'tie': tie,
                    'reason': f'difference below threshold; {preference} preferred' if tie
                    else 'higher validation accuracy'}


def _same(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)


def sequence_neighbor(value, direction, mantissas=(1, 3)):
    """Next point of the 1-3-10 sequence above (direction > 0) or below value."""
    exponent = math.floor(math.log10(value))
    points = sorted(float(f'{m}e{e}') for e in range(exponent - 2, exponent + 3) for m in mantissas)
    if direction > 0:
        return min(p for p in points if p > value and not _same(p, value))
    return max(p for p in points if p < value and not _same(p, value))


def border_extension(rows, grid, mantissas=(1, 3)):
    """('up'|'down', new rate) when the best rate lies on an end of the tried grid.

    Failed runs count as tried and worst, so a diverged extension closes the border.
    """
    ranked = rank(rows)
    if not ranked:
        return None
    tried = sorted(grid)
    best = ranked[0]['learning_rate']
    if _same(best, tried[-1]):
        return 'up', sequence_neighbor(best, +1, mantissas)
    if _same(best, tried[0]):
        return 'down', sequence_neighbor(best, -1, mantissas)
    return None


def best_neighbor(rows, grid, rate):
    """Better-ranked adjacent rate of the tried grid (failed runs rank last)."""
    tried = sorted(grid)
    index = next(i for i, value in enumerate(tried) if _same(value, rate))
    neighbors = [tried[i] for i in (index - 1, index + 1) if 0 <= i < len(tried)]
    order = [r['learning_rate'] for r in rank(rows)]
    def position(value):
        return next((i for i, v in enumerate(order) if _same(v, value)), len(order))
    return min(neighbors, key=lambda value: (position(value), value))


# ---------------------------------------------------------------- execution

def load_protocol(path):
    path = Path(path).resolve()
    protocol = json.loads(path.read_text(encoding='utf-8'))
    if (not isinstance(protocol, dict) or protocol.get('schema_version') != 2
            or not REQUIRED <= set(protocol)
            or set(protocol['stages']) != {str(i) for i in range(8)}):
        raise ValueError('Invalid v2 search protocol.')
    return protocol


def protocol_sha256(path):
    """SHA-256 of the protocol file with LF line endings.

    Git checkouts with core.autocrlf write CRLF on Windows and LF elsewhere;
    normalizing keeps the hash (and thus run reuse) identical across machines.
    """
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def materialize(config, root, seed):
    """Runner config from a portable (tp3-relative) configuration."""
    config = copy.deepcopy(config)
    config.pop('model_seed', None)
    for key in ('dataset', 'cache'):
        config[key] = str((Path(root) / config[key]).resolve())
    if 'data' in config:
        for key in ('validation_dataset', 'validation_cache'):
            config['data'][key] = str((Path(root) / config['data'][key]).resolve())
    config['model_seed'] = seed
    return config


def summarize(report, label):
    """Compact, JSON-safe row used by the stage decisions and the analysis."""
    config = {k: v for k, v in report['config'].items() if k != 'model_seed'}
    metrics = report['metrics']['validation'] if report['status'] == 'completed' else None
    return {'label': label, 'run_id': report['run_id'], 'config_id': report['config_id'],
            'seed': report['config']['model_seed'], 'status': report['status'],
            'stop_reason': report['stop_reason'], 'converged': report['stop_reason'] == 'early_stopping',
            'accuracy': metrics['accuracy'] if metrics else None,
            'loss': metrics['loss'] if metrics else None,
            'chosen_epoch': report['chosen_epoch'], 'epochs_run': report['history']['epocas_corridas'],
            'parameter_count': report['parameter_count'], 'duration_seconds': report['duration_seconds'],
            'activation': config['activation'], 'architecture': config['architecture'],
            'batch_size': config['batch_size'], 'optimizer': config['optimizer']['name'],
            'learning_rate': config['optimizer']['learning_rate'],
            'dataset_sha256': report['dataset']['sha256'], 'split_sha256': report['dataset']['split_sha256'],
            'config': config}


def _job(config, output, metadata, root, pause_after=None, verbose=0):
    """Run one configuration/seed, or reuse/resume the stored run with that id."""
    config = validate_config(config)
    data_hashes(config)
    run_id = f'{config_identity(config, path_root=root)}-seed-{config["model_seed"]}'
    directory = Path(output) / 'runs' / run_id
    resume = False
    if (directory / 'results.json').exists():
        report = json.loads((directory / 'results.json').read_text(encoding='utf-8'))
        if (report['config'] != recorded_config(config, root) or report['metadata'] != metadata
                or report['dataset']['sha256'] != sha256_file(config['dataset'])):
            raise ValueError(f'Stored run {run_id} differs from the current protocol.')
        if report['status'] != 'interrupted':
            if report['status'] == 'completed' and sha256_file(
                    directory / report['artifacts']['best_model']) != report['model_sha256']:
                raise ValueError(f'Stored model of {run_id} differs from its recorded metrics.')
            return report
        resume = True
    elif (directory / 'checkpoint.npz').exists():
        resume = True
    print(f'Running {run_id}', flush=True)
    report = run(config, Path(output) / 'runs', resume=resume, metadata=metadata, path_root=root,
                 pause_after=pause_after, verbose=verbose)
    print(f'{run_id}: {report["status"]} ({report["stop_reason"]}), '
          f'{report["history"]["epocas_corridas"]} epochs, {report["duration_seconds"]:.1f}s', flush=True)
    if report['status'] == 'interrupted':
        raise KeyboardInterrupt('Run paused; repeat the same stage command to resume.')
    return report


def execute_jobs(jobs, output, metadata, root, workers=1, pause_after=None, verbose=0):
    """Deduplicate configurations/seeds; reuse or resume exact stored runs."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer.')
    unique = {}
    for config, label in jobs:
        c = validate_config(config)
        unique.setdefault((config_identity(c, path_root=root), c['model_seed']), c)
    args = (output, metadata, root, pause_after, verbose)
    if workers == 1 or len(unique) <= 1:
        reports = {key: _job(config, *args) for key, config in unique.items()}
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {key: pool.submit(_job, config, *args) for key, config in unique.items()}
            reports = {key: future.result() for key, future in futures.items()}
    return [summarize(reports[(config_identity(validate_config(config), path_root=root),
                              config['model_seed'])], label) for config, label in jobs]
