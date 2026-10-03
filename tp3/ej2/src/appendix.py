"""Appendix to the ej2 v2 search: post-hoc checks that cannot change the selection.

Part A (relu): the selected configuration with activation relu, seeds 0, 1, 2, 3, 42,
compared with the selected tanh configuration by the stage-7 significance rule.
Part B (stages_3_5): the factors of stages 3-5 (width, depth, batch size) around their
incumbent, with seeds 0, 1, 42 instead of seed 42 alone.

Runs already stored in results/v2/runs are reused read-only by config_id; missing runs
are trained into results/v2_appendix/<part>/runs. Nothing in results/v2 is written.
The reserved test file is never read: the shared runner rejects it before loading data.

    python -m tps_sia.tp3.ej2.src.appendix run --part relu [--workers 4]
    python -m tps_sia.tp3.ej2.src.appendix run --part stages_3_5 [--workers 4]
    python -m tps_sia.tp3.ej2.src.appendix report
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tps_sia.tp3.shared.experiments import (config_identity, recorded_config, sha256_file,
                                            validate_config, write_json)
from tps_sia.tp3.ej2.src.staged_search import (CONFIG, TP3, _job, compare_groups, group_summary,
                                               load_protocol, materialize, summarize)

V2 = TP3 / 'ej2' / 'results' / 'v2'
OUTPUT = TP3 / 'ej2' / 'results' / 'v2_appendix'
RELU_SEEDS = [42, 0, 1, 2, 3]
STAGE_SEEDS = [42, 0, 1]


# ---------------------------------------------------------------- run reuse

def stored_report(directory: Path, config: dict, root: Path, dataset_sha: str):
    """Completed report stored for this exact configuration, or None.

    A stored run with the same id but another configuration, dataset or model file
    is an error, never a silent reuse.
    """
    path = directory / 'results.json'
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding='utf-8'))
    if report['status'] != 'completed':
        return None
    if report['config'] != recorded_config(config, root) or report['dataset']['sha256'] != dataset_sha:
        raise ValueError(f'Stored run {directory.name} differs from the requested configuration.')
    if sha256_file(directory / report['artifacts']['best_model']) != report['model_sha256']:
        raise ValueError(f'Stored model of {directory.name} differs from its recorded metrics.')
    return report


def execute(jobs, reuse_runs: Path, output: Path, metadata: dict, root: Path = TP3, workers: int = 1):
    """Rows for (config, label) jobs: reused from reuse_runs, else run or resumed in output."""
    rows, pending, hashes = [None] * len(jobs), {}, {}
    for index, (config, label) in enumerate(jobs):
        c = validate_config(config)
        run_id = f'{config_identity(c, path_root=root)}-seed-{c["model_seed"]}'
        dataset_sha = hashes.setdefault(c['dataset'], sha256_file(c['dataset']))
        report = stored_report(Path(reuse_runs) / run_id, c, root, dataset_sha)
        if report is not None:
            rows[index] = {**summarize(report, label), 'origin': 'v2'}
        else:
            origin = 'appendix_reused' if (Path(output) / 'runs' / run_id / 'results.json').exists() else 'appendix_new'
            pending.setdefault(run_id, (c, []))[1].append((index, label, origin))
    args = (Path(output), metadata, Path(root))
    if workers == 1 or len(pending) <= 1:
        reports = {run_id: _job(c, *args) for run_id, (c, _) in pending.items()}
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {run_id: pool.submit(_job, c, *args) for run_id, (c, _) in pending.items()}
            reports = {run_id: future.result() for run_id, future in futures.items()}
    for run_id, (_, targets) in pending.items():
        for index, label, origin in targets:
            rows[index] = {**summarize(reports[run_id], label), 'origin': origin}
    return rows


# ---------------------------------------------------------------- part A

def relu_jobs(selection: dict, root: Path = TP3, seeds=RELU_SEEDS):
    """Selected tanh configuration and the same with relu, for every seed."""
    base = {k: v for k, v in selection['config'].items() if k != 'model_seed'}
    relu = {**copy.deepcopy(base), 'activation': 'relu'}
    return ([(materialize(base, root, s), 'tanh (seleccionada)') for s in seeds]
            + [(materialize(relu, root, s), 'relu') for s in seeds])


def compare_relu(rows, seeds, multiplier):
    groups = group_summary(rows, seeds)
    tanh = next(g for g in groups if g['config']['activation'] == 'tanh')
    relu = next(g for g in groups if g['config']['activation'] == 'relu')
    if relu['status'] != 'completed':
        return groups, None
    leader, other = (relu, tanh) if groups.index(relu) < groups.index(tanh) else (tanh, relu)
    return groups, compare_groups(leader, other, multiplier)


# ---------------------------------------------------------------- part B

FACTORS = (('width', 3, 'architecture'), ('depth', 4, 'architecture'), ('batch_size', 5, 'batch_size'))


def factor_variants(protocol: dict, name: str, width: int):
    """([(value, label)], incumbent value) exactly as stages 3, 4 and 5 build them.

    width is the hidden width of the stage base (only used by depth, as in stage 4).
    """
    stages = protocol['stages']
    if name == 'width':
        return [([784, w, 10], f'width-{w}') for w in stages['3']['widths']], [784, stages['3']['incumbent'], 10]
    if name == 'depth':
        variants = []
        for divisors in stages['4']['depth_divisors']:
            if any(width % d for d in divisors):
                raise ValueError(f'Width {width} is not divisible by {divisors}.')
            hidden = [width // d for d in divisors]
            variants.append(([784, *hidden, 10], 'hidden-' + '-'.join(map(str, hidden))))
        return variants, [784, *[width // d for d in stages['4']['incumbent']], 10]
    if name == 'batch_size':
        return [(b, f'batch-{b}') for b in stages['5']['batch_sizes']], stages['5']['incumbent']
    raise ValueError(f'Unknown factor {name}.')


def factor_summary(rows, original_rows, incumbent_value, key, seeds, multiplier):
    """3-seed groups, seed-42 originals and the change caused by adding seeds."""
    groups = group_summary(rows, seeds)
    original = {r['config_id']: r for r in original_rows}
    for group in groups:
        first = original.get(group['config_id'])
        group['value'] = group['config'][key]
        group['seed_42_original'] = None if first is None else {
            'accuracy': first['accuracy'], 'loss': first['loss'], 'run_id': first['run_id']}
        group['difference_vs_seed_42'] = (None if first is None or group['status'] != 'completed'
                                          else group['accuracy_mean'] - first['accuracy'])
    completed = [g for g in groups if g['status'] == 'completed']
    incumbent = next(g for g in groups if g['value'] == incumbent_value)
    best_seed_42 = min((r for r in original_rows if r['status'] == 'completed'),
                       key=lambda r: (-r['accuracy'], r['loss'], r['parameter_count'], r['config_id']))
    best = completed[0]
    return {
        'groups': groups,
        'incumbent_config_id': incumbent['config_id'],
        'best_3_seeds_config_id': best['config_id'],
        'best_seed_42_config_id': best_seed_42['config_id'],
        'same_winner': best['config_id'] == best_seed_42['config_id'],
        'best_vs_incumbent': (None if best is incumbent else compare_groups(best, incumbent, multiplier)),
    }


# ---------------------------------------------------------------- commands

def run_part(part: str, v2: Path = V2, output: Path = OUTPUT, workers: int = 4, root: Path = TP3,
             config_path: Path = CONFIG):
    protocol = load_protocol(config_path)
    multiplier = protocol['improvement_rule']['sigma_multiplier']
    selection = json.loads((v2 / 'selection.json').read_text(encoding='utf-8'))
    metadata = {'protocol': protocol['protocol'], 'search_sha256': selection['search_sha256'],
                'appendix': part}
    directory = output / part
    directory.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if part == 'relu':
        rows = execute(relu_jobs(selection, root), v2 / 'runs', directory, metadata, root, workers)
        groups, comparison = compare_relu(rows, RELU_SEEDS, multiplier)
        if any(r['origin'] != 'v2' for r in rows if r['activation'] == 'tanh'):
            raise ValueError('The selected tanh runs must come from results/v2.')
        result = {'seeds': RELU_SEEDS, 'groups': groups, 'relu_vs_tanh': comparison}
    elif part == 'stages_3_5':
        factors = {}
        for name, stage, key in FACTORS:
            # Same base as the original stage: the winner of the previous stage.
            base = json.loads((v2 / f'stage_{stage - 1}.json').read_text(encoding='utf-8'))['winner']['config']
            variants, incumbent = factor_variants(protocol, name, base['architecture'][1])
            jobs = [(materialize({**base, key: value}, root, seed), label)
                    for value, label in variants for seed in STAGE_SEEDS]
            rows = execute(jobs, v2 / 'runs', directory, metadata, root, workers)
            original = json.loads((v2 / f'stage_{stage}.json').read_text(encoding='utf-8'))['runs']
            order = list(dict.fromkeys(r['config_id'] for r in rows))
            factors[name] = {'stage': stage, 'key': key, 'base_config': base, 'order': order,
                             **factor_summary(rows, original, incumbent, key, STAGE_SEEDS, multiplier),
                             'runs': rows}
        result = {'seeds': STAGE_SEEDS, 'factors': factors}
    else:
        raise ValueError(f'Unknown part {part}.')
    runs = rows if part == 'relu' else [r for f in result['factors'].values() for r in f['runs']]
    write_json(directory / 'summary.json', {
        'schema_version': 1, 'part': part, 'protocol': protocol['protocol'],
        'search_sha256': selection['search_sha256'], 'selected_config_id': selection['config_id'],
        'note': 'Post-hoc appendix; it cannot change selection.json.',
        'wall_seconds': time.perf_counter() - start,
        'runs_reused_from_v2': sum(r['origin'] == 'v2' for r in runs),
        'runs_trained_or_resumed': sum(r['origin'] != 'v2' for r in runs),
        **({'runs': rows} if part == 'relu' else {}), **result})
    return directory / 'summary.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--part', choices=('relu', 'stages_3_5'), required=True)
    run_parser.add_argument('--workers', type=int, default=4)
    sub.add_parser('report')
    for p in (run_parser, sub.choices['report']):
        p.add_argument('--v2-dir', type=Path, default=V2)
        p.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    if args.command == 'run':
        path = run_part(args.part, args.v2_dir.resolve(), args.output_dir.resolve(), args.workers)
        print(f'Summary: {path}')
    else:
        from tps_sia.tp3.ej2.src.appendix_report import write_report
        print(f'Report: {write_report(args.output_dir.resolve())}')


if __name__ == '__main__':
    main()
