"""Execute one agreed comparison stage using the public training workflow."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import time

import numpy as np
from .dataset import DEFAULT_DATASET, load_dataset
from .train import run_training, write_json, model_identity


METRICS = ['max_pixel_errors', 'mean_pixel_errors', 'mse_per_pixel',
           'exact_fraction', 'within_tolerance_fraction', 'epochs_completed', 'duration_seconds']


def _protocol(path):
    protocol = json.loads(Path(path).read_text(encoding='utf-8'))
    if set(protocol) - {'reuse_runs'} != {'stage', 'approval', 'seeds', 'selection', 'variants'}:
        raise ValueError('Protocol requires stage, approval, seeds, selection and variants')
    if not isinstance(protocol['approval'], str) or not protocol['approval'].strip():
        raise ValueError('Record the user decision authorizing this exact protocol in approval')
    if protocol['selection'] != 'max_mean_mse':
        raise ValueError('Supported selection: max_mean_mse (then deterministic run ID tie break)')
    seeds = protocol['seeds']
    if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be distinct integers shared by all variants')
    variants = protocol['variants']
    if not variants:
        raise ValueError('At least one variant is required')
    names = []
    for variant in variants:
        if set(variant) != {'name', 'architecture', 'optimizer', 'batch_size', 'max_epochs',
                            'evaluate_every', 'stop_on_tolerance'}:
            raise ValueError('Each variant requires name and all training settings except seed')
        name = variant['name']
        if not isinstance(name, str) or not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', name):
            raise ValueError('Variant names must be lowercase kebab-case')
        architecture = variant['architecture']
        if (not isinstance(architecture, list) or len(architecture) < 3
                or len(architecture) % 2 != 1 or any(type(n) is not int or n < 1 for n in architecture)
                or architecture[0] != 35 or architecture[-1] != 35
                or architecture[len(architecture)//2] != 2):
            raise ValueError('Variants require 35 inputs/outputs and a two-dimensional central code')
        for key in ['batch_size', 'max_epochs', 'evaluate_every']:
            if type(variant[key]) is not int or variant[key] < 1:
                raise ValueError(f'{key} must be a positive integer')
        if type(variant['stop_on_tolerance']) is not bool:
            raise ValueError('stop_on_tolerance must be boolean')
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError('Variant names must be unique')
    reuse = protocol.get('reuse_runs', {})
    expected_ids = {f'{name}-seed-{seed}' for name in names for seed in seeds}
    if not isinstance(reuse, dict) or set(reuse) - expected_ids:
        raise ValueError('reuse_runs must map configured run IDs to existing run directories')
    return protocol


def _csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(protocol, rows, output):
    aggregates = []
    for variant in protocol['variants']:
        group = [row for row in rows if row['variant'] == variant['name']]
        complete = [row for row in group if row['status'] == 'completed']
        aggregate = {'variant': variant['name'], 'runs': len(group), 'completed': len(complete),
                     'failed': sum(row['status'] == 'failed' for row in group),
                     'interrupted': sum(row['status'] == 'interrupted' for row in group),
                     'not_run': sum(row['status'] == 'not_run' for row in group),
                     'success_fraction': sum(row.get('all_within_tolerance', False) for row in complete) / len(group)}
        for metric in METRICS:
            values = [row[metric] for row in complete]
            aggregate[metric + '_mean'] = float(np.mean(values)) if values else None
            aggregate[metric + '_std'] = float(np.std(values)) if values else None
        aggregates.append(aggregate)
    # An incomplete variant is not eligible to win an across-seed comparison.
    comparable = [group for group in aggregates if group['completed'] == group['runs']]
    winner = min(comparable, key=lambda group: (group['max_pixel_errors_mean'],
                 group['mean_pixel_errors_mean'], group['mse_per_pixel_mean'], group['variant'])) if comparable else None
    eligible = [row for row in rows if winner and row['variant'] == winner['variant']]
    chosen = min(eligible, key=lambda row: (row['max_pixel_errors'], row['mean_pixel_errors'],
                                          row['mse_per_pixel'], row['run_id'])) if eligible else None
    selected = None
    if chosen:
        source = Path(chosen['source_run'])
        manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
        selected = {'run_id': chosen['run_id'], 'model': manifest['model'],
                    'config': str((source/'config.json').resolve()),
                    'seed': chosen['seed'], 'criterion': protocol['selection'],
                    'all_within_tolerance': chosen['all_within_tolerance']}
    summary = {'stage': protocol['stage'], 'approval': protocol['approval'],
               'budget': {'runs': len(rows), 'max_epochs_total': sum(
                   v['max_epochs'] for v in protocol['variants'] for seed in protocol['seeds']
                   if f"{v['name']}-seed-{seed}" not in protocol.get('reuse_runs', {}))},
               'runs': rows, 'aggregates': aggregates, 'selected': selected,
               'selected_variant': winner['variant'] if winner else None}
    write_json(output/'summary.json', summary)
    _csv(output/'runs.csv', rows)
    _csv(output/'aggregates.csv', aggregates)
    from .plots import plt
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    complete = [row for row in rows if row['status'] == 'completed']
    for axis, metric in zip(axes.flat, ['max_pixel_errors', 'mean_pixel_errors', 'mse_per_pixel',
                                       'within_tolerance_fraction', 'epochs_completed', 'duration_seconds']):
        axis.bar(range(len(complete)), [row[metric] for row in complete])
        axis.set_xticks(range(len(complete)), [row['run_id'] for row in complete], rotation=60, ha='right', fontsize=7)
        axis.set_title(metric)
    figure.suptitle(f"{protocol['stage']} — {len(complete)}/{len(rows)} completed; other statuses in runs.csv")
    figure.tight_layout()
    figure.savefig(output/'comparison.png', dpi=120)
    plt.close(figure)
    return summary


def run_comparison(protocol_path, output, dataset_path=DEFAULT_DATASET):
    """Run only the explicit variants/seeds in one stage; never expand its budget."""
    protocol = _protocol(protocol_path)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Comparison output must be empty to preserve previous runs')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'protocol.json', protocol)
    write_json(output/'environment.json', {'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS'),
                                         'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS')})
    rows = []
    interrupted = False
    for variant in protocol['variants']:
        for seed in protocol['seeds']:
            run_id = f"{variant['name']}-seed-{seed}"
            run = output / run_id
            run.mkdir()
            config = {key: value for key, value in variant.items() if key != 'name'}
            config['seed'] = seed
            config_path = output / f'{run_id}.json'
            write_json(config_path, config)
            row = {'run_id': run_id, 'variant': variant['name'], 'seed': seed,
                   'config': str(config_path.resolve()), 'status': 'not_run',
                   'source_run': str(run.resolve()), 'reused': False}
            started = time.perf_counter()
            if not interrupted:
                try:
                    reuse = protocol.get('reuse_runs', {}).get(run_id)
                    if reuse:
                        source = Path(reuse).resolve()
                        manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
                        if (manifest['config'] != config
                                or manifest['dataset']['sha256'] != load_dataset(dataset_path).sha256):
                            raise ValueError('Reused run must match the exact configuration and dataset')
                        result = {'manifest': manifest, 'metrics': json.loads((source/'metrics.json').read_text(encoding='utf-8'))}
                        if not Path(manifest['model']['path']).is_file():
                            raise ValueError('Reused model artifact is missing')
                        if model_identity(manifest['model']['path'])['sha256'] != manifest['model']['sha256']:
                            raise ValueError('Reused model artifact hash does not match manifest')
                        row.update(source_run=str(source), reused=True)
                    else:
                        result = run_training(config_path, run, dataset_path)
                        result['manifest']['thread_environment'] = {
                            'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS'),
                            'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS')}
                        write_json(run/'manifest.json', result['manifest'])
                    reason = result['manifest']['stop_reason']
                    row.update(result['metrics'], epochs_completed=result['manifest']['epochs_completed'],
                               duration_seconds=result['manifest']['duration_seconds'], stop_reason=reason)
                    row['status'] = 'interrupted' if reason == 'interrupted' else ('failed' if reason == 'numerical_failure' else 'completed')
                    interrupted = row['status'] == 'interrupted'
                except KeyboardInterrupt:
                    row.update(status='interrupted', error='KeyboardInterrupt')
                    interrupted = True
                except Exception as error:
                    row.update(status='failed', error=f'{type(error).__name__}: {error}')
            row.setdefault('duration_seconds', time.perf_counter()-started)
            write_json(run/'comparison-status.json', row)
            rows.append(row)
            # Preserve progress after each attempt, including failures and interrupted runs.
            write_json(output/'progress.json', rows)
    return _summarize(protocol, rows, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    try:
        result = run_comparison(args.protocol, args.output, args.dataset)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result['selected']))


if __name__ == '__main__':
    main()
