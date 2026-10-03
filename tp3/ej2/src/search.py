"""Búsqueda por etapas de ej2, con partición común y selección sin test."""
from __future__ import annotations

import argparse
import copy
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from tps_sia.tp3.shared.experiments import fingerprint, run, sha256_file, validate_config, write_json
from .experiments import CONFIG, candidate_config, load_search

ARCHITECTURES = [[784, 64, 10], [784, 128, 10], [784, 256, 10], [784, 128, 64, 10]]
RULE = 'mean validation accuracy descending, mean validation cross-entropy ascending, parameter count ascending, config_id ascending'


def validate_rates(search, base_dir):
    configs = [validate_config(candidate_config(search, c['name'], 42, base_dir)) for c in search['candidates']]
    expected = {(name, rate) for name in ('sgd', 'momentum') for rate in (1e-5, 1e-4, 1e-3, 1e-2)}
    expected |= {('adam', rate) for rate in (1e-5, 1e-4, 1e-3)}
    if {(c['optimizer']['name'], c['optimizer']['learning_rate']) for c in configs} != expected:
        raise ValueError('Expected the predefined eleven optimizer/rate combinations.')
    common = [{k: v for k, v in c.items() if k != 'optimizer'} for c in configs]
    if any(c != common[0] for c in common[1:]):
        raise ValueError('Rate comparisons must share every setting except optimizer and learning rate.')
    base = common[0]
    if (base['architecture'] != [784, 128, 10] or base['activation'] != 'relu'
            or base['batch_size'] != 32 or base['epochs'] != 30
            or Path(base['dataset']).name != 'digits.csv'):
        raise ValueError('Step 7 requires digits.csv, [784,128,10], ReLU, batch 32 and 30 epochs.')
    for c in configs:
        if c['optimizer']['name'] == 'momentum' and c['optimizer']['momentum'] != 0.9:
            raise ValueError('The momentum comparison requires momentum 0.9.')
    baseline = validate_config(candidate_config(search, 'baseline', 42, base_dir))
    if baseline['optimizer'] != {'name': 'sgd', 'learning_rate': 0.01}:
        raise ValueError('The baseline control must use SGD with learning rate 0.01.')


def rank(reports):
    return sorted((r for r in reports if r['status'] == 'completed'), key=lambda r: (
        -r['metrics']['validation']['accuracy'], r['metrics']['validation']['loss'],
        r['parameter_count'], r['config_id']))


def confirm_groups(reports, seeds):
    """Rank complete groups on common seeds; never choose a favorable seed."""
    groups = {}
    for report in reports:
        if report['status'] != 'completed':
            raise ValueError('Confirmation requires all runs to complete successfully.')
        groups.setdefault(report['config_id'], []).append(report)
    if not groups:
        raise ValueError('No confirmation runs.')
    provenance = {(r['dataset']['sha256'], r['dataset']['split_sha256']) for r in reports}
    if len(provenance) != 1:
        raise ValueError('Confirmation requires a common dataset and partition.')
    summary = []
    for config_id, group in groups.items():
        if sorted(r['config']['model_seed'] for r in group) != sorted(seeds):
            raise ValueError('Each finalist requires exactly the common seeds.')
        entry = {'config_id': config_id, 'seeds': list(seeds),
                 'run_ids': [r['run_id'] for r in group],
                 'parameters': group[0]['parameter_count'],
                 'chosen_epochs': [r['chosen_epoch'] for r in group]}
        for key in ('accuracy', 'loss', 'macro_f1', 'balanced_accuracy'):
            values = [r['metrics']['validation'][key] for r in group]
            entry[key + '_mean'] = float(np.mean(values))
            entry[key + '_std'] = float(np.std(values))
        entry['duration_seconds_mean'] = float(np.mean([r['duration_seconds'] for r in group]))
        summary.append(entry)
    return sorted(summary, key=lambda s: (-s['accuracy_mean'], s['loss_mean'], s['parameters'], s['config_id']))


def _job(config, label, output, protocol_hash):
    config = validate_config(config)
    group = {k: v for k, v in config.items() if k not in ('model_seed', 'cache')}
    run_id = fingerprint({'config': group, 'initialization': {'mode': 'fresh'}})[:16] + f'-seed-{config["model_seed"]}'
    path = output / 'runs' / run_id / 'results.json'
    meta = {'candidate': label, 'protocol_sha256': protocol_hash}
    resume = False
    if path.exists():
        report = json.loads(path.read_text())
        if (report['config'] != config or report['metadata'] != meta
                or report['dataset']['sha256'] != sha256_file(config['dataset'])):
            raise ValueError('Stored run differs from the current experiment.')
        if report['status'] != 'interrupted':
            if report['status'] == 'completed':
                model_path = path.parent / report['artifacts']['best_model']
                if sha256_file(model_path) != report['model_sha256']:
                    raise ValueError('Stored model differs from the recorded metrics.')
            print(f'Reusing {label}, seed {config["model_seed"]}: {report["status"]}', flush=True)
            return report
        resume = True
    elif (path.parent / 'checkpoint.npz').exists():
        resume = True
    print(f'Running {label}, seed {config["model_seed"]}', flush=True)
    report = run(config, output / 'runs', resume=resume, metadata=meta)
    print(f'{label}: {report["status"]}, {report["duration_seconds"]:.1f}s', flush=True)
    if report['status'] == 'interrupted':
        raise KeyboardInterrupt('Run paused; repeat the same search command to resume.')
    return report



def execute(config_path: Path, output: Path, workers: int = 1):
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer.')
    config_path, output = config_path.resolve(), output.resolve()
    search = load_search(config_path)
    if search['seeds'] != [42, 0, 1] or len(search['candidates']) != 11:
        raise ValueError('Step 7 requires the eleven rate candidates and seeds [42, 0, 1].')
    validate_rates(search, config_path.parent)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {'schema_version': 1, 'search_sha256': sha256_file(config_path),
                'search': search, 'architectures': ARCHITECTURES, 'architecture_optimizers': 2,
                'finalists': 3, 'selection_rule': RULE, 'delivery_seed': 42,
                'delivery_policy': 'best development checkpoint; no full-data retraining in step 7'}
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('Existing search has a different protocol. Use a new output directory.')
    else:
        write_json(protocol_path, protocol)

    def job(config, label):
        return _job(config, label, output, sha256_file(protocol_path))

    def save_stage(name, reports, extra=None):
        write_json(output / f'{name}.json', {'schema_version': 1,
                   'run_ids': [r['run_id'] for r in reports], **(extra or {})})

    def jobs(pairs):
        if workers == 1:
            return [job(config, label) for config, label in pairs]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_job, config, label, output, sha256_file(protocol_path)) for config, label in pairs]
            return [future.result() for future in futures]

    base = candidate_config(search, 'baseline', 42, config_path.parent)
    pilot_config = copy.deepcopy(base)
    pilot_config['epochs'] = 2
    pilot = job(pilot_config, 'pilot')
    if pilot['status'] != 'completed':
        raise ValueError('Pilot failed; do not launch the search.')
    save_stage('pilot', [pilot], {'seconds_per_epoch': pilot['duration_seconds'] / 2,
               'estimated_rate_stage_seconds': pilot['duration_seconds'] / 2 * base['epochs'] * 11})

    rates = jobs([(candidate_config(search, c['name'], 42, config_path.parent), c['name'])
                  for c in search['candidates']])
    promising = []
    for report in rank(rates):
        if report['config']['optimizer']['name'] not in [r['config']['optimizer']['name'] for r in promising]:
            promising.append(report)
        if len(promising) == 2:
            break
    if len(promising) != 2:
        raise ValueError('Need successful runs from at least two optimizers.')
    save_stage('rates', rates, {'architecture_parent_ids': [r['run_id'] for r in promising]})
    architectures = []
    architecture_jobs = []
    for parent in promising:
        for architecture in ARCHITECTURES:
            if architecture == parent['config']['architecture']:
                architectures.append(parent)
                continue
            config = copy.deepcopy(parent['config'])
            config['architecture'] = architecture
            label = parent['metadata']['candidate'] + '-hidden-' + '-'.join(map(str, architecture[1:-1]))
            architecture_jobs.append((config, label))
    architectures += jobs(architecture_jobs)
    save_stage('architectures', architectures)
    pool = {r['config_id']: r for r in rates + architectures}
    finalists = rank(pool.values())[:3]
    baseline = next(r for r in rates if r['metadata']['candidate'] == 'baseline')
    if baseline['config_id'] not in [r['config_id'] for r in finalists]:
        finalists.append(baseline)
    # Persist adaptive decisions before launching any confirmation runs.
    save_stage('finalists', finalists, {'seeds': search['seeds'], 'rule': search['selection_rule']})
    confirmed = []
    confirmation_jobs = []
    for finalist in finalists:
        for seed in search['seeds']:
            if seed == 42:
                confirmed.append(finalist)
            else:
                config = copy.deepcopy(finalist['config'])
                config['model_seed'] = seed
                confirmation_jobs.append((config, finalist['metadata']['candidate']))
    confirmed += jobs(confirmation_jobs)
    summary = confirm_groups(confirmed, search['seeds'])
    save_stage('confirmation', confirmed, {'ranking': summary, 'rule': RULE})
    winner = summary[0]
    candidate = next(r for r in confirmed if r['config_id'] == winner['config_id'] and r['config']['model_seed'] == 42)
    model_path = Path(candidate['run_directory']) / candidate['artifacts']['best_model']
    selection = {'schema_version': 1, 'selection_rule': RULE, 'config': candidate['config'],
                 'seeds': search['seeds'], 'delivery_seed': 42, 'config_id': winner['config_id'],
                 'protocol_sha256': sha256_file(protocol_path), 'dataset': candidate['dataset'],
                 'candidate_model': str(model_path.relative_to(output)), 'model_sha256': sha256_file(model_path),
                 'delivery_policy': protocol['delivery_policy'], 'candidate_epoch': candidate['chosen_epoch'],
                 'retraining_epochs': int(np.median(winner['chosen_epochs'])),
                 'retraining_epoch_rule': 'median best validation-loss epoch across common seeds',
                 'evidence': {'rate_stage': 'rates.json', 'architecture_stage': 'architectures.json',
                              'confirmation_stage': 'confirmation.json', 'ranking': summary},
                 'limitations': ['Validation has no digit 8; its recall is not evaluable.',
                                 'Repeated validation selection can be optimistic; final test remains reserved.']}
    write_json(output / 'selection.json', selection)
    print(f'Selected {winner["config_id"]}: mean accuracy {winner["accuracy_mean"]:.6f}; {model_path}', flush=True)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1, help='Corridas independientes simultáneas (default: 1)')
    args = parser.parse_args()
    execute(args.config, args.output_dir, args.workers)


if __name__ == '__main__':
    main()
