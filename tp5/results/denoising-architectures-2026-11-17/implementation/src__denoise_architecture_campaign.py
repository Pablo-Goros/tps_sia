"""Authorized 3-depth x 4-code-width x 3-seed denoising comparison.

Reuses the public epoch trainer; leaves the original campaign untouched.
Selects using the original validation set, then evaluates on fresh corruptions.
"""
import argparse
import copy
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np

from .autoencoder import Autoencoder
from .dataset import load_dataset
from .denoise_campaign import (read_json, sha256, verified_campaign,
                               verify_collections, load_collection)
from .denoise_evaluation import grouped_metrics, write_csv
from .denoise_training import train_run, select_runs, validation_score
from .noise import make_collection
from .train import write_json


TP5_ROOT = Path(__file__).resolve().parents[1]
ENCODERS = {'simple': [35, 40, 20], 'current': [35, 40, 20, 10],
            'deeper': [35, 40, 30, 20, 10]}
DIMENSIONS = [5, 10, 15, 20]
SEEDS = [42, 43, 44]


def variants(config, source_model):
    """Resolve the exact design agreed on 2026-11-17, without changing training."""
    if (set(config) != {'source_campaign', 'seeds', 'dimensions', 'encoders',
                       'max_epochs', 'evaluate_every', 'final_seed'}
            or config['encoders'] != ENCODERS or config['dimensions'] != DIMENSIONS
            or config['seeds'] != SEEDS or config['max_epochs'] != 3000
            or config['evaluate_every'] != 100 or config['final_seed'] != 20261117):
        raise ValueError('Configuration differs from the authorized architecture comparison')
    if source_model['architecture'] != [35, 40, 20, 10, 2, 10, 20, 40, 35]:
        raise ValueError('Source is not the agreed current architecture')
    result = {}
    for family, encoder in config['encoders'].items():
        for dimension in config['dimensions']:
            model = copy.deepcopy(source_model)
            model['architecture'] = encoder + [dimension] + encoder[::-1]
            result[f'{family}-d{dimension}'] = model
    return result


def run_config(variant, model, seed, manifest):
    return {'variant': variant, 'seed': seed, 'model': model,
            'max_epochs': manifest['config']['max_epochs'],
            'evaluate_every': manifest['config']['evaluate_every'],
            'training_levels': manifest['noise']['training_levels'],
            'selection_levels': manifest['noise']['selection_levels'],
            'dataset_sha256': manifest['dataset']['sha256'],
            'validation_sha256': manifest['validation']['sha256']}


def verify_reuse(directory, expected, row, validation):
    """Check protocol, persisted checkpoints, history and recalculated validation."""
    directory = Path(directory)
    original = read_json(directory/'config.json')
    if {k: v for k, v in original.items() if k != 'variant'} != {
            k: v for k, v in expected.items() if k != 'variant'}:
        raise ValueError('Reused run protocol does not match')
    if (row['status'] != 'completed' or row['epochs_completed'] != expected['max_epochs']
            or read_json(directory/'manifest.json') != row):
        raise ValueError('Reused run is incomplete or its manifest differs')
    for entry, filename in [(row['best']['model'], 'best_model.npz'),
                            (row['last_model'], 'last_model.npz')]:
        if sha256(directory/filename) != entry['sha256']:
            raise ValueError('Reused checkpoint hash mismatch')
    model = Autoencoder.load(directory/'best_model.npz')
    effective = Autoencoder(**expected['model'], seed=expected['seed']).configuration
    if model.configuration != effective or row['effective_model'] != effective:
        raise ValueError('Reused model configuration mismatch')
    score = validation_score(grouped_metrics(validation, model.reconstruct(validation['inputs'])),
                             expected['selection_levels'])
    np.testing.assert_allclose(score, row['best']['score'], rtol=0, atol=1e-14)
    history = read_json(directory/'history.json')
    best = min(history['validation'], key=lambda point: (point['score'], point['epoch']))
    if (best['epoch'] != row['best']['epoch'] or best['score'] != row['best']['score']
            or model.history['epocas_corridas'] != best['epoch']
            or Autoencoder.load(directory/'last_model.npz').history['epocas_corridas'] != expected['max_epochs']):
        raise ValueError('Reused history or checkpoint epoch mismatch')
    return {'original_variant': original['variant'], 'validation_recomputed': score,
            'files': {p.name: sha256(p) for p in sorted(directory.iterdir()) if p.is_file()}}


def prepare(config_path, output):
    config = read_json(config_path)
    source = TP5_ROOT/config['source_campaign']
    original = verified_campaign(source)
    collections = verify_collections(source)
    models = variants(config, original['variants']['winner'])
    versions = {'python': platform.python_version(), 'numpy': np.__version__}
    source_training = read_json(source/'training.json')
    if source_training['status'] != 'completed' or source_training['versions'] != versions:
        raise ValueError('Reuse requires the completed source campaign and identical Python/NumPy versions')
    output = Path(output)
    if output.exists():
        raise ValueError('Campaign output must be new')
    # Verify all reusable runs before creating output or consuming any training budget.
    validation = load_collection(source, 'validation')
    validation = {key: values[validation['levels'] != .3] for key, values in validation.items()}
    manifest = {'schema_version': 1, 'config': config, 'variants': models,
                'noise': original['protocol']['noise'], 'dataset': original['dataset'],
                'validation': collections['collections']['validation'], 'versions': versions,
                'budget': {'total_runs': 36, 'reused_runs': 3, 'new_runs': 33,
                           'max_new_epochs': 99000, 'represented_epochs': 108000},
                'source': {'path': config['source_campaign'], 'files': {}},
                'reuse': {}, 'final_seed': config['final_seed'],
                'selection_policy': 'mean validation errors across seeds, then mean MSE; exact ties: variant order',
                'limitations': ['depth also changes parameter count',
                               'current and deeper retain width-10 layers next to the code',
                               'known characters only; fresh noise is not new-character generalization']}
    for seed in SEEDS:
        expected = run_config('current-d10', models['current-d10'], seed, manifest)
        directory = source/f'runs/latent-10-seed-{seed}'
        row = read_json(directory/'manifest.json')
        manifest['reuse'][f'current-d10-seed-{seed}'] = {
            'source_directory': f'runs/latent-10-seed-{seed}',
            **verify_reuse(directory, expected, row, validation)}
    output.mkdir(parents=True)
    shutil.copytree(source/'reference', output/'reference')
    (output/'collections').mkdir()
    shutil.copyfile(source/'collections/validation.npz', output/'collections/validation.npz')
    manifest['validation']['path'] = 'collections/validation.npz'
    # Freeze provenance, implementation and all original campaign file hashes.
    source_files = {p.relative_to(source).as_posix(): sha256(p)
                    for p in sorted(source.rglob('*')) if p.is_file()}
    manifest['source']['files'] = source_files
    (output/'implementation').mkdir()
    for name in ['src/denoise_architecture_campaign.py', 'src/denoise_training.py',
                 'src/denoise_evaluation.py', 'src/noise.py', 'src/autoencoder.py',
                 'src/dataset.py', '../tp3/shared/mlp.py', '../tp3/shared/optimizers.py']:
        target = output/'implementation'/name.replace('../', '').replace('/', '__')
        shutil.copyfile(TP5_ROOT/name, target)
    manifest['frozen_files'] = {
        p.relative_to(output).as_posix(): sha256(p)
        for folder in ['reference', 'collections', 'implementation']
        for p in sorted((output/folder).rglob('*')) if p.is_file()}
    manifest['thread_environment'] = {k: os.environ.get(k) for k in
                                      ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']}
    write_json(output/'manifest.json', manifest)
    write_json(output/'protocol.json', config)
    return manifest


def verified_manifest(campaign):
    campaign = Path(campaign)
    manifest = read_json(campaign/'manifest.json')
    source = read_json(campaign/'reference/config.json')
    original_model = {key: source[key] for key in ['architecture', 'optimizer', 'loss',
                                                   'latent_activation', 'lr_scheduler', 'batch_size']}
    if variants(manifest['config'], original_model) != manifest['variants']:
        raise ValueError('Variants disagree with the frozen design')
    for name, digest in manifest['frozen_files'].items():
        if sha256(campaign/name) != digest:
            raise ValueError(f'Frozen file hash mismatch: {name}')
    if sha256(campaign/'collections/validation.npz') != manifest['validation']['sha256']:
        raise ValueError('Validation hash mismatch')
    return manifest


def train(campaign):
    campaign = Path(campaign)
    manifest = verified_manifest(campaign)
    if (campaign/'training.json').exists() or (campaign/'runs').exists():
        raise ValueError('Campaign already started; no automatic retries or extra training')
    if manifest['versions'] != {'python': platform.python_version(), 'numpy': np.__version__}:
        raise ValueError('Training environment changed after preparation')
    clean = load_dataset(campaign/'reference/font.h').vectors
    with np.load(campaign/'collections/validation.npz', allow_pickle=False) as saved:
        validation = {key: saved[key][saved['levels'] != .3] for key in saved.files}
    source = TP5_ROOT/manifest['source']['path']
    # A second reuse check prevents changes between prepare and train.
    for name, entry in manifest['reuse'].items():
        directory = source/entry['source_directory']
        seed = int(name.rsplit('-', 1)[1])
        verified = verify_reuse(directory, run_config('current-d10', manifest['variants']['current-d10'],
                                                      seed, manifest), read_json(directory/'manifest.json'), validation)
        if verified['files'] != entry['files']:
            raise ValueError('Source run changed since preparation')
    (campaign/'runs').mkdir()
    started = time.perf_counter()
    ledger = {'status': 'running', 'runs': [], 'new_epochs_completed': 0,
              'budget': manifest['budget'], 'manifest_sha256': sha256(campaign/'manifest.json')}
    write_json(campaign/'training.json', ledger)
    try:
        for variant, model in manifest['variants'].items():
            for seed in SEEDS:
                name = f'{variant}-seed-{seed}'
                directory = campaign/'runs'/name
                config = run_config(variant, model, seed, manifest)
                if name in manifest['reuse']:
                    entry = manifest['reuse'][name]
                    shutil.copytree(source/entry['source_directory'], directory)
                    row = read_json(directory/'manifest.json')
                    shutil.copyfile(directory/'config.json', directory/'source-config.json')
                    shutil.copyfile(directory/'manifest.json', directory/'source-manifest.json')
                    row.update(run_id=name, variant=variant, reused=True, provenance=entry,
                               new_epochs_completed=0)
                    write_json(directory/'config.json', config)
                    print(f'REUSED {name}', flush=True)
                else:
                    row = train_run(config, clean, validation, directory)
                    row.update(reused=False, new_epochs_completed=row['epochs_completed'])
                for key, filename in [('last_model', 'last_model.npz')]:
                    row[key] = {'path': f'runs/{name}/{filename}', 'sha256': sha256(directory/filename)}
                row['best']['model'] = {'path': f'runs/{name}/best_model.npz',
                                        'sha256': sha256(directory/'best_model.npz')}
                write_json(directory/'manifest.json', row)
                ledger['runs'].append(row)
                ledger['new_epochs_completed'] += row['new_epochs_completed']
                write_json(campaign/'training.json', ledger)
                if row['status'] != 'completed' or row['epochs_completed'] != 3000:
                    raise RuntimeError(f'Incomplete run: {name}')
                print(f'COMPLETED {len(ledger["runs"])}/36 new_epochs={ledger["new_epochs_completed"]}', flush=True)
        selection = select_runs(ledger['runs'], list(manifest['variants']), SEEDS)
        selection.update(runs=ledger['runs'], manifest_sha256=ledger['manifest_sha256'])
        write_json(campaign/'selection.json', selection)
        ledger['status'] = 'completed'
    except BaseException as error:
        ledger['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        ledger['error'] = f'{type(error).__name__}: {error}'
        ledger['runs'] = [read_json(p) for p in sorted((campaign/'runs').glob('*/manifest.json'))]
        ledger['new_epochs_completed'] = sum(0 if row.get('reused') else row['epochs_completed']
                                              for row in ledger['runs'])
        raise
    finally:
        ledger['duration_seconds'] = time.perf_counter()-started
        write_json(campaign/'training.json', ledger)
    summarize(campaign)
    return selection


def summarize(campaign):
    campaign = Path(campaign)
    manifest = verified_manifest(campaign)
    selection = read_json(campaign/'selection.json')
    runs = selection['runs']
    rows = []
    for variant, model in manifest['variants'].items():
        group = [r for r in runs if r['variant'] == variant]
        errors = [r['best']['score'][0] for r in group]
        architecture = model['architecture']
        rows.append({'variant': variant, 'family': variant.split('-')[0],
                     'dimension': architecture[len(architecture)//2],
                     'parameters': sum((a+1)*b for a, b in zip(architecture, architecture[1:])),
                     'mean_pixel_errors': float(np.mean(errors)),
                     'std_population': float(np.std(errors)), 'min': min(errors), 'max': max(errors),
                     'mean_mse': float(np.mean([r['best']['score'][1] for r in group])),
                     'best_epochs': '/'.join(str(r['best']['epoch']) for r in group)})
    rows.sort(key=lambda r: (r['mean_pixel_errors'], r['mean_mse'], list(manifest['variants']).index(r['variant'])))
    write_csv(campaign/'validation-aggregates.csv', rows)
    write_csv(campaign/'validation-runs.csv', [
        {'run_id': r['run_id'], 'variant': r['variant'], 'seed': r['seed'], 'reused': r['reused'],
         'best_epoch': r['best']['epoch'], 'mean_pixel_errors': r['best']['score'][0],
         'mse_per_pixel': r['best']['score'][1], 'historical_duration_seconds': r['duration_seconds']}
        for r in runs])
    return rows


def verify_completed(campaign):
    campaign = Path(campaign)
    manifest = verified_manifest(campaign)
    ledger = read_json(campaign/'training.json')
    selection = read_json(campaign/'selection.json')
    expected = select_runs(ledger['runs'], list(manifest['variants']), SEEDS)
    if (ledger['status'] != 'completed' or ledger['new_epochs_completed'] != 99000
            or len(ledger['runs']) != 36 or sum(r['reused'] for r in ledger['runs']) != 3
            or selection['run_id'] != expected['run_id'] or selection['variant'] != expected['variant']
            or selection['aggregates'] != expected['aggregates'] or selection['runs'] != ledger['runs']
            or ledger['manifest_sha256'] != sha256(campaign/'manifest.json')
            or selection['manifest_sha256'] != ledger['manifest_sha256']):
        raise ValueError('Incomplete campaign or selection mismatch')
    for row in ledger['runs']:
        if row['status'] != 'completed' or row['epochs_completed'] != 3000:
            raise ValueError('Incomplete training run')
        config = run_config(row['variant'], manifest['variants'][row['variant']], row['seed'], manifest)
        if read_json(campaign/'runs'/row['run_id']/'config.json') != config:
            raise ValueError('Run configuration mismatch')
        effective = Autoencoder(**config['model'], seed=row['seed']).configuration
        for entry in [row['best']['model'], row['last_model']]:
            if sha256(campaign/entry['path']) != entry['sha256']:
                raise ValueError('Checkpoint hash mismatch')
            if Autoencoder.load(campaign/entry['path']).configuration != effective:
                raise ValueError('Checkpoint configuration mismatch')
    return manifest, ledger, selection


def evaluate(campaign, output=None):
    campaign = Path(campaign)
    manifest, ledger, selection = verify_completed(campaign)
    output = Path(output) if output else campaign/'evaluation'
    if output.exists():
        raise ValueError('Evaluation output must be new')
    output.mkdir(parents=True)
    # Generate only AFTER selection.json is frozen; never use final data for ranking.
    collection = make_collection(load_dataset(campaign/'reference/font.h').vectors,
                                 manifest['noise']['evaluation_levels'],
                                 manifest['noise']['realizations_per_character'], manifest['final_seed'])
    np.savez_compressed(output/'collection.npz', **collection)
    examples = [int(np.flatnonzero((collection['levels'] == p) & (collection['character_indices'] == c))[0])
                for p in manifest['noise']['evaluation_levels'] for c in [0, 1, 8, 16, 24, 31]]
    models = [('ae-reference', campaign/'reference/model.npz')]+[
        (r['run_id'], campaign/r['best']['model']['path']) for r in ledger['runs']]
    selected_names = {r['run_id'] for r in ledger['runs']
                      if r['variant'] in [selection['variant'], 'current-d10']}
    rows = [{'model': 'noisy-input', **r} for r in grouped_metrics(collection, collection['inputs'])]
    character_rows = []
    artifacts = {}
    for name, path in models:
        model = Autoencoder.load(path)
        predictions = model.reconstruct(collection['inputs'])
        rows.extend({'model': name, **r} for r in grouped_metrics(collection, predictions))
        character_rows.extend({'model': name, **r} for r in grouped_metrics(collection, predictions, by_character=True))
        if name in selected_names or name == 'ae-reference':
            np.savez_compressed(output/f'{name}.npz', predictions=predictions,
                                codes=model.encode(collection['inputs']))
            artifacts[name] = sha256(output/f'{name}.npz')
    aggregates = []
    for variant in manifest['variants']:
        names = {r['run_id'] for r in ledger['runs'] if r['variant'] == variant}
        for level in manifest['noise']['evaluation_levels']:
            group = [r for r in rows if r['model'] in names and r['level'] == level]
            aggregates.append({'variant': variant, 'level': level,
                **{f'{key}_{stat}': float(function([r[key] for r in group]))
                   for key in ['mean_pixel_errors', 'exact_fraction', 'within_tolerance_fraction', 'mse_per_pixel']
                   for stat, function in [('mean', np.mean), ('std_population', np.std), ('min', np.min), ('max', np.max)]}})
    write_csv(output/'by-level.csv', rows)
    write_csv(output/'by-character.csv', character_rows)
    write_csv(output/'seed-aggregates.csv', aggregates)
    result = {'selection_sha256': sha256(campaign/'selection.json'), 'selected_run': selection['run_id'],
              'selected_variant': selection['variant'], 'seed': manifest['final_seed'],
              'collection_sha256': sha256(output/'collection.npz'), 'by_level': rows,
              'seed_aggregates': aggregates, 'prediction_artifacts': artifacts, 'example_indices': examples,
              'example_policy': 'first realization of characters 0,1,8,16,24,31 at each level',
              'models': {name: sha256(path) for name, path in models}, 'threshold': .5,
              'versions': {'python': platform.python_version(), 'numpy': np.__version__}}
    write_json(output/'metrics.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--config', default='configs/denoising-architectures.json')
    run.add_argument('--output', required=True)
    evaluation = sub.add_parser('evaluate')
    evaluation.add_argument('--campaign', required=True)
    evaluation.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'run':
        prepare(args.config, args.output)
        selection = train(args.output)
        print(f'SELECTED {selection["variant"]}: {selection["run_id"]}', flush=True)
        evaluate(args.output)
    else:
        evaluate(args.campaign, args.output)


if __name__ == '__main__':
    main()
