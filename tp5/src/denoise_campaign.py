"""Frozen campaign orchestration and reusable numerical evaluation artifacts."""
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
from .autoencoder import Autoencoder
from .dataset import load_dataset
from .noise import make_collection
from .denoise_training import train_run, select_runs
from .denoise_evaluation import grouped_metrics, write_csv
from .train import write_json
from .denoise import APPROVED_PROTOCOL


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_campaign(campaign):
    campaign = Path(campaign)
    manifest = read_json(campaign/'manifest.json')
    protocol = {k: v for k, v in manifest['protocol'].items() if k != 'source_selection'}
    if json.dumps(protocol, sort_keys=True) != json.dumps(APPROVED_PROTOCOL, sort_keys=True):
        raise ValueError('Campaign differs from the approved protocol')
    for entry in manifest['reference'].values():
        if sha256(campaign/entry['path']) != entry['sha256']:
            raise ValueError('Frozen reference hash mismatch')
    # Derive configurations again from the frozen source rather than trusting
    # editable variant fields in the campaign manifest.
    source = read_json(campaign/'reference/config.json')
    winner = {k: source[k] for k in ('architecture', 'optimizer')}
    winner.update(loss=source.get('loss', 'half_squared_error'),
                  latent_activation=source.get('latent_activation', 'tanh'),
                  lr_scheduler=source.get('lr_scheduler'), batch_size=manifest['protocol']['batch_size'])
    alternative = {**winner, 'architecture': list(winner['architecture'])}
    alternative['architecture'][len(alternative['architecture'])//2] = 10
    if manifest['variants'] != {'winner': winner, 'latent-10': alternative}:
        raise ValueError('Campaign variants disagree with frozen reference')
    return manifest


def prepare_collections(campaign):
    """Add B2 collections to a B1 campaign, or verify previously saved copies."""
    campaign = Path(campaign)
    manifest = verified_campaign(campaign)
    directory = campaign/'collections'
    if directory.exists():
        return verify_collections(campaign)
    noise = manifest['protocol']['noise']
    dataset = load_dataset(campaign/manifest['dataset']['path'])
    directory.mkdir()
    info = {'schema_version': 1, 'ordering': 'level, character_index, realization_index',
            'numpy_version': np.__version__, 'dataset_sha256': dataset.sha256, 'collections': {}}
    for name, seed in [('validation', noise['validation_seed']), ('final', noise['evaluation_seed'])]:
        collection = make_collection(dataset.vectors, noise['evaluation_levels'], noise['realizations_per_character'], seed)
        path = directory/f'{name}.npz'
        np.savez_compressed(path, **collection)
        info['collections'][name] = {'path': f'collections/{name}.npz', 'sha256': sha256(path),
                                     'seed': seed, 'count': len(collection['inputs']),
                                     'levels': noise['evaluation_levels'], 'generator': 'numpy.default_rng/PCG64'}
    write_json(directory/'manifest.json', info)
    return info


def verify_collections(campaign):
    campaign = Path(campaign)
    info = read_json(campaign/'collections/manifest.json')
    manifest = verified_campaign(campaign)
    if info['dataset_sha256'] != manifest['dataset']['sha256']:
        raise ValueError('Collection dataset mismatch')
    for entry in info['collections'].values():
        if sha256(campaign/entry['path']) != entry['sha256']:
            raise ValueError('Collection hash mismatch')
    return info


def load_collection(campaign, name):
    info = verify_collections(campaign)
    with np.load(Path(campaign)/info['collections'][name]['path'], allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def train_campaign(campaign):
    campaign = Path(campaign)
    manifest = verified_campaign(campaign)
    collections = verify_collections(campaign)
    validation = load_collection(campaign, 'validation')
    mask = validation['levels'] != .3
    validation = {key: values[mask] for key, values in validation.items()}
    clean = load_dataset(campaign/manifest['dataset']['path']).vectors
    runs_directory = campaign/'runs'
    if runs_directory.exists() or (campaign/'selection.json').exists():
        raise ValueError('Campaign already started; automatic retries are disabled')
    runs_directory.mkdir()
    protocol = manifest['protocol']
    ledger = {'status': 'running', 'epochs_completed': 0, 'max_total_epochs': 18000,
              'runs': [], 'versions': {'python': platform.python_version(), 'numpy': np.__version__},
              'validation_sha256': collections['collections']['validation']['sha256'],
              'campaign_manifest_sha256': sha256(campaign/'manifest.json')}
    write_json(campaign/'training.json', ledger)
    try:
        for variant, model_config in manifest['variants'].items():
            for seed in protocol['seeds']:
                config = {'variant': variant, 'seed': seed, 'model': model_config,
                          'max_epochs': protocol['max_epochs'], 'evaluate_every': protocol['evaluate_every'],
                          'training_levels': protocol['noise']['training_levels'],
                          'selection_levels': protocol['noise']['selection_levels'],
                          'dataset_sha256': manifest['dataset']['sha256'],
                          'validation_sha256': ledger['validation_sha256']}
                result = train_run(config, clean, validation, runs_directory/f'{variant}-seed-{seed}')
                ledger['runs'].append(result)
                ledger['epochs_completed'] += result['epochs_completed']
                write_json(campaign/'training.json', ledger)
                if result['status'] == 'interrupted':
                    raise KeyboardInterrupt('Run interrupted; no selection or automatic retry')
                if result['status'] != 'completed':
                    raise RuntimeError('Campaign interrupted; no selection or automatic retry')
        selection = select_runs(ledger['runs'], list(manifest['variants']), protocol['seeds'])
        for run in ledger['runs']:
            path = f"runs/{run['run_id']}/best_model.npz"
            run['best']['model'] = {'path': path, 'sha256': sha256(campaign/path)}
        selection['runs'] = ledger['runs']
        selection['campaign_manifest_sha256'] = ledger['campaign_manifest_sha256']
        selection['validation_sha256'] = ledger['validation_sha256']
        write_json(campaign/'selection.json', selection)
        write_csv(campaign/'validation-summary.csv', [
            {'run_id': r['run_id'], 'variant': r['variant'], 'seed': r['seed'],
             'best_epoch': r['best']['epoch'], 'mean_pixel_errors': r['best']['score'][0],
             'mse_per_pixel': r['best']['score'][1]} for r in ledger['runs']])
        ledger['status'] = 'completed'
    except BaseException as error:
        ledger['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        ledger['error'] = f'{type(error).__name__}: {error}'
        # Include partially completed run manifests in the accounting.
        ledger['runs'] = [read_json(path) for path in sorted(runs_directory.glob('*/manifest.json'))]
        ledger['epochs_completed'] = sum(r['epochs_completed'] for r in ledger['runs'])
        write_json(campaign/'training.json', ledger)
        raise
    write_json(campaign/'training.json', ledger)
    return selection


def evaluate_campaign(campaign, output=None):
    """Reload all six best checkpoints and frozen AE; save arrays and tables only."""
    campaign = Path(campaign)
    manifest = verified_campaign(campaign)
    selection = read_json(campaign/'selection.json')
    ledger = read_json(campaign/'training.json')
    if ledger['status'] != 'completed' or ledger['epochs_completed'] != 18000:
        raise ValueError('Final evaluation requires the completed approved campaign')
    if selection['campaign_manifest_sha256'] != sha256(campaign/'manifest.json'):
        raise ValueError('Selection campaign hash mismatch')
    expected = select_runs(selection['runs'], list(manifest['variants']), manifest['protocol']['seeds'])
    if selection['run_id'] != expected['run_id'] or selection['variant'] != expected['variant']:
        raise ValueError('Selection no longer matches validation ranking')
    final = load_collection(campaign, 'final')
    models = [('ae-reference', campaign/'reference/model.npz', sha256(campaign/'reference/model.npz'))]
    for run in selection['runs']:
        entry = run['best']['model']
        models.append((run['run_id'], campaign/entry['path'], entry['sha256']))
    for _, path, digest in models:
        if sha256(path) != digest:
            raise ValueError('Checkpoint hash mismatch')
    output = Path(output) if output is not None else campaign/'evaluation'
    if output.exists():
        raise ValueError('Evaluation output must be new')
    output.mkdir(parents=True)
    rows, character_rows, artifacts = [], [], {}
    predictions = [('noisy-input', final['inputs'], None)]
    for name, path, digest in models:
        model = Autoencoder.load(path)
        predictions.append((name, model.reconstruct(final['inputs']), model.encode(final['inputs'])))
    for name, prediction, codes in predictions:
        errors = np.count_nonzero((prediction >= .5) != final['targets'], axis=1)
        input_errors = np.count_nonzero(final['inputs'] != final['targets'], axis=1)
        payload = {'predictions': prediction, 'binary': (prediction >= .5).astype(np.uint8),
                   'pixel_errors': errors, 'input_pixel_errors': input_errors,
                   'error_reduction': input_errors-errors,
                   'mse_per_example': np.mean((prediction-final['targets'])**2, axis=1)}
        if codes is not None:
            payload['codes'] = codes
        path = output/f'{name}.npz'
        np.savez_compressed(path, **payload)
        artifacts[name] = {'path': path.name, 'sha256': sha256(path)}
        rows.extend({'model': name, **r} for r in grouped_metrics(final, prediction))
        character_rows.extend({'model': name, **r} for r in grouped_metrics(final, prediction, by_character=True))
    write_csv(output/'by-level.csv', rows)
    write_csv(output/'by-character.csv', character_rows)
    info = verify_collections(campaign)
    result = {'schema_version': 1, 'selected_run': selection['run_id'], 'by_level': rows,
              'final_collection': info['collections']['final'], 'artifacts': artifacts,
              'models': {name: {'sha256': digest, 'path': str(path.relative_to(campaign))} for name, path, digest in models},
              'selection_sha256': sha256(campaign/'selection.json'), 'threshold': .5,
              'example_indices': [int(np.flatnonzero((final['levels'] == p) & (final['character_indices'] == c))[0])
                                  for p in manifest['protocol']['noise']['evaluation_levels'] for c in [0,1,8,16,24,31]],
              'example_policy': 'first realization of characters 0,1,8,16,24,31 at each level; fixed before inspection',
              'out_of_training_range': [.3], 'versions': {'python': platform.python_version(), 'numpy': np.__version__}}
    # Seed variation is separate from image variation, with one value per run.
    result['seed_aggregates'] = []
    for variant in manifest['variants']:
        names = [r['run_id'] for r in selection['runs'] if r['variant'] == variant]
        for level in manifest['protocol']['noise']['evaluation_levels']:
            group = [r for r in rows if r['model'] in names and r['level'] == level]
            result['seed_aggregates'].append({'variant': variant, 'level': level, 'seeds': manifest['protocol']['seeds'],
                **{f'{key}_{stat}': float(function([r[key] for r in group]))
                   for key in ('mean_pixel_errors', 'exact_fraction', 'within_tolerance_fraction', 'mse_per_pixel', 'mean_error_reduction')
                   for stat, function in [('mean', np.mean), ('std_population', np.std), ('min', np.min), ('max', np.max)]}})
    write_csv(output/'seed-aggregates.csv', result['seed_aggregates'])
    write_json(output/'metrics.json', result)
    return result
