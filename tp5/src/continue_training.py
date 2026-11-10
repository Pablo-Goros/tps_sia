"""Bounded paired continuations from archived RMSProp runs (never overwrite them)."""
import argparse
import json
from pathlib import Path
import platform
import time

import matplotlib
import numpy as np

from .autoencoder import Autoencoder
from .dataset import DEFAULT_DATASET, load_dataset
from .metrics import reconstruction_metrics
from .plots import plot_training
from .train import _evaluation_artifacts, model_identity, write_json


def preflight(source, dataset):
    """Reload and compare with archived evidence before any parameter updates."""
    source = Path(source)
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    identity = model_identity(source/'model.npz')
    if identity['sha256'] != manifest['model']['sha256']:
        raise ValueError('Source model hash does not match its manifest')
    if dataset.sha256 != manifest['dataset']['sha256']:
        raise ValueError('Source dataset hash does not match')
    model = Autoencoder.load(source/'model.npz')
    if model.configuration['optimizer']['name'] != 'rmsprop':
        raise ValueError('This experiment requires RMSProp')
    metrics = reconstruction_metrics(dataset.vectors, model.reconstruct(dataset.vectors))
    archived = json.loads((source/'metrics.json').read_text(encoding='utf-8'))
    for key, value in metrics.items():
        if isinstance(value, float):
            if not np.isclose(value, archived[key], rtol=1e-12, atol=1e-14):
                raise ValueError(f'Reloaded metric differs: {key}')
        elif value != archived[key]:
            raise ValueError(f'Reloaded metric differs: {key}')
    with np.load(source/'model.npz', allow_pickle=False) as archive:
        config = json.loads(str(archive['configuracion']))
        state = config.get('optimizer_state', {})
        if config['version'] < 2 or state.get('updates') != model.history['updates'][-1]:
            raise ValueError('Missing or inconsistent persisted update counter')
        names = state.get('second_moment', [])
        if not names or any(name not in archive.files for name in names):
            raise ValueError('Missing persisted RMSProp moments')
    if model.history['epocas_corridas'] != manifest['epochs_completed']:
        raise ValueError('Persisted epoch count differs from manifest')
    return model, {'source_model': identity, 'source_manifest': model_identity(source/'manifest.json'),
                   'initial_metrics': metrics, 'initial_epoch': model.history['epocas_corridas'],
                   'initial_updates': state['updates'], 'moment_tensors': len(names),
                   'rng': config['rng'], 'effective_model': model.configuration}


def continue_run(source, output, learning_rate, additional_epochs, evaluate_every,
                 deadline, dataset):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    model, origin = preflight(source, dataset)
    model.set_learning_rate(learning_rate)
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    initial_epoch = origin['initial_epoch']
    target_epoch = initial_epoch + additional_epochs
    manifest = {**origin, 'mode': 'continuation', 'effective_model': model.configuration,
                'dataset': {'path': dataset.path, 'sha256': dataset.sha256},
                'additional_epochs_budget': additional_epochs, 'evaluate_every': evaluate_every,
                'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                             'matplotlib': matplotlib.__version__}, 'threshold': 0.5,
                'training_loss_normalization': '0.5 * mean(sum((prediction-target)^2, axis=1))',
                'stop_reason': 'running'}
    write_json(output/'config.json', {'source': str(Path(source).resolve()),
               'learning_rate': learning_rate, 'additional_epochs': additional_epochs,
               'evaluate_every': evaluate_every, 'stop_on_tolerance': True})
    evaluations = []
    reason = 'additional_epochs'
    try:
        while True:
            epoch = model.history['epocas_corridas']
            metrics = reconstruction_metrics(dataset.vectors, model.reconstruct(dataset.vectors))
            evaluations.append({'epoch': epoch, **metrics})
            model.save(output/'checkpoint.npz')
            manifest.update(epochs_completed=epoch, additional_epochs_completed=epoch-initial_epoch,
                            checkpoint=model_identity(output/'checkpoint.npz'))
            write_json(output/'manifest.json', manifest)
            print(f'{output.name}: epoch={epoch} max={metrics["max_pixel_errors"]} '
                  f'within={metrics["within_tolerance_fraction"]:.5f}', flush=True)
            if metrics['all_within_tolerance']:
                reason = 'pixel_tolerance'
                break
            if epoch >= target_epoch:
                break
            if time.perf_counter() >= deadline:
                reason = 'time_budget'
                break
            # Check the shared wall budget at every epoch, including between evaluations.
            for _ in range(min(evaluate_every, target_epoch-epoch)):
                if time.perf_counter() >= deadline:
                    break
                model.train(dataset.vectors, 1)
                if model.history['stop_reason'] in ('interrupted', 'numerical_failure'):
                    reason = model.history['stop_reason']
                    break
            if reason in ('interrupted', 'numerical_failure'):
                evaluations.append({'epoch': model.history['epocas_corridas'],
                                    **reconstruction_metrics(dataset.vectors, model.reconstruct(dataset.vectors))})
                break
    except KeyboardInterrupt:
        reason = 'interrupted'
    except FloatingPointError:
        reason = 'numerical_failure'
    model.save(output/'model.npz')
    metrics = _evaluation_artifacts(model, dataset, output)
    history = {key: model.history[key] for key in ['epochs', 'costo', 'mse', 'updates', 'learning_rate']}
    history['evaluations'] = evaluations
    write_json(output/'history.json', history)
    plot_training(history, output/'curves.png')
    manifest.update(stop_reason=reason, epochs_completed=model.history['epocas_corridas'],
                    additional_epochs_completed=model.history['epocas_corridas']-initial_epoch,
                    duration_seconds=time.perf_counter()-start, model=model_identity(output/'model.npz'))
    write_json(output/'manifest.json', manifest)
    return {'metrics': metrics, 'manifest': manifest}


def run_experiment(protocol_path, output, dataset_path=DEFAULT_DATASET):
    protocol = json.loads(Path(protocol_path).read_text(encoding='utf-8'))
    # This command deliberately supports only the agreed experiment and hard ceilings.
    if (protocol['seeds'] != [42, 43] or protocol['learning_rates'] != [0.002, 0.001]
            or protocol['additional_epochs'] != 1000 or protocol['evaluate_every'] != 100
            or protocol['max_seconds'] != 600):
        raise ValueError('Protocol must match the approved 4-run / 4000-epoch / 600-second budget')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Experiment output directory must be empty')
    dataset = load_dataset(dataset_path)
    sources = {seed: Path(protocol['source_root'])/f'deeper-rmsprop-0p002-seed-{seed}'
               for seed in protocol['seeds']}
    checks = {str(seed): preflight(source, dataset)[1] for seed, source in sources.items()}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'protocol.json', protocol)
    write_json(output/'preflight.json', checks)
    started = time.perf_counter()
    deadline = started + protocol['max_seconds']
    summary = {'runs': [], 'additional_epochs_completed': 0, 'status': 'running'}
    for rate in protocol['learning_rates']:
        for seed, source in sources.items():
            name = f'seed-{seed}-lr-{str(rate).replace(".", "p")}'
            if time.perf_counter() >= deadline:
                summary['runs'].append({'run_id': name, 'seed': seed, 'learning_rate': rate,
                                        'status': 'not_run', 'reason': 'time_budget'})
            else:
                result = continue_run(source, output/name, rate, protocol['additional_epochs'],
                                      protocol['evaluate_every'], deadline, dataset)
                summary['additional_epochs_completed'] += result['manifest']['additional_epochs_completed']
                summary['runs'].append({'run_id': name, 'seed': seed, 'learning_rate': rate,
                                        'status': 'finished', **result})
            summary['duration_seconds'] = time.perf_counter()-started
            write_json(output/'summary.json', summary)
            if summary['runs'][-1].get('manifest', {}).get('stop_reason') in ('interrupted', 'numerical_failure'):
                summary['status'] = 'stopped'
                write_json(output/'summary.json', summary)
                return summary
    summary['status'] = 'completed' if all(
        r.get('manifest', {}).get('stop_reason') in ('additional_epochs', 'pixel_tolerance')
        for r in summary['runs']) else 'time_budget'
    summary['source_hashes_unchanged'] = all(
        model_identity(sources[int(seed)]/'model.npz')['sha256'] == check['source_model']['sha256']
        for seed, check in checks.items())
    write_json(output/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run_experiment(args.protocol, args.output)
    print(json.dumps({'status': result['status'], 'additional_epochs_completed': result['additional_epochs_completed']}))


if __name__ == '__main__':
    main()
