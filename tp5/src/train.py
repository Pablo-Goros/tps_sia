"""Reproducible JSON training and reload-only evaluation commands."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import matplotlib
from .autoencoder import Autoencoder
from .dataset import DEFAULT_DATASET, load_dataset
from .metrics import reconstruction_metrics
from .plots import plot_reconstructions, plot_training


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def model_identity(path):
    return {'path': str(Path(path).resolve()),
            'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def _evaluation_artifacts(model, dataset, output):
    prediction = model.reconstruct(dataset.vectors)
    metrics = reconstruction_metrics(dataset.vectors, prediction)
    write_json(output/'metrics.json', {**metrics, 'labels': dataset.labels})
    np.savez_compressed(output/'predictions.npz', codes=model.encode(dataset.vectors),
                        predictions=prediction, binary=prediction >= 0.5, labels=dataset.labels)
    plot_reconstructions(dataset, prediction, metrics, output/'reconstructions.png')
    return metrics


def evaluate_model(model_path, output, dataset_path=DEFAULT_DATASET):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_path)
    model = Autoencoder.load(model_path)
    metrics = _evaluation_artifacts(model, dataset, output)
    write_json(output/'manifest.json', {'model': model_identity(model_path),
               'dataset': {'path': dataset.path, 'sha256': dataset.sha256}, 'mode': 'evaluation'})
    return metrics


def run_training(config_path, output, dataset_path=DEFAULT_DATASET):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    required = {'architecture', 'seed', 'optimizer', 'batch_size', 'max_epochs',
                'evaluate_every', 'stop_on_tolerance'}
    if set(config) != required:
        raise ValueError(f'Configuration must contain exactly {sorted(required)}')
    for key in ['max_epochs', 'evaluate_every', 'batch_size']:
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if type(config['seed']) is not int or type(config['stop_on_tolerance']) is not bool:
        raise ValueError('seed must be integer and stop_on_tolerance boolean')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty to preserve previous runs')
    dataset = load_dataset(dataset_path)
    model = Autoencoder(architecture=config['architecture'], seed=config['seed'],
                        optimizer=config['optimizer'], batch_size=config['batch_size'])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'config.json', config)
    start = time.perf_counter()
    manifest = {'config': config, 'effective_model': model.configuration,
                'dataset': {'path': dataset.path, 'sha256': dataset.sha256},
                'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                             'matplotlib': matplotlib.__version__},
                'training_loss_normalization': '0.5 * mean(sum((prediction-target)^2, axis=1))',
                'threshold': 0.5, 'stop_reason': 'running'}
    write_json(output/'manifest.json', manifest)
    evaluations = []
    reason = 'max_epochs'
    try:
        while True:
            epoch = model.history['epocas_corridas']
            metrics = reconstruction_metrics(dataset.vectors, model.reconstruct(dataset.vectors))
            evaluations.append({'epoch': epoch, **metrics})
            if config['stop_on_tolerance'] and metrics['all_within_tolerance']:
                reason = 'pixel_tolerance'
                break
            if epoch >= config['max_epochs']:
                break
            model.train(dataset.vectors, min(config['evaluate_every'], config['max_epochs']-epoch))
            if model.history['stop_reason'] in ['interrupted', 'numerical_failure']:
                reason = model.history['stop_reason']
                evaluations.append({'epoch': model.history['epocas_corridas'], **reconstruction_metrics(dataset.vectors, model.reconstruct(dataset.vectors))})
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
                    duration_seconds=time.perf_counter()-start, model=model_identity(output/'model.npz'))
    write_json(output/'manifest.json', manifest)
    return {'metrics': metrics, 'manifest': manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--config', required=True, type=Path)
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('--model', required=True, type=Path)
    for command in [train, evaluate]:
        command.add_argument('--output', required=True, type=Path)
        command.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    try:
        if args.command == 'train':
            result = run_training(args.config, args.output, args.dataset)['metrics']
        else:
            result = evaluate_model(args.model, args.output, args.dataset)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
