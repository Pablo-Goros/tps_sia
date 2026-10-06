"""Validated development runs, provenance, resumable checkpoints and artifacts.

Paths are supplied by consumers. No exercise defaults or final-test evaluation
belong here. The epoch budget is total, including epochs restored on resume.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import numpy as np

from .atomic_files import replace
from .digit_dataset import cargar, particionar
from .metrics import evaluate_model
from .mlp import MLP
from .optimizers import construir_optimizador


DEFAULTS = {
    'architecture': [784, 128, 10], 'activation': 'relu', 'activation_parameters': {'beta': 1.0},
    'output': 'softmax', 'loss': 'cross_entropy',
    'optimizer': {'name': 'sgd', 'learning_rate': 0.01},
    'strategy': 'mini_batch', 'batch_size': 32, 'shuffle': True,
    'initialization': 'auto', 'model_seed': 42, 'split_seed': 42,
    'validation_fraction': 0.2, 'epochs': 30,
    'stopping': {'epsilon': 0.0, 'patience': None, 'min_delta': 0.0,
                 'monitor': 'validation_loss'},
    'preprocessing': {'name': 'identity'}, 'checkpoint_every': 1,
    'weight_logging': {'enabled': False, 'every_updates': 100, 'selected_weights': []},
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def portable_path(path: str | Path, root: str | Path) -> str:
    """POSIX path relative to root, so provenance does not depend on the machine."""
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f'{path} must be inside {root} for portable provenance.') from exc


def recorded_config(config: dict, path_root: str | Path | None = None) -> dict:
    """Config as stored; with path_root, dataset and cache become root-relative."""
    if path_root is None:
        return config
    recorded = {**config, 'dataset': portable_path(config['dataset'], path_root),
                'cache': portable_path(config['cache'], path_root)}
    if 'data' in config:
        recorded['data'] = {**config['data']}
        for key in ('validation_dataset', 'validation_cache'):
            recorded['data'][key] = portable_path(config['data'][key], path_root)
    return recorded


def config_identity(config: dict, initialization: dict | None = None,
                    path_root: str | Path | None = None) -> str:
    """Stable id shared by every seed; excludes the model seed and cache path.

    Without path_root the historical id (absolute dataset path) is preserved.
    """
    group = {k: v for k, v in recorded_config(config, path_root).items()
             if k not in ('model_seed', 'cache')}
    return fingerprint({'config': group, 'initialization': initialization or {'mode': 'fresh'}})[:16]


def write_json(path: Path, value: dict) -> None:
    """Replace only complete JSON files; reject NaN rather than emitting it."""
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     delete=False) as file:
        temporary = Path(file.name)
        file.write(data)
    try:
        replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_history(path: Path, history: dict) -> None:
    epochs = history.get('epochs') or list(range(1, history['epocas_corridas'] + 1))
    validation_epochs = history.get('validation_epochs') or epochs[:len(history['costo_validacion'])]
    validation = dict(zip(validation_epochs, zip(history['costo_validacion'], history['accuracy_validacion'])))
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'train_loss', 'validation_loss', 'train_accuracy',
                         'validation_accuracy', 'learning_rate', 'updates', 'gradient_norm', 'training_objective', 'l2_penalty'])
        for i, epoch in enumerate(epochs):
            val = validation.get(epoch, (None, None))
            writer.writerow([epoch, history['costo'][i], val[0], history['accuracy'][i], val[1],
                             (history.get('learning_rate') or [None] * len(epochs))[i],
                             (history.get('updates') or [None] * len(epochs))[i],
                             history['norma_gradiente'][i],
                             history.get('training_objective', [])[i] if i < len(history.get('training_objective', [])) else None,
                             history.get('l2_penalty', [])[i] if i < len(history.get('l2_penalty', [])) else None])


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}.')


def validate_config(config: dict) -> dict:
    """Canonicalize defaults and reject unsupported options before reading data."""
    if not isinstance(config, dict) or set(config) - (set(DEFAULTS) | {'dataset', 'cache', 'data', 'l2', 'augmentation', 'lr_scheduler', 'batch_norm'}):
        raise ValueError('Unknown experiment configuration fields.')
    c = copy.deepcopy(DEFAULTS)
    for key, value in config.items():
        if key in ('activation_parameters', 'stopping', 'preprocessing', 'weight_logging'):
            if not isinstance(value, dict) or set(value) - set(DEFAULTS[key]):
                raise ValueError(f'Unknown {key} configuration.')
            c[key].update(value)
        else:
            c[key] = copy.deepcopy(value)
    for key in ('dataset', 'cache'):
        if key not in c or not isinstance(c[key], (str, Path)):
            raise ValueError(f'{key} path is required.')
        c[key] = str(Path(c[key]).resolve())
    # Check resolved paths as well, so a symlink to the final test is rejected.
    if (Path(config['dataset']).name.casefold() == 'digits_test.csv'
            or Path(c['dataset']).name.casefold() == 'digits_test.csv'):
        raise ValueError('digits_test.csv is reserved for final evaluation.')
    if c['cache'] == c['dataset'] or Path(c['cache']).suffix != '.npz':
        raise ValueError('Cache must be a separate .npz file.')
    if 'data' in c:
        c['data'] = validate_data(c['data'], c['dataset'], c['cache'])
    for key in ('epochs', 'checkpoint_every', 'model_seed', 'split_seed'):
        _integer(c[key], key, 0 if key.endswith('seed') else 1)
    if not isinstance(c['shuffle'], bool):
        raise ValueError('shuffle must be boolean.')
    if c['strategy'] not in ('online', 'batch', 'mini_batch'):
        raise ValueError('Unknown training strategy.')
    if ((c['strategy'] == 'batch' and c['batch_size'] is not None)
            or (c['strategy'] == 'online' and (isinstance(c['batch_size'], bool) or c['batch_size'] != 1))):
        raise ValueError('Batch requires null batch_size; online requires 1.')
    if c['strategy'] == 'mini_batch':
        _integer(c['batch_size'], 'batch_size', 2)
    if not isinstance(c['architecture'], list) or c['architecture'][0:1] != [784] or c['architecture'][-1:] != [10]:
        raise ValueError('Digit architecture requires 784 inputs and 10 outputs.')
    if (c['output'], c['loss']) not in (('softmax', 'cross_entropy'), ('logistica', 'half_squared_error')):
        raise ValueError('Incompatible output and loss.')
    for key in ('epsilon', 'min_delta'):
        value = c['stopping'][key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be finite and nonnegative.')
    if c['stopping']['patience'] is not None:
        _integer(c['stopping']['patience'], 'patience')
    if c['stopping']['monitor'] not in ('validation_loss', 'validation_accuracy'):
        raise ValueError('Unknown validation monitor.')
    if (isinstance(c['validation_fraction'], bool) or not isinstance(c['validation_fraction'], (int, float))
            or not np.isfinite(c['validation_fraction']) or not 0 < c['validation_fraction'] < 1):
        raise ValueError('validation_fraction must be between 0 and 1.')
    if c['preprocessing']['name'] not in ('identity', 'standardize'):
        raise ValueError('Preprocessing supports identity or standardize.')
    if c.get('augmentation') and c['preprocessing']['name'] != 'identity':
        raise ValueError('Zero-padded translations currently require identity preprocessing.')
    logging = c['weight_logging']
    if not isinstance(logging['enabled'], bool):
        raise ValueError('weight_logging.enabled must be boolean.')
    _integer(logging['every_updates'], 'every_updates')
    if not isinstance(logging['selected_weights'], list):
        raise ValueError('selected_weights must be a list.')
    beta = c['activation_parameters']['beta']
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not np.isfinite(beta) or beta <= 0:
        raise ValueError('Activation beta must be positive and finite.')
    model = build_model(c)  # Also validates optimizer and architecture.
    if 'lr_scheduler' in c:
        c['lr_scheduler'] = model.lr_scheduler
    if 'batch_norm' in c:
        c['batch_norm'] = model.batch_norm
    for index in logging['selected_weights']:
        if not isinstance(index, (list, tuple)) or len(index) != 3:
            raise ValueError('Selected weight must be [layer, row, column].')
        for value in index:
            _integer(value, 'weight index', 0)
        layer, row, col = index
        if layer >= len(model.pesos) or row >= model.pesos[layer].shape[0] or col >= model.pesos[layer].shape[1]:
            raise ValueError('Selected weight index out of range.')
    # Record effective optimizer defaults, including a constant learning rate.
    c['optimizer'] = model.optimizador.configuracion()
    if 'nombre' in c['optimizer']:
        c['optimizer'] = {'name': 'sgd', 'learning_rate': model.optimizador.eta}
    return c


def build_model(config: dict) -> MLP:
    return MLP(config['architecture'], activacion=config['activation'], salida=config['output'],
               beta=config['activation_parameters']['beta'], tamano_lote=config['batch_size'],
               inicializacion=config['initialization'], semilla=config['model_seed'],
               optimizador=construir_optimizador(config['optimizer']), l2=config.get('l2', 0.0),
               augmentation=config.get('augmentation'), lr_scheduler=config.get('lr_scheduler'),
               batch_norm=config.get('batch_norm'))


def preprocess(X: np.ndarray, settings: dict) -> np.ndarray:
    """Apply saved train-only preprocessing to any partition."""
    if settings['name'] == 'identity':
        return X
    return (X - np.asarray(settings['mean'])) / np.asarray(settings['scale'])


def validate_data(spec: dict, dataset: str, cache: str) -> dict:
    """Explicit indices and independently sourced validation, without reading CSVs."""
    allowed = {'train_indices', 'validation_indices', 'validation_dataset',
               'validation_cache', 'dataset_sha256', 'validation_sha256'}
    if not isinstance(spec, dict) or set(spec) != allowed:
        raise ValueError(f'Explicit data requires exactly {sorted(allowed)}.')
    result = copy.deepcopy(spec)
    for key in ('train_indices', 'validation_indices'):
        indices = result[key]
        if (not isinstance(indices, list) or not indices
                or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError(f'{key} must contain unique nonnegative integer indices.')
    for key in ('dataset_sha256', 'validation_sha256'):
        value = result[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'{key} must be a SHA-256 digest.')
    for key in ('validation_dataset', 'validation_cache'):
        result[key] = str(Path(result[key]).resolve())
    if (Path(spec['validation_dataset']).name.casefold() == 'digits_test.csv'
            or Path(result['validation_dataset']).name.casefold() == 'digits_test.csv'):
        raise ValueError('digits_test.csv is reserved for final evaluation.')
    if (result['validation_cache'] in (dataset, result['validation_dataset'])
            or cache == result['validation_dataset']
            or (result['validation_dataset'] == dataset and result['validation_cache'] != cache)
            or Path(result['validation_cache']).suffix != '.npz'):
        raise ValueError('Validation cache must be a separate .npz file.')
    if result['validation_dataset'] == dataset:
        if set(result['train_indices']) & set(result['validation_indices']):
            raise ValueError('Train and validation indices overlap.')
        if result['dataset_sha256'] != result['validation_sha256']:
            raise ValueError('One source must have one digest.')
    elif result['validation_cache'] == cache:
        raise ValueError('Different datasets require different caches.')
    return result


def data_hashes(config: dict) -> dict:
    """Verify pinned sources before reuse, resume or training."""
    hashes = {'dataset_sha256': sha256_file(config['dataset'])}
    if 'data' in config:
        spec = config['data']
        hashes['validation_sha256'] = sha256_file(spec['validation_dataset'])
        if any(hashes[k] != spec[k] for k in hashes):
            raise ValueError('Dataset changed; regenerate data indices and use a new output directory.')
    return hashes


def explicit_partitions(config: dict, X: np.ndarray, y: np.ndarray):
    spec = config['data']
    if spec['validation_dataset'] == config['dataset']:
        V, v = X, y
    else:
        V, v = cargar(spec['validation_dataset'], spec['validation_cache'])
    train = np.asarray(spec['train_indices'], dtype=np.int64)
    validation = np.asarray(spec['validation_indices'], dtype=np.int64)
    if train.max() >= len(X) or validation.max() >= len(V):
        raise ValueError('Explicit indices are outside the source dataset.')
    # Exact image leakage is forbidden even when row indices or sources differ.
    def hashes(rows):
        return {hashlib.sha256(row.tobytes()).digest() for row in
                np.ascontiguousarray(rows, dtype=np.float32) + np.float32(0.0)}
    if hashes(X[train]) & hashes(V[validation]):
        raise ValueError('Identical images appear in both train and validation.')
    data_hashes(config)
    return X[train], y[train], V[validation], v[validation], train, validation


def run(config: dict, output_root: str | Path, *, resume: bool = False,
        initial_model: str | Path | None = None, pause_after: int | None = None,
        verbose: int = 0, metadata: dict | None = None,
        path_root: str | Path | None = None) -> dict:
    """path_root stores dataset, cache and source paths relative to that root,
    and computes config_id without machine-dependent paths."""
    c = validate_config(config)
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError('metadata must be a JSON object.')
    metadata = copy.deepcopy(metadata or {})
    fingerprint(metadata)
    if pause_after is not None:
        _integer(pause_after, 'pause_after')
    _integer(verbose, 'verbose', 0)
    initialization = {'mode': 'fresh'}
    if initial_model is not None:
        source_path = (str(Path(initial_model).resolve()) if path_root is None
                       else portable_path(initial_model, path_root))
        initialization = {'mode': 'existing_weights', 'path': source_path,
                          'sha256': sha256_file(initial_model)}
    recorded = recorded_config(c, path_root)
    config_id = config_identity(c, initialization, path_root)
    run_id = f'{config_id}-seed-{c["model_seed"]}'
    directory = Path(output_root).resolve() / run_id
    source_hashes = data_hashes(c)
    dataset_hash = source_hashes['dataset_sha256']
    identity = {'config': recorded, 'initialization': initialization, 'dataset_sha256': dataset_hash,
                'metadata': metadata}
    if 'data' in c:
        identity['data_sources'] = source_hashes
    if resume:
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['identity'] != identity:
            raise ValueError('Resume requires the same configuration, source and initialization.')
        if (directory / 'results.json').exists():
            previous = json.loads((directory / 'results.json').read_text(encoding='utf-8'))
            if previous['status'] != 'interrupted':
                raise FileExistsError('A finished run cannot be overwritten or resumed.')
        model = MLP.cargar(directory / 'checkpoint.npz')
        if model.historia.stop_reason != 'interrupted' and model.historia.stop_reason is not None:
            raise ValueError('Checkpoint already finished its training policy.')
    else:
        model = build_model(c)
        if initial_model is not None:
            source = MLP.cargar(initial_model)
            if (source.arquitectura != model.arquitectura or source.salida != model.salida
                    or source.activacion.nombre != model.activacion.nombre or source.beta != model.beta):
                raise ValueError('Initial model must have compatible architecture and activations.')
            model.pesos = [p.copy() for p in source.pesos]
            model.biases = [b.copy() for b in source.biases]
        if directory.exists():
            raise FileExistsError('An existing run cannot be overwritten.')
    X, y = cargar(c['dataset'], c['cache'])
    # Detect source changes during loading, before training.
    if sha256_file(c['dataset']) != dataset_hash:
        raise ValueError('Dataset changed during loading.')
    if 'data' in c:
        Xt, yt, Xv, yv, train_indices, validation_indices = explicit_partitions(c, X, y)
    else:
        Xt, yt, Xv, yv, train_indices, validation_indices = particionar(
            X, y, semilla=c['split_seed'], validation_fraction=c['validation_fraction'], return_indices=True)
    split_hash = fingerprint({'train': train_indices.tolist(), 'validation': validation_indices.tolist()})
    if resume:
        with np.load(directory / 'split.npz', allow_pickle=False) as saved:
            if not np.array_equal(saved['train'], train_indices) or not np.array_equal(saved['validation'], validation_indices):
                raise ValueError('Saved partition differs from current partition.')
    else:
        settings = {'name': c['preprocessing']['name']}
        if settings['name'] == 'standardize':
            mean, scale = Xt.mean(axis=0, dtype=float), Xt.std(axis=0, dtype=float)
            scale[scale == 0] = 1
            settings.update(mean=mean.tolist(), scale=scale.tolist())
        if initial_model is not None and source.preprocessing != settings:
            raise ValueError('Initial model preprocessing differs from this train partition.')
        model.preprocessing = settings
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / 'manifest.json', {'schema_version': 1, 'identity': identity})
        np.savez_compressed(directory / 'split.npz', train=train_indices, validation=validation_indices)
    Xt, Xv = preprocess(Xt, model.preprocessing), preprocess(Xv, model.preprocessing)
    if 'data' in c:
        data_hashes(c)
    remaining = c['epochs'] - model.epochs_completed
    if remaining <= 0:
        raise ValueError('No epochs remain in the total budget.')
    started = time.perf_counter()
    history = model.entrenar(
        Xt, yt, epocas=remaining, X_val=Xv, y_val=yv, verbose=verbose,
        epsilon=c['stopping']['epsilon'], patience=c['stopping']['patience'],
        min_delta=c['stopping']['min_delta'], monitor=c['stopping']['monitor'], mezclar=c['shuffle'],
        checkpoint_path=directory / 'checkpoint.npz', checkpoint_every=c['checkpoint_every'],
        best_model_path=directory / 'best_model.npz', pause_after=pause_after,
        weight_log_path=directory / 'weights.jsonl' if c['weight_logging']['enabled'] else None,
        log_every_updates=c['weight_logging']['every_updates'],
        selected_weights=c['weight_logging']['selected_weights'])
    duration = time.perf_counter() - started
    status = {'numerical_failure': 'failed', 'interrupted': 'interrupted'}.get(history.stop_reason, 'completed')
    run_directory = str(directory)
    if path_root is not None:
        try:
            run_directory = portable_path(directory, path_root)
        except ValueError:
            run_directory = run_id  # Outside the root: only the id is portable.
    report = {
        'schema_version': 2, 'run_id': run_id, 'config_id': config_id, 'config': recorded,
        'initialization': initialization, 'metadata': metadata, 'status': status, 'stop_reason': history.stop_reason,
        'run_directory': run_directory, 'history': history.to_dict(),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'platform': platform.platform(),
                        'threads': {k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}},
        'duration_seconds': history.tiempo_segundos, 'last_call_seconds': duration,
        'parameter_count': sum(p.size for p in model.parametros),
        'chosen_epoch': model.early_stopping['best_epoch'],
        'selection_monitor': c['stopping']['monitor'], 'initialization_layers': model.initialization_layers,
        'preprocessing': model.preprocessing,
        'dataset': {'source': recorded['dataset'], 'sha256': dataset_hash, 'split_sha256': split_hash,
                    'split_indices': 'split.npz', 'split_seed': c['split_seed'],
                    'validation_fraction': c['validation_fraction'],
                    'train_samples': len(yt), 'validation_samples': len(yv),
                    'class_counts': y.sum(axis=0).astype(int).tolist(),
                    'train_class_counts': yt.sum(axis=0).astype(int).tolist(),
                    'validation_class_counts': yv.sum(axis=0).astype(int).tolist()},
        'metrics': None, 'artifacts': {'checkpoint': 'checkpoint.npz', 'history': 'history.csv', 'split': 'split.npz'},
    }
    if status == 'completed':
        chosen = MLP.cargar(directory / 'best_model.npz')
        report['metrics'] = {'train': evaluate_model(chosen, Xt, yt), 'validation': evaluate_model(chosen, Xv, yv)}
        report['artifacts']['best_model'] = 'best_model.npz'
        report['model_sha256'] = sha256_file(directory / 'best_model.npz')
    if c['weight_logging']['enabled']:
        report['artifacts']['weights'] = 'weights.jsonl'
    if 'data' in c:
        report['dataset'].update(validation_source=recorded['data']['validation_dataset'],
                                 validation_sha256=source_hashes['validation_sha256'],
                                 partition_method='explicit_indices')
    if path_root is not None:
        report['paths_relative_to'] = Path(path_root).resolve().name
    write_history(directory / 'history.csv', history.to_dict())
    write_json(directory / 'results.json', report)
    return report
