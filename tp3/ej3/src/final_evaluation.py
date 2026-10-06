"""Evaluate a confirmed, frozen ej3 checkpoint without training or selection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tps_sia.tp3.shared.digit_dataset import cargar
from tps_sia.tp3.shared.experiments import preprocess, sha256_file, write_json
from tps_sia.tp3.shared.metrics import classification_metrics, cross_entropy
from tps_sia.tp3.shared.mlp import MLP
from .protocol import CONFIG, OUTPUT, TP3, Development, read_json


def frozen_candidate(config=CONFIG, selection=OUTPUT / 'selection.json', root=TP3):
    """Validate development provenance before reading any test data."""
    path = Path(selection).resolve()
    selected = read_json(path)
    development = Development(config, root=root)
    if selected['identity'] != development.identity:
        raise ValueError('Selection belongs to another development protocol.')
    confirmation = read_json(path.parent / 'stage_confirmation.json')
    winner = confirmation['winner']
    if (confirmation['identity'] != selected['identity']
            or winner['config_id'] != selected['config_id']
            or winner['config'] != selected['config']
            or selected['seeds'] != development.protocol['confirmation_seeds']
            or winner['status'] != 'completed'):
        raise ValueError('Selection does not match completed confirmation.')
    row = next(r for r in confirmation['runs']
               if r['config_id'] == selected['config_id']
               and r['seed'] == development.protocol['primary_seed'])
    candidate = (path.parent / selected['validation_model']).resolve()
    expected_dir = (path.parent / 'runs' / row['run_id']).resolve()
    report = read_json(expected_dir / 'results.json')
    if (candidate != (expected_dir / report['artifacts']['best_model']).resolve()
            or report['status'] != 'completed'
            or report['config_id'] != selected['config_id']
            or report['metadata'] != development.identity
            or report['config'] != {**selected['config'], 'model_seed': row['seed']}
            or report['chosen_epoch'] != selected['chosen_epoch']
            or report['dataset'] != selected['dataset']
            or report['model_sha256'] != selected['model_sha256']
            or sha256_file(candidate) != selected['model_sha256']):
        raise ValueError('Frozen checkpoint or run provenance differs from selection.')
    model = MLP.cargar(candidate)
    if (model.preprocessing != report['preprocessing']
            or model.arquitectura != selected['config']['architecture']
            or model.salida != selected['config']['output']
            or model.l2 != selected['config'].get('l2', 0.0)
            or model.augmentation != selected['config'].get('augmentation')
            or model.lr_scheduler != selected['config'].get('lr_scheduler')
            or model.batch_norm != selected['config'].get('batch_norm')
            or model.seed != row['seed']):
        raise ValueError('Checkpoint configuration or preprocessing differs from its run.')
    return model, selected, report


def evaluate(config=CONFIG, selection=OUTPUT / 'selection.json', output=None,
             *, dry_run=False, batch_size=256, root=TP3):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('Prediction batch size must be a positive integer.')
    selection = Path(selection).resolve()
    output = Path(output).resolve() if output else selection.parent / 'final'
    # Refuse all existing output, including incomplete evaluations.
    if output.exists():
        raise FileExistsError(f'Final output already exists: {output}')
    model, selected, run = frozen_candidate(config, selection, root)
    provenance = {'selection_sha256': sha256_file(selection),
                  'model_sha256': selected['model_sha256'],
                  'config_id': selected['config_id'], 'identity': selected['identity'],
                  'validation_model': selected['validation_model'],
                  'seed': model.seed, 'chosen_epoch': selected['chosen_epoch'],
                  'preprocessing': model.preprocessing,
                  'delivery_policy': 'primary-seed validation checkpoint; no retraining'}
    if dry_run:
        return {'status': 'ready', 'provenance': provenance,
                'test_loaded': False, 'output': str(output)}
    test = Path(root) / 'data/digits_test.csv'
    digest = sha256_file(test)
    X, y = cargar(test, Path(root) / 'ej3/cache/digits_test.npz')
    if sha256_file(test) != digest:
        raise ValueError('Test changed while loading.')
    probabilities = np.concatenate([
        model.predecir(preprocess(X[i:i + batch_size], model.preprocessing))
        for i in range(0, len(X), batch_size)])
    actual, predicted = y.argmax(axis=1), probabilities.argmax(axis=1)
    metrics = classification_metrics(actual, predicted)
    metrics['cross_entropy'] = cross_entropy(actual, probabilities) if model.salida == 'softmax' else None
    if sha256_file(selection) != provenance['selection_sha256']:
        raise ValueError('Selection changed during final evaluation.')
    if sha256_file(selection.parent / selected['validation_model']) != provenance['model_sha256']:
        raise ValueError('Checkpoint changed during final evaluation.')
    report = {'schema_version': 1, 'evaluated_at': datetime.now(timezone.utc).isoformat(),
              'provenance': provenance,
              'test': {'path': 'data/digits_test.csv', 'sha256': digest, 'samples': len(X)},
              'results': metrics, 'correct': int(np.sum(actual == predicted)),
              'target_met': metrics['accuracy'] >= 0.98,
              'validation': run['metrics']['validation'],
              'artifacts': {'predictions': 'predictions.npz'},
              'limitations': ['Test metrics must not select hyperparameters or epochs.']}
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'predictions.npz', actual=actual, predicted=predicted,
                        probabilities=probabilities)
    report['predictions_sha256'] = sha256_file(output / 'predictions.npz')
    write_json(output / 'final_evaluation.json', report)
    (output / 'final_evaluation.md').write_text(
        '# Evaluación final — ejercicio 3\n\n'
        f"Modelo congelado: `{selected['config_id']}`, semilla {model.seed}, "
        f"época {selected['chosen_epoch']}. Sin reentrenar.\n\n"
        f"Accuracy de test: **{100 * metrics['accuracy']:.2f} %**, "
        f"{report['correct']}/{len(X)} aciertos. Objetivo ≥98 %: "
        f"{'cumplido' if report['target_met'] else 'no alcanzado'}.\n\n"
        'Métricas por clase, macro-F1, balanced accuracy, matriz de confusión y '
        'procedencia: `final_evaluation.json`. Predicciones: `predictions.npz`.\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true', help='Validate checkpoint without reading test')
    mode.add_argument('--evaluate-test', action='store_true', help='Evaluate frozen model on reserved test')
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--selection', type=Path, default=OUTPUT / 'selection.json')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    report = evaluate(args.config, args.selection, args.output_dir, dry_run=args.dry_run)
    print('Checkpoint verified; test was not loaded.' if args.dry_run
          else f"Final test accuracy: {100 * report['results']['accuracy']:.2f}%")


if __name__ == '__main__':
    main()
