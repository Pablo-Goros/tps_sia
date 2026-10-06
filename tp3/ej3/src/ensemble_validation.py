"""Validate one equal-weight ensemble of all confirmed winner seeds; never load test."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tps_sia.tp3.shared.experiments import preprocess, sha256_file, write_json
from tps_sia.tp3.shared.metrics import classification_metrics, cross_entropy
from tps_sia.tp3.shared.mlp import MLP
from .final_evaluation import frozen_candidate
from .protocol import Development, read_json, TP3


def validate(config, selection, output, *, root=TP3, top_configurations=1):
    if top_configurations not in (1, 2, 3):
        raise ValueError('Only the top one, two or three confirmed configurations are supported.')
    selection, output = Path(selection).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Ensemble output already exists: {output}')
    development = Development(config, root=root)
    _, selected, _ = frozen_candidate(config, selection, root=root)
    confirmation = read_json(selection.parent / 'stage_confirmation.json')
    groups = confirmation['ranking'][:top_configurations]
    configurations = {g['config_id']: g['config'] for g in groups}
    rows = [r for r in confirmation['runs'] if r['config_id'] in configurations]
    seeds = development.protocol['confirmation_seeds']
    if (len(groups) != top_configurations
            or any(len([r for r in rows if r['config_id'] == config_id]) != len(seeds)
                   or {r['seed'] for r in rows if r['config_id'] == config_id} != set(seeds)
                   for config_id in configurations)):
        raise ValueError('Ensemble requires exactly every confirmed winner seed.')
    indices = np.asarray(development.data['validation_indices'], dtype=np.int64)
    X, actual = development.X[indices], development.y[indices].argmax(axis=1)
    members, probabilities = [], []
    for row in sorted(rows, key=lambda r: seeds.index(r['seed'])):
        directory = selection.parent / 'runs' / row['run_id']
        report = read_json(directory / 'results.json')
        path = directory / report['artifacts']['best_model']
        digest = sha256_file(path)
        if (report['status'] != 'completed' or report['metadata'] != development.identity
                or report['config'] != {**configurations[row['config_id']], 'model_seed': row['seed']}
                or report['config_id'] != row['config_id']
                or report['chosen_epoch'] != row['chosen_epoch']
                or digest != report['model_sha256']):
            raise ValueError(f'Checkpoint provenance mismatch: {path}')
        model = MLP.cargar(path)
        if (model.seed != row['seed'] or model.salida != 'softmax'
                or model.arquitectura != report['config']['architecture']
                or model.preprocessing != report['preprocessing']
                or model.l2 != report['config'].get('l2', 0)
                or model.augmentation != report['config'].get('augmentation')
                or model.batch_norm != report['config'].get('batch_norm')):
            raise ValueError(f'Model configuration mismatch: {path}')
        predicted = np.concatenate([model.predecir(preprocess(X[i:i+256], model.preprocessing))
                                    for i in range(0, len(X), 256)])
        accuracy = float(np.mean(predicted.argmax(axis=1) == actual))
        if not np.isclose(accuracy, row['accuracy'], atol=1e-12, rtol=0):
            raise ValueError('Recomputed member accuracy differs from recorded checkpoint.')
        if sha256_file(path) != digest:
            raise ValueError('Member changed while predicting.')
        members.append({'seed': row['seed'], 'config_id': row['config_id'],
                        'model': str(path.relative_to(Path(root).resolve())),
                        'model_sha256': digest, 'chosen_epoch': row['chosen_epoch'],
                        'validation_accuracy': accuracy, 'weight': 1 / len(rows)})
        probabilities.append(predicted)
        del model
    average = np.mean(probabilities, axis=0)
    metrics = classification_metrics(actual, average.argmax(axis=1))
    metrics['cross_entropy'] = cross_entropy(actual, average)
    report = {'schema_version': 1, 'protocol': 'ej3-ensemble-validation-v1',
              'scope': 'development validation only; test not loaded',
              'rule': 'equal probability mean of every seed of the top confirmed configurations; no weight tuning',
              'source_selection': str(selection.relative_to(Path(root).resolve())),
              'source_selection_sha256': sha256_file(selection),
              'identity': development.identity, 'config_ids': list(configurations),
              'top_configurations': top_configurations,
              'members': members, 'validation': metrics, 'validation_samples': len(X),
              'correct': int(np.sum(average.argmax(axis=1) == actual)),
              'validation_target_met': metrics['accuracy'] >= 0.98,
              'limitations': ['Validation guided development; final target must be checked on test.',
                              'The ensemble contains multiple models, not a single MLP.']}
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'predictions.npz', actual=actual, probabilities=average,
                        member_probabilities=np.stack(probabilities))
    report['predictions_sha256'] = sha256_file(output / 'predictions.npz')
    write_json(output / 'validation.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=TP3 / 'ej3/configs/search_translation_widths.json')
    parser.add_argument('--selection', type=Path, default=TP3 / 'ej3/results/search_translation_widths/selection.json')
    parser.add_argument('--output-dir', type=Path, default=TP3 / 'ej3/results/ensemble_validation')
    parser.add_argument('--top-configurations', type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    result = validate(args.config, args.selection, args.output_dir,
                      top_configurations=args.top_configurations)
    print(f"Ensemble validation accuracy: {100 * result['validation']['accuracy']:.5f}%")


if __name__ == '__main__':
    main()
