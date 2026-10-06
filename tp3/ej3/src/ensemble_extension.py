"""Validation-only equal-weight extension with explicit rate0.015 seeds; no test."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from tps_sia.tp3.shared.experiments import config_identity, recorded_config, preprocess, sha256_file, write_json
from tps_sia.tp3.shared.metrics import classification_metrics, cross_entropy
from tps_sia.tp3.shared.mlp import MLP
from .ensemble_final_evaluation import frozen_ensemble, CONFIG, ENSEMBLE
from .protocol import Development, TP3, read_json


def extend(seeds, output, root=TP3):
    if not seeds or len(set(seeds)) != len(seeds) or any(s not in (42, 0, 1) for s in seeds):
        raise ValueError('Specify unique explicit seeds among42/0/1.')
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    ensemble = root / 'ej3/results/ensemble_validation_top3/validation.json'
    manifest, frozen = frozen_ensemble(root / 'ej3/configs/search_translation_widths.json', ensemble, root)
    development = Development(root / 'ej3/configs/search_translation_rates_fine.json', root=root)
    for key in ('dataset_sha256', 'split_sha256', 'split_manifest_sha256'):
        if development.identity[key] != manifest['identity'][key]:
            raise ValueError('Extension uses a different dataset or partition.')
    indices = np.asarray(development.data['validation_indices'], dtype=np.int64)
    X, actual = development.X[indices], development.y[indices].argmax(axis=1)
    with np.load(ensemble.parent / 'predictions.npz') as saved:
        if not np.array_equal(actual, saved['actual']):
            raise ValueError('Cached validation rows changed.')
        probabilities = list(saved['member_probabilities'])
        original_average = saved['probabilities'].copy()
    if len(probabilities) != len(frozen['members']) or not np.allclose(np.mean(probabilities, axis=0), original_average):
        raise ValueError('Original cached member predictions disagree.')
    members = copy.deepcopy(manifest['members'])
    overrides_path = root / 'ej3/configs/translation_rate_015.json'
    overrides = read_json(overrides_path)
    base = copy.deepcopy(development.reference)
    for key, value in overrides.items():
        if isinstance(value, dict) and key in ('optimizer', 'stopping', 'preprocessing'):
            base[key].update(value)
        else:
            base[key] = value
    for seed in seeds:
        config = development.config(base, seed)
        cid = config_identity(config, path_root=root)
        directory = root / 'ej3/results/search_translation_rates_fine/runs' / f'{cid}-seed-{seed}'
        report = read_json(directory / 'results.json')
        if (report['status'] != 'completed' or report['metadata'] != development.identity
                or report['config'] != recorded_config(config, root) or report['config_id'] != cid
                or report['dataset']['sha256'] != manifest['identity']['dataset_sha256']
                or report['dataset']['split_sha256'] != manifest['identity']['split_sha256']):
            raise ValueError(f'Seed{seed} is incomplete or belongs to another experiment.')
        path = directory / report['artifacts']['best_model']
        digest = sha256_file(path)
        if digest != report['model_sha256']:
            raise ValueError('Extension checkpoint hash changed.')
        model = MLP.cargar(path)
        if (model.seed != seed or model.salida != 'softmax'
                or model.arquitectura != config['architecture']
                or model.preprocessing != report['preprocessing']
                or model.augmentation != config['augmentation'] or model.l2 != config.get('l2', 0)
                or model.batch_norm != config.get('batch_norm')):
            raise ValueError('Extension model configuration changed.')
        probability = np.concatenate([model.predecir(preprocess(X[i:i+256], model.preprocessing))
                                      for i in range(0, len(X), 256)])
        accuracy = float(np.mean(probability.argmax(axis=1) == actual))
        if not np.isclose(accuracy, report['metrics']['validation']['accuracy'], atol=1e-12, rtol=0):
            raise ValueError('Extension member validation accuracy changed.')
        if sha256_file(path) != digest:
            raise ValueError('Extension checkpoint changed during prediction.')
        probabilities.append(probability)
        members.append({'seed': seed, 'config_id': cid, 'model': str(path.relative_to(root)),
                        'model_sha256': digest, 'chosen_epoch': report['chosen_epoch'],
                        'validation_accuracy': accuracy})
        del model
    average = np.mean(probabilities, axis=0)
    for member in members:
        member['weight'] = 1 / len(members)
    metrics = classification_metrics(actual, average.argmax(axis=1))
    metrics['cross_entropy'] = cross_entropy(actual, average)
    original_correct = original_average.argmax(axis=1) == actual
    new_correct = average.argmax(axis=1) == actual
    report = {'schema_version': 1, 'protocol': 'ej3-ensemble-rate015-extension-v1',
              'scope': 'validation only; test never loaded', 'rule': 'equal probability mean of every listed member',
              'base_manifest_sha256': sha256_file(ensemble), 'base_predictions_sha256': manifest['predictions_sha256'],
              'extension_protocol': development.identity, 'extension_config_sha256': sha256_file(overrides_path),
              'added_seeds': seeds, 'members': members, 'validation': metrics,
              'correct': int(new_correct.sum()), 'validation_samples': len(actual),
              'validation_target_met': metrics['accuracy'] >= 0.98,
              'comparison': {'original_accuracy': manifest['validation']['accuracy'],
                             'original_correct': int(original_correct.sum()),
                             'corrected_errors': int(np.sum(~original_correct & new_correct)),
                             'new_errors': int(np.sum(original_correct & ~new_correct))},
              'limitations': ['Explicit available seeds only; no claim of complete three-seed confirmation.',
                              'The original test result does not apply to this new ensemble.']}
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'predictions.npz', actual=actual, probabilities=average,
                        member_probabilities=np.stack(probabilities))
    report['predictions_sha256'] = sha256_file(output / 'predictions.npz')
    write_json(output / 'validation.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    r = extend(args.seeds, args.output_dir)
    print(f"Validation ensemble ({len(r['members'])} models): {100*r['validation']['accuracy']:.5f}% "
          f"({r['correct']}/{r['validation_samples']})")
    print(r['comparison'])


if __name__ == '__main__':
    main()
