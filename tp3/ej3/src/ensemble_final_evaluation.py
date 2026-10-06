"""Final test evaluation of the already selected, frozen equal-weight ensemble."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tps_sia.tp3.shared.digit_dataset import cargar
from tps_sia.tp3.shared.experiments import config_identity, recorded_config, preprocess, sha256_file, write_json
from tps_sia.tp3.shared.metrics import classification_metrics, cross_entropy
from tps_sia.tp3.shared.mlp import MLP
from .final_evaluation import frozen_candidate
from .protocol import Development, read_json, TP3

CONFIG = TP3 / 'ej3/configs/search_translation_widths.json'
ENSEMBLE = TP3 / 'ej3/results/ensemble_validation_top3/validation.json'


def frozen_ensemble(config, ensemble, root=TP3):
    root, ensemble = Path(root).resolve(), Path(ensemble).resolve()
    manifest = read_json(ensemble)
    if manifest.get('protocol') == 'ej3-ensemble-rate015-extension-v1':
        return frozen_extension(config, ensemble, manifest, root)
    digest = sha256_file(ensemble)
    development = Development(config, root=root)
    source = root / manifest['source_selection']
    _, selected, _ = frozen_candidate(config, source, root=root)
    if (manifest['protocol'] != 'ej3-ensemble-validation-v1'
            or manifest['identity'] != development.identity
            or sha256_file(source) != manifest['source_selection_sha256']
            or manifest['top_configurations'] not in (1, 2, 3)):
        raise ValueError('Ensemble source provenance changed.')
    confirmation = read_json(source.parent / 'stage_confirmation.json')
    groups = confirmation['ranking'][:manifest['top_configurations']]
    configurations = {g['config_id']: g['config'] for g in groups}
    seeds = development.protocol['confirmation_seeds']
    expected = {(cid, seed) for cid in configurations for seed in seeds}
    members = manifest['members']
    if (list(configurations) != manifest['config_ids']
            or any(g['status'] != 'completed' for g in groups)
            or len(members) != len(expected)
            or {(m['config_id'], m['seed']) for m in members} != expected
            or any(m['weight'] != 1 / len(expected) for m in members)):
        raise ValueError('Ensemble membership or equal weights changed.')
    predictions = ensemble.parent / 'predictions.npz'
    if sha256_file(predictions) != manifest['predictions_sha256']:
        raise ValueError('Validation predictions changed.')
    frozen = []
    for member in members:
        row = next(r for r in confirmation['runs']
                   if (r['config_id'], r['seed']) == (member['config_id'], member['seed']))
        directory = source.parent / 'runs' / row['run_id']
        report = read_json(directory / 'results.json')
        path = (root / member['model']).resolve()
        if (path != (directory / report['artifacts']['best_model']).resolve()
                or report['status'] != 'completed'
                or report['metadata'] != selected['identity']
                or report['config'] != {**configurations[member['config_id']], 'model_seed': member['seed']}
                or report['config_id'] != member['config_id']
                or report['chosen_epoch'] != member['chosen_epoch']
                or row['chosen_epoch'] != member['chosen_epoch']
                or row['accuracy'] != member['validation_accuracy']
                or report['model_sha256'] != member['model_sha256']
                or sha256_file(path) != member['model_sha256']):
            raise ValueError(f'Ensemble checkpoint changed: {path}')
        model = MLP.cargar(path)
        if (model.seed != member['seed'] or model.salida != 'softmax'
                or model.arquitectura != report['config']['architecture']
                or model.preprocessing != report['preprocessing']
                or model.l2 != report['config'].get('l2', 0)
                or model.augmentation != report['config'].get('augmentation')
                or model.batch_norm != report['config'].get('batch_norm')):
            raise ValueError('Ensemble checkpoint configuration changed.')
        frozen.append({**member, 'config': report['config'], 'preprocessing': model.preprocessing})
        del model
    return manifest, {'ensemble_manifest_sha256': digest,
                      'source_selection_sha256': sha256_file(source),
                      'identity': development.identity, 'members': frozen,
                      'combination': 'equal probability mean; no retraining or weight tuning'}


def frozen_extension(config, ensemble, manifest, root):
    base_path = root / 'ej3/results/ensemble_validation_top3/validation.json'
    base_manifest, base_frozen = frozen_ensemble(config, base_path, root)
    development = Development(root / 'ej3/configs/search_translation_rates_fine.json', root=root)
    overrides_path = root / 'ej3/configs/translation_rate_015.json'
    seeds = manifest['added_seeds']
    if (not seeds or len(set(seeds)) != len(seeds) or any(s not in (42, 0, 1) for s in seeds)
            or sha256_file(base_path) != manifest['base_manifest_sha256']
            or base_manifest['predictions_sha256'] != manifest['base_predictions_sha256']
            or development.identity != manifest['extension_protocol']
            or sha256_file(overrides_path) != manifest['extension_config_sha256']
            or any(development.identity[k] != base_manifest['identity'][k]
                   for k in ('dataset_sha256', 'split_sha256', 'split_manifest_sha256'))):
        raise ValueError('Extension source provenance changed.')
    members = manifest['members']
    n = len(base_frozen['members']) + len(seeds)
    if len(members) != n or any(m['weight'] != 1 / n for m in members):
        raise ValueError('Extension membership or equal weights changed.')
    frozen = []
    for member, original in zip(members, base_frozen['members']):
        if any(member[k] != original[k] for k in
               ('model', 'model_sha256', 'seed', 'config_id', 'chosen_epoch', 'validation_accuracy')):
            raise ValueError('Original ensemble members changed in extension.')
        frozen.append({**original, 'weight': 1 / n})
    base = copy.deepcopy(development.reference)
    for key, value in read_json(overrides_path).items():
        if isinstance(value, dict) and key in ('optimizer', 'stopping', 'preprocessing'):
            base[key].update(value)
        else:
            base[key] = value
    for member, seed in zip(members[len(frozen):], seeds):
        expected = development.config(base, seed)
        cid = config_identity(expected, path_root=root)
        directory = root / 'ej3/results/search_translation_rates_fine/runs' / f'{cid}-seed-{seed}'
        report = read_json(directory / 'results.json')
        path = (root / member['model']).resolve()
        if (member['seed'] != seed or member['config_id'] != cid
                or report['status'] != 'completed' or report['metadata'] != development.identity
                or report['config'] != recorded_config(expected, root)
                or path != (directory / report['artifacts']['best_model']).resolve()
                or report['chosen_epoch'] != member['chosen_epoch']
                or report['metrics']['validation']['accuracy'] != member['validation_accuracy']
                or report['model_sha256'] != member['model_sha256']
                or sha256_file(path) != member['model_sha256']):
            raise ValueError('Extension checkpoint provenance changed.')
        model = MLP.cargar(path)
        if (model.seed != seed or model.salida != 'softmax'
                or model.arquitectura != expected['architecture']
                or model.preprocessing != report['preprocessing']
                or model.augmentation != expected['augmentation'] or model.l2 != expected.get('l2', 0)
                or model.batch_norm != expected.get('batch_norm')):
            raise ValueError('Extension checkpoint configuration changed.')
        frozen.append({**member, 'config': report['config'], 'preprocessing': model.preprocessing})
        del model
    if sha256_file(ensemble.parent / 'predictions.npz') != manifest['predictions_sha256']:
        raise ValueError('Extension validation predictions changed.')
    normalized = {**manifest, 'source_selection': base_manifest['source_selection']}
    return normalized, {'ensemble_manifest_sha256': sha256_file(ensemble),
                        'source_selection_sha256': base_frozen['source_selection_sha256'],
                        'identity': base_frozen['identity'], 'extension_identity': development.identity,
                        'members': frozen, 'combination': 'equal probability mean; no retraining or weight tuning'}


def evaluate(config=CONFIG, ensemble=ENSEMBLE, output=None, *, dry_run=False, root=TP3):
    ensemble = Path(ensemble).resolve()
    output = Path(output).resolve() if output else ensemble.parent / 'final'
    if output.exists():
        raise FileExistsError(f'Final output already exists: {output}')
    manifest, frozen = frozen_ensemble(config, ensemble, root=root)
    if dry_run:
        return {'status': 'ready', 'members': len(frozen['members']), 'test_loaded': False}
    # Persist the exact delivery decision BEFORE opening any test data.
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'frozen_ensemble.json', frozen)
    test = Path(root) / 'data/digits_test.csv'
    digest = sha256_file(test)
    X, y = cargar(test, Path(root) / 'ej3/cache/digits_test.npz')
    if sha256_file(test) != digest:
        raise ValueError('Test changed while loading.')
    probabilities = np.zeros_like(y, dtype=np.float64)
    for member in frozen['members']:
        path = Path(root) / member['model']
        if sha256_file(path) != member['model_sha256']:
            raise ValueError('Frozen member changed before inference.')
        model = MLP.cargar(path)
        for i in range(0, len(X), 256):
            probabilities[i:i+256] += model.predecir(
                preprocess(X[i:i+256], member['preprocessing']))
        if sha256_file(path) != member['model_sha256']:
            raise ValueError('Frozen member changed during inference.')
        del model
    probabilities /= len(frozen['members'])
    if sha256_file(ensemble) != frozen['ensemble_manifest_sha256']:
        raise ValueError('Ensemble selection changed during evaluation.')
    if sha256_file(Path(root) / manifest['source_selection']) != frozen['source_selection_sha256']:
        raise ValueError('Source selection changed during evaluation.')
    actual, predicted = y.argmax(axis=1), probabilities.argmax(axis=1)
    metrics = classification_metrics(actual, predicted)
    metrics['cross_entropy'] = cross_entropy(actual, probabilities)
    np.savez_compressed(output / 'predictions.npz', actual=actual, predicted=predicted,
                        probabilities=probabilities)
    report = {'schema_version': 1, 'evaluated_at': datetime.now(timezone.utc).isoformat(),
              'provenance': frozen, 'frozen_ensemble_sha256': sha256_file(output / 'frozen_ensemble.json'),
              'test': {'path': 'data/digits_test.csv', 'sha256': digest, 'samples': len(X)},
              'results': metrics, 'correct': int(np.sum(actual == predicted)),
              'target_met': metrics['accuracy'] >= 0.98, 'validation': manifest['validation'],
              'artifacts': {'predictions': 'predictions.npz', 'frozen_ensemble': 'frozen_ensemble.json'},
              'predictions_sha256': sha256_file(output / 'predictions.npz'),
              'limitations': ['Test must not guide subsequent model selection.',
                              'Delivery is an ensemble of MLPs, not one MLP.']}
    if manifest.get('protocol') == 'ej3-ensemble-rate015-extension-v1':
        report['limitations'].append(
            'This test was already evaluated for the original nine-model ensemble; it is not a new independent holdout.')
    write_json(output / 'final_evaluation.json', report)
    (output / 'final_evaluation.md').write_text(
        '# Evaluación final del ensamble — ejercicio 3\n\n'
        f"Ensamble congelado de {len(frozen['members'])} MLP, pesos iguales, sin reentrenar.\n\n"
        f"Accuracy de test: **{100 * metrics['accuracy']:.5f} %**, {report['correct']}/{len(X)} aciertos. "
        f"Objetivo ≥98 %: {'cumplido' if report['target_met'] else 'no alcanzado'}.\n\n"
        f"Accuracy de validación: {100 * manifest['validation']['accuracy']:.5f} %.\n\n"
        'Procedencia, métricas por clase y matriz de confusión en `final_evaluation.json`.\n',
        encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--evaluate-test', action='store_true')
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--ensemble', type=Path, default=ENSEMBLE)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    r = evaluate(args.config, args.ensemble, args.output_dir, dry_run=args.dry_run)
    print('Frozen ensemble verified; test not loaded.' if args.dry_run else
          f"Final ensemble test accuracy: {100 * r['results']['accuracy']:.5f}% ({r['correct']}/{r['test']['samples']})")


if __name__ == '__main__':
    main()
