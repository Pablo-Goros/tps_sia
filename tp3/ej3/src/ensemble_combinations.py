"""Validation-only search of complete three-seed configuration groups, from cached predictions."""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from tps_sia.tp3.shared.experiments import sha256_file, write_json
from tps_sia.tp3.shared.metrics import classification_metrics, cross_entropy
from .ensemble_final_evaluation import frozen_ensemble
from .protocol import TP3, read_json


def compare(output, root=TP3):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    source = root / 'ej3/results/ensemble_rate015_seeds42_0_1/validation.json'
    manifest, frozen = frozen_ensemble(root / 'ej3/configs/search_translation_widths.json', source, root)
    members = manifest['members']
    groups = list(dict.fromkeys(m['config_id'] for m in members))
    if len(groups) != 4 or any(sorted(m['seed'] for m in members if m['config_id'] == cid) != [0, 1, 42]
                               for cid in groups):
        raise ValueError('Require exactly four configurations with all three planned seeds.')
    with np.load(source.parent / 'predictions.npz') as data:
        actual = data['actual'].copy()
        predictions = data['member_probabilities'].copy()
    if (predictions.shape != (12, len(actual), 10) or not np.all(np.isfinite(predictions))
            or not np.allclose(predictions.sum(axis=2), 1, atol=1e-10)):
        raise ValueError('Invalid cached probabilities.')
    rows = []
    best_predictions = None
    for n in range(1, 5):
        for combination in itertools.combinations(groups, n):
            indices = [i for i, member in enumerate(members) if member['config_id'] in combination]
            average = predictions[indices].mean(axis=0)
            metrics = classification_metrics(actual, average.argmax(axis=1))
            metrics['cross_entropy'] = cross_entropy(actual, average)
            candidate = {'config_ids': list(combination), 'model_count': len(indices),
                         'correct': int(np.sum(average.argmax(axis=1) == actual)),
                         'validation': metrics, 'member_indices': indices}
            rows.append(candidate)
    rows.sort(key=lambda row: (-row['validation']['accuracy'], row['validation']['cross_entropy'],
                              row['model_count'], row['config_ids']))
    winner = rows[0]
    selected_members = [{**members[i], 'weight': 1 / winner['model_count']}
                        for i in winner['member_indices']]
    best_predictions = predictions[winner['member_indices']].mean(axis=0)
    configurations = {cid: next(m['config'] for m in frozen['members'] if m['config_id'] == cid)
                      for cid in groups}
    reference = read_json(root / 'ej3/results/ensemble_rate015_seed42/validation.json')
    report = {'schema_version': 1, 'protocol': 'ej3-ensemble-config-subsets-v1',
              'scope': 'validation only; test not loaded', 'source_manifest_sha256': sha256_file(source),
              'source_predictions_sha256': manifest['predictions_sha256'],
              'rule': 'all 15 nonempty subsets of four configurations, always all three seeds; equal model weights',
              'ranking_rule': ['accuracy descending', 'cross entropy ascending', 'model count ascending', 'config ids'],
              'configurations': configurations, 'ranking': rows, 'winner': winner,
              'validation_samples': len(actual),
              'reference_ten_models': {'accuracy': reference['validation']['accuracy'], 'correct': reference['correct']},
              'limitations': ['Selected among fifteen combinations on repeatedly used development validation.',
                              'No new independent test evaluation; previous test results do not apply to this composition.']}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'comparison.json', report)
    selection = {'schema_version': 1, 'protocol': report['protocol'], 'scope': report['scope'],
                 'members': selected_members, 'validation': winner['validation'],
                 'correct': winner['correct'], 'validation_samples': len(actual),
                 'source_manifest_sha256': report['source_manifest_sha256'],
                 'source_predictions_sha256': report['source_predictions_sha256'],
                 'limitations': report['limitations']}
    np.savez_compressed(output / 'predictions.npz', actual=actual, probabilities=best_predictions)
    selection['predictions_sha256'] = sha256_file(output / 'predictions.npz')
    write_json(output / 'selection.json', selection)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=TP3 / 'ej3/results/ensemble_combinations')
    args = parser.parse_args()
    report = compare(args.output_dir)
    for row in report['ranking'][:5]:
        descriptions = [(report['configurations'][cid]['architecture'][1],
                         report['configurations'][cid]['optimizer']['learning_rate']) for cid in row['config_ids']]
        print(f"{descriptions}: {row['model_count']} models, {100*row['validation']['accuracy']:.5f}%, "
              f"{row['correct']}/{report['validation_samples']}, loss={row['validation']['cross_entropy']:.6f}")


if __name__ == '__main__':
    main()
