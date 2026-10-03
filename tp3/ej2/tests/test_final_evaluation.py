"""Final-evaluation checks with synthetic data only (the real test CSV is never read).

    python -m unittest -v tps_sia.tp3.ej2.tests.test_final_evaluation
"""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from tps_sia.tp3.ej2.src import final_evaluation
from tps_sia.tp3.ej2.src.final_evaluation import evaluate_predictions, main
from tps_sia.tp3.shared.experiments import sha256_file
from tps_sia.tp3.shared.mlp import MLP
from tps_sia.tp3.shared.tests.test_experiments import synthetic_csv

OUTPUTS = ('final_evaluation.json', 'final_evaluation.md', 'confusion_matrix.png', 'per_class_recall.png')


class MetricTests(unittest.TestCase):
    def test_accuracy_with_and_without_the_unseen_class(self):
        actual = np.array([0, 0, 1, 8, 8, 9])
        predicted = np.array([0, 1, 1, 0, 9, 9])
        result = evaluate_predictions(actual, predicted)
        self.assertAlmostEqual(result['accuracy_all'], 3 / 6)
        self.assertAlmostEqual(result['accuracy_without_unseen'], 3 / 4)
        self.assertEqual((result['samples_all'], result['samples_without_unseen'], result['unseen_samples']), (6, 4, 2))
        self.assertEqual(result['predictions_of_unseen'], 0)
        self.assertEqual(result['unseen_assigned_to'], {'0': 1, '9': 1})
        eight = result['per_class'][8]
        self.assertEqual((eight['support'], eight['predictions'], eight['recall']), (2, 0, 0.0))
        self.assertIsNone(eight['precision'])               # 0/0: no prediction of 8.

    def test_predicted_unseen_class_is_counted(self):
        result = evaluate_predictions(np.array([1, 2, 8]), np.array([8, 2, 8]))
        self.assertEqual(result['predictions_of_unseen'], 2)
        self.assertAlmostEqual(result['per_class'][8]['precision'], 0.5)
        self.assertAlmostEqual(result['accuracy_without_unseen'], 0.5)   # The 1 predicted as 8 still counts.

    def test_row_normalized_confusion(self):
        result = evaluate_predictions(np.array([0, 0, 0, 1]), np.array([0, 0, 1, 1]))
        rows = result['confusion_matrix_row_normalized']
        self.assertAlmostEqual(rows[0][0], 2 / 3)
        self.assertAlmostEqual(sum(rows[0]), 1.0)
        self.assertIsNone(rows[5])                           # No samples of 5: undefined row.


def build_results_dir(root: Path, sha=None):
    """Synthetic selection.json, stage_7.json and candidate checkpoint."""
    run_dir = root / 'v2' / 'runs' / 'cfg-seed-42'
    run_dir.mkdir(parents=True)
    model = MLP([784, 4, 10], semilla=0)
    model.guardar(run_dir / 'best_model.npz')
    (run_dir / 'results.json').write_text(json.dumps(
        {'chosen_epoch': 3, 'metrics': {'validation': {'accuracy': 0.9, 'loss': 0.3}}}), encoding='utf-8')
    (root / 'v2' / 'stage_7.json').write_text(json.dumps({'ranking': [
        {'config_id': 'cfg', 'seeds': [0, 1, 42], 'accuracy_mean': 0.91, 'accuracy_std': 0.01}]}), encoding='utf-8')
    (root / 'v2' / 'selection.json').write_text(json.dumps({
        'config_id': 'cfg', 'candidate_model': 'runs/cfg-seed-42/best_model.npz',
        'model_sha256': sha or sha256_file(run_dir / 'best_model.npz'), 'search_sha256': 's',
        'protocol': 'p', 'delivery_seed': 42, 'dataset': {'validation_samples': 10},
        'config': {'architecture': [784, 4, 10], 'activation': 'tanh'}}), encoding='utf-8')
    synthetic_csv(root / 'digits_test.csv')                 # Synthetic: digits 0, 5 and 8.
    return ['--results-dir', str(root / 'v2'), '--test-csv', str(root / 'digits_test.csv'),
            '--cache-dir', str(root / 'cache')]


class CommandTests(unittest.TestCase):
    def test_end_to_end_and_no_silent_overwrite(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = build_results_dir(root)
            main(args)
            for name in OUTPUTS:
                self.assertTrue((root / 'v2' / name).exists(), name)
            report = json.loads((root / 'v2' / 'final_evaluation.json').read_text(encoding='utf-8'))
            results = report['results']
            self.assertEqual((results['samples_all'], results['unseen_samples']), (30, 10))
            self.assertEqual(report['test']['path'], 'digits_test.csv')
            self.assertIn('Test, sin el 8', (root / 'v2' / 'final_evaluation.md').read_text(encoding='utf-8'))
            before = (root / 'v2' / 'final_evaluation.json').read_bytes()
            with patch.object(final_evaluation, 'cargar', side_effect=AssertionError('test read again')):
                with self.assertRaises(SystemExit):
                    main(args)
            self.assertEqual((root / 'v2' / 'final_evaluation.json').read_bytes(), before)
            main(args + ['--overwrite'])                     # Deliberate repetition only.
            self.assertTrue((root / 'v2' / 'final_evaluation.json').exists())

    def test_wrong_checkpoint_hash_is_rejected_before_reading_test(self):
        with TemporaryDirectory() as tmp:
            args = build_results_dir(Path(tmp), sha='0' * 64)
            with patch.object(final_evaluation, 'cargar', side_effect=AssertionError('test read')):
                with self.assertRaisesRegex(ValueError, 'SHA-256'):
                    main(args)
            self.assertFalse((Path(tmp) / 'v2' / 'final_evaluation.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
