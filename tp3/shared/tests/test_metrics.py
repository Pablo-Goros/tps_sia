"""Hand-computed class metrics, absent classes and probability validation."""
import json
import unittest

import numpy as np

from tps_sia.tp3.shared.metrics import classification_metrics, confusion_matrix, cross_entropy


class MetricsTests(unittest.TestCase):
    def test_hand_computed_and_absent_classes(self):
        r = classification_metrics(np.array([0, 0, 1, 1, 1]), np.array([0, 1, 1, 1, 2]))
        self.assertEqual(np.asarray(r['confusion_matrix']).shape, (10, 10))
        self.assertEqual(r['accuracy'], 3 / 5)
        self.assertEqual(r['per_class'][0]['precision'], 1)
        self.assertEqual(r['per_class'][0]['recall'], 1 / 2)
        self.assertEqual(r['per_class'][1]['f1'], 2 / 3)
        self.assertAlmostEqual(r['balanced_accuracy'], (1 / 2 + 2 / 3) / 2)
        self.assertAlmostEqual(r['macro_f1'], 2 / 3)
        self.assertEqual(r['average_classes']['macro_f1'], [0, 1])
        self.assertIsNone(r['per_class'][2]['recall'])
        self.assertEqual(r['per_class'][2]['precision'], 0)
        self.assertEqual(r['per_class'][2]['f1'], 0)
        self.assertIsNone(r['per_class'][8]['precision'])
        self.assertIsNone(r['per_class'][8]['recall'])
        self.assertIsNone(r['per_class'][8]['f1'])
        json.dumps(r, allow_nan=False)

    def test_no_predictions_for_observed_class(self):
        r = classification_metrics(np.array([0, 1]), np.array([0, 0]))
        self.assertIsNone(r['per_class'][1]['precision'])
        self.assertEqual(r['per_class'][1]['recall'], 0)
        self.assertEqual(r['per_class'][1]['f1'], 0)
        self.assertEqual(r['macro_f1'], 1 / 3)

    def test_confusion_rejects_invalid_labels(self):
        for a, p in [([], []), ([0], [10]), ([0.0], [0]), ([True], [0]), ([0, 1], [0]), ([[0]], [[0]])]:
            with self.subTest(actual=a, predicted=p), self.assertRaises(ValueError):
                confusion_matrix(np.asarray(a), np.asarray(p))

    def test_cross_entropy(self):
        probs = np.zeros((2, 10))
        probs[0, :2], probs[1, :2] = [0.8, 0.2], [0.3, 0.7]
        self.assertAlmostEqual(cross_entropy(np.array([0, 1]), probs), -np.log([0.8, 0.7]).mean())
        probs[0, :2] = [0, 1]
        self.assertTrue(np.isfinite(cross_entropy(np.array([0, 1]), probs)))
        for bad in (probs * 2, np.full((2, 10), np.nan), np.zeros((2, 9))):
            with self.assertRaises(ValueError):
                cross_entropy(np.array([0, 1]), bad)


if __name__ == '__main__':
    unittest.main()
