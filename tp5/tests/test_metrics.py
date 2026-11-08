import unittest
import numpy as np
from src.metrics import reconstruction_metrics


class MetricsTests(unittest.TestCase):
    def test_pixel_tolerance_and_inclusive_threshold(self):
        target = np.zeros((3, 35))
        prediction = target.copy()
        prediction[1, 0] = 0.5
        prediction[2, :2] = 1
        metrics = reconstruction_metrics(target, prediction)
        self.assertEqual(metrics['pixel_errors'], [0, 1, 2])
        self.assertEqual(metrics['max_pixel_errors'], 2)
        self.assertAlmostEqual(metrics['exact_fraction'], 1/3)
        self.assertAlmostEqual(metrics['within_tolerance_fraction'], 2/3)
        self.assertFalse(metrics['all_within_tolerance'])
        self.assertAlmostEqual(metrics['mse_per_pixel'], 2.25/105)
        self.assertAlmostEqual(metrics['training_loss'], 2.25/6)
        self.assertTrue(reconstruction_metrics(target, target)['all_within_tolerance'])
