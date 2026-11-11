import unittest
import numpy as np
from src.denoise_evaluation import recovery_metrics

class RecoveryTests(unittest.TestCase):
    def test_exact_improved_worse_and_inclusive_threshold(self):
        target = np.zeros((3, 35))
        noisy = target.copy(); noisy[0, :2] = 1; noisy[1, 0] = 1
        prediction = target.copy(); prediction[0, 0] = .5; prediction[1, :2] = 1
        result = recovery_metrics(target, noisy, prediction)
        self.assertEqual(result['input_pixel_errors'], [2, 1, 0])
        self.assertEqual(result['pixel_errors'], [1, 2, 0])
        self.assertEqual(result['mean_error_reduction'], 0)
        self.assertEqual(result['improved_fraction'], 1/3)
        self.assertEqual(result['worsened_fraction'], 1/3)
        self.assertEqual(result['unchanged_fraction'], 1/3)
        self.assertEqual(result['exact_fraction'], 1/3)
