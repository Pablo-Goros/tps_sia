"""Reproducible corruption at the public data boundary."""
import unittest
import numpy as np
from src.dataset import load_dataset
from src.noise import corrupt, training_inputs, make_collection

class NoiseTests(unittest.TestCase):
    def test_extremes_reproducibility_and_collection_alignment(self):
        clean = load_dataset().vectors
        saved = clean.copy()
        np.testing.assert_array_equal(corrupt(clean, 0, np.random.default_rng(1)), clean)
        np.testing.assert_array_equal(corrupt(clean, 1, np.random.default_rng(1)), 1-clean)
        a = training_inputs(clean, [0, .05, .1, .2], 42, 10)
        np.testing.assert_array_equal(a, training_inputs(clean, [0, .05, .1, .2], 42, 10))
        collection = make_collection(clean, [0, .05, .1, .2, .3], 100, 20261110)
        self.assertEqual(collection['inputs'].shape, (12832, 35))
        self.assertEqual(np.count_nonzero(collection['levels'] == 0), 32)
        np.testing.assert_array_equal(collection['targets'], clean[collection['character_indices']])
        other = make_collection(clean, [0, .05, .1, .2, .3], 100, 20261111)
        self.assertFalse(np.array_equal(collection['inputs'], other['inputs']))
        np.testing.assert_array_equal(clean, saved)
        for values, probability in [(clean, -1), (clean, float('nan')), (clean+.2, .1)]:
            with self.assertRaises(ValueError):
                corrupt(values, probability, np.random.default_rng(1))
        with self.assertRaises(ValueError):
            training_inputs(clean, [0, .1], -1, 0)
