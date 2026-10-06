"""Train-only translations, image integrity and resumable RNG checks."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from tps_sia.tp3.shared.augmentation import translate_images, rotate_images
from tps_sia.tp3.shared.mlp import MLP
from tps_sia.tp3.shared.optimizers import Momentum
from tps_sia.tp3.shared.experiments import validate_config


class AugmentationTests(unittest.TestCase):
    def test_shifts_zero_pad_without_wrapping_or_mutating(self):
        X = np.zeros((2, 784), dtype=np.float32)
        X.reshape(-1, 28, 28)[0, 0, 0] = 0.7
        X.reshape(-1, 28, 28)[1, 27, 27] = 0.9
        original = X.copy()
        rng = type('FixedRng', (), {'integers': lambda self, *a, **k: np.array([[1, 2], [1, 2]])})()
        shifted = translate_images(X, 2, rng).reshape(-1, 28, 28)
        self.assertAlmostEqual(float(shifted[0, 1, 2]), 0.7, places=6)
        self.assertEqual(np.count_nonzero(shifted[0]), 1)
        self.assertEqual(np.count_nonzero(shifted[1]), 0)
        np.testing.assert_array_equal(X, original)

    def test_determinism_and_zero_shift(self):
        X = np.arange(3 * 784).reshape(3, 784)
        np.testing.assert_array_equal(translate_images(X, 2, np.random.default_rng(42)),
                                      translate_images(X, 2, np.random.default_rng(42)))
        np.testing.assert_array_equal(translate_images(X, 0, None), X)
        for shift in (-1, True, 28, 1.5):
            with self.assertRaises(ValueError):
                translate_images(X, shift, np.random.default_rng())

    def test_exact_resume_and_clean_validation(self):
        X = np.zeros((7, 784), dtype=np.float64)
        X[:, 12*28+12] = 1
        y = np.eye(2)[np.arange(7) % 2]
        original = X.copy()
        def model():
            return MLP([784, 4, 2], semilla=2, tamano_lote=3,
                       augmentation={'name': 'translation', 'max_shift': 1},
                       optimizador=Momentum(learning_rate=0.1, momentum=0.9))
        full, split = model(), model()
        options = dict(X_val=X, y_val=y, patience=20)
        with patch('tps_sia.tp3.shared.mlp.translate_images', wraps=translate_images) as transform:
            full.entrenar(X, y, epocas=5, **options)
            self.assertEqual(transform.call_count, 15)  # Three train batches per epoch only.
        with TemporaryDirectory() as directory:
            checkpoint, best = Path(directory)/'checkpoint.npz', Path(directory)/'best.npz'
            split.entrenar(X, y, epocas=5, pause_after=2,
                           checkpoint_path=checkpoint, best_model_path=best, **options)
            restored = MLP.cargar(checkpoint)
            restored.entrenar(X, y, epocas=3, best_model_path=best, **options)
            self.assertEqual(restored.augmentation, full.augmentation)
            for a, b in zip(full.parametros, restored.parametros):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(full.historia.costo_validacion, restored.historia.costo_validacion)
            self.assertEqual(full.rng.bit_generator.state, restored.rng.bit_generator.state)
            self.assertAlmostEqual(full.historia.costo_validacion[-1], full.costo(X, y))
            chosen = MLP.cargar(best)
            self.assertEqual(chosen.augmentation, full.augmentation)
            self.assertAlmostEqual(chosen.costo(X, y), min(restored.historia.costo_validacion))
        np.testing.assert_array_equal(X, original)

    def test_standardization_is_rejected_and_defaults_unchanged(self):
        base = {'dataset': 'synthetic.csv', 'cache': 'synthetic.npz'}
        self.assertNotIn('augmentation', validate_config(base))
        with self.assertRaises(ValueError):
            validate_config({**base, 'preprocessing': {'name': 'standardize'},
                             'augmentation': {'name': 'translation', 'max_shift': 1}})

    def test_rotation_zero_identity_padding_and_integrity(self):
        X = np.ones((2, 784), dtype=np.float32)
        original = X.copy()
        rng = np.random.default_rng(42)
        state = rng.bit_generator.state
        np.testing.assert_array_equal(rotate_images(X, 0, rng), X)
        self.assertEqual(state, rng.bit_generator.state)
        fixed = type('FixedRng', (), {'uniform': lambda self, *args: np.full(args[-1], 5.)})()
        rotated = rotate_images(X, 5, fixed).reshape(-1, 28, 28)
        self.assertEqual(rotated.dtype, X.dtype)
        self.assertEqual(rotated[0, 0, 0], 0)
        self.assertAlmostEqual(float(rotated[0, 14, 14]), 1)
        self.assertTrue(np.all((rotated >= 0) & (rotated <= 1.000001)))
        np.testing.assert_array_equal(X, original)
        zero_angle = type('FixedRng', (), {'uniform': lambda self, *args: np.zeros(args[-1])})()
        np.testing.assert_array_equal(rotate_images(X, 5, zero_angle), X)
        for angle in (-1, 6, True, float('nan')):
            with self.assertRaises(ValueError):
                rotate_images(X, angle, rng)

    def test_rotation_resume_and_clean_validation(self):
        X = np.zeros((7, 784))
        X[:, 8*28+14] = 1
        y = np.eye(2)[np.arange(7) % 2]
        def model():
            return MLP([784, 4, 2], semilla=2, tamano_lote=3,
                       augmentation={'name': 'translation_rotation', 'max_shift': 1,
                                     'max_angle_degrees': 5}, optimizador=Momentum(0.015, 0.9))
        full, partial = model(), model()
        options = dict(X_val=X, y_val=y, patience=20)
        with patch('tps_sia.tp3.shared.mlp.rotate_images', wraps=rotate_images) as transform:
            full.entrenar(X, y, epocas=5, **options)
            self.assertEqual(transform.call_count, 15)
        with TemporaryDirectory() as temp:
            path = Path(temp)/'checkpoint.npz'
            partial.entrenar(X, y, epocas=5, pause_after=2, checkpoint_path=path, **options)
            restored = MLP.cargar(path)
            restored.entrenar(X, y, epocas=3, **options)
        for a, b in zip(full.parametros, restored.parametros):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(full.rng.bit_generator.state, restored.rng.bit_generator.state)
        self.assertEqual(full.historia.costo_validacion, restored.historia.costo_validacion)
        self.assertAlmostEqual(full.historia.costo_validacion[-1], full.costo(X, y))

    def test_rotation_configuration_bound(self):
        base = {'dataset': 'synthetic.csv', 'cache': 'synthetic.npz'}
        for angle in (0, 6, True):
            with self.assertRaises(ValueError):
                validate_config({**base, 'augmentation': {'name': 'translation_rotation',
                                  'max_shift': 1, 'max_angle_degrees': angle}})


if __name__ == '__main__':
    unittest.main()
