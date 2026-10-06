"""Numerical gradients and exact resume checks for the ej3 L2 extension."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from tps_sia.tp3.shared.mlp import MLP
from tps_sia.tp3.shared.optimizers import Momentum
from tps_sia.tp3.shared.experiments import validate_config, config_identity


class RegularizationTests(unittest.TestCase):
    def setUp(self):
        self.X = np.random.default_rng(12).normal(size=(7, 3))
        self.y = np.eye(2)[np.arange(7) % 2]

    def model(self, strength=0.001):
        return MLP([3, 4, 2], semilla=2, tamano_lote=3, l2=strength,
                   optimizador=Momentum(learning_rate=0.1, momentum=0.9))

    def test_gradient_matches_regularized_objective_and_excludes_biases(self):
        model, base = self.model(), self.model(0)
        for output in ('softmax', 'logistica'):
            model.salida = base.salida = output
            grads, plain = model.backprop(self.X, self.y), base.backprop(self.X, self.y)
            for i, (parameter, gradient) in enumerate(zip(model.parametros, grads)):
                difference = gradient - plain[i]
                np.testing.assert_allclose(difference, model.l2 * parameter if i < len(model.pesos)
                                           else np.zeros_like(parameter), atol=1e-15)
                for index in np.ndindex(parameter.shape):
                    original = parameter[index]
                    parameter[index] = original + 1e-6
                    upper = model.training_objective(self.X, self.y)
                    parameter[index] = original - 1e-6
                    lower = model.training_objective(self.X, self.y)
                    parameter[index] = original
                    self.assertAlmostEqual(gradient[index], (upper-lower)/2e-6, places=7)

    def test_validation_loss_unpenalized_and_resume_exact(self):
        full, split = self.model(), self.model()
        options = dict(X_val=self.X, y_val=self.y, patience=20)
        full.entrenar(self.X, self.y, epocas=5, **options)
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'checkpoint.npz'
            best = Path(directory) / 'best.npz'
            split.entrenar(self.X, self.y, epocas=5, pause_after=2,
                           checkpoint_path=checkpoint, best_model_path=best, **options)
            restored = MLP.cargar(checkpoint)
            self.assertEqual(restored.l2, 0.001)
            restored.entrenar(self.X, self.y, epocas=3, best_model_path=best, **options)
            for a, b in zip(full.parametros, restored.parametros):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(full.historia.training_objective, restored.historia.training_objective)
            self.assertEqual(full.historia.costo_validacion, restored.historia.costo_validacion)
            self.assertAlmostEqual(restored.historia.costo_validacion[-1], restored.costo(self.X, self.y))
            for loss, penalty, objective in zip(restored.historia.costo,
                                               restored.historia.l2_penalty,
                                               restored.historia.training_objective):
                self.assertAlmostEqual(objective, loss + penalty)
            chosen = MLP.cargar(best)
            self.assertEqual(chosen.l2, restored.l2)
            self.assertAlmostEqual(chosen.costo(self.X, self.y), min(restored.historia.costo_validacion))

    def test_invalid_strength_and_default_config_identity(self):
        for strength in (-1, True, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.model(strength)
        base = {'dataset': 'synthetic.csv', 'cache': 'synthetic.npz'}
        original = validate_config(base)
        self.assertNotIn('l2', original)
        regularized = validate_config({**base, 'l2': 0.001})
        self.assertNotEqual(config_identity(original), config_identity(regularized))


if __name__ == '__main__':
    unittest.main()
