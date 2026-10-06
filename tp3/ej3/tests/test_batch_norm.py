from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from tps_sia.tp3.shared.mlp import MLP
from tps_sia.tp3.shared.optimizers import Momentum
from tps_sia.tp3.shared.experiments import validate_config


class BatchNormTests(unittest.TestCase):
    def test_multilayer_numerical_gradient_and_stateless_backprop(self):
        X = np.random.default_rng(1).normal(size=(6, 3))
        y = np.eye(2)[np.arange(6) % 2]
        for output in ('softmax', 'logistica'):
            model = MLP([3, 4, 3, 2], semilla=3, batch_norm={}, salida=output, l2=0.001)
            for gamma, beta in zip(model.bn_gamma, model.bn_beta):
                gamma[:] = np.linspace(0.7, 1.3, len(gamma))
                beta[:] = np.linspace(-0.2, 0.2, len(beta))
            gradients = model.backprop(X, y)
            for parameter, gradient in zip(model.parametros, gradients):
                for index in np.ndindex(parameter.shape):
                    previous = parameter[index]
                    parameter[index] = previous+1e-6
                    upper = model.training_objective(X, y)
                    parameter[index] = previous-1e-6
                    lower = model.training_objective(X, y)
                    parameter[index] = previous
                    self.assertAlmostEqual(gradient[index], (upper-lower)/2e-6, places=7)
            self.assertEqual(model.bn_state['batches_tracked'], [0, 0])
            for mean in model.bn_state['running_mean']:
                np.testing.assert_array_equal(mean, np.zeros_like(mean))

    def test_inference_is_batch_independent_and_never_updates_statistics(self):
        X = np.random.default_rng(2).normal(size=(12, 3))
        y = np.eye(2)[np.arange(12) % 2]
        model = MLP([3, 4, 2], semilla=2, batch_norm={}, tamano_lote=4)
        model.entrenar(X, y, epocas=3, X_val=X, y_val=y, patience=None)
        state = model._snapshot()
        whole = model.predecir(X)
        separate = np.concatenate([model.predecir(row[None, :]) for row in X])
        np.testing.assert_allclose(whole, separate, atol=1e-14)
        combined = model.predecir(np.concatenate([X[:1], np.full((7, 3), 1e4)]))
        np.testing.assert_allclose(whole[:1], combined[:1], atol=1e-14)
        self.assertEqual(model.bn_state['batches_tracked'], [9])
        for key in ('running_mean', 'running_var'):
            for a, b in zip(state['bn_state'][key], model.bn_state[key]):
                np.testing.assert_array_equal(a, b)

    def test_exact_resume_best_model_and_singleton_tail(self):
        X = np.random.default_rng(4).uniform(size=(7, 784))
        y = np.eye(2)[np.arange(7) % 2]
        def model():
            return MLP([784, 4, 2], semilla=4, tamano_lote=3, batch_norm={},
                       augmentation={'name': 'translation', 'max_shift': 1},
                       optimizador=Momentum(0.015, 0.9))
        full, partial = model(), model()
        options = dict(X_val=X, y_val=y, patience=20)
        full.entrenar(X, y, epocas=5, **options)
        self.assertEqual(full.bn_state['batches_tracked'], [10])
        with TemporaryDirectory() as temp:
            checkpoint, best = Path(temp)/'checkpoint.npz', Path(temp)/'best.npz'
            partial.entrenar(X, y, epocas=5, pause_after=2, checkpoint_path=checkpoint,
                             best_model_path=best, **options)
            restored = MLP.cargar(checkpoint)
            restored.entrenar(X, y, epocas=3, best_model_path=best, **options)
            for a, b in zip(full.parametros, restored.parametros):
                np.testing.assert_array_equal(a, b)
            for key in ('running_mean', 'running_var'):
                for a, b in zip(full.bn_state[key], restored.bn_state[key]):
                    np.testing.assert_array_equal(a, b)
            self.assertEqual(full.historia.costo_validacion, restored.historia.costo_validacion)
            self.assertEqual(full.rng.bit_generator.state, restored.rng.bit_generator.state)
            chosen = MLP.cargar(best)
            self.assertAlmostEqual(chosen.costo(X, y), min(restored.historia.costo_validacion))
            np.testing.assert_allclose(chosen.predecir(X), np.concatenate([
                chosen.predecir(X[:3]), chosen.predecir(X[3:])]))

    def test_failed_epoch_restores_batch_norm_statistics_and_momentum(self):
        X = np.random.default_rng(1).normal(size=(6, 3))
        y = np.eye(2)[np.arange(6) % 2]
        model = MLP([3, 4, 2], batch_norm={}, tamano_lote=3,
                    optimizador=Momentum(0.015, 0.9))
        original = model._snapshot()
        gradient = model._gradientes
        calls = 0
        def interrupted(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = gradient(*args, **kwargs)
            if calls == 2:
                raise KeyboardInterrupt
            return result
        with patch.object(model, '_gradientes', side_effect=interrupted):
            model.entrenar(X, y, epocas=1, X_val=X, y_val=y)
        self.assertEqual(model.historia.stop_reason, 'interrupted')
        self.assertEqual(model.updates_completed, 0)
        self.assertEqual(model.bn_state['batches_tracked'], [0])
        for a, b in zip(model.parametros, original['parameters']):
            np.testing.assert_array_equal(a, b)
        for key in ('running_mean', 'running_var'):
            for a, b in zip(model.bn_state[key], original['bn_state'][key]):
                np.testing.assert_array_equal(a, b)

    def test_configuration_and_legacy_defaults(self):
        base = {'dataset': 'synthetic.csv', 'cache': 'synthetic.npz'}
        self.assertNotIn('batch_norm', validate_config(base))
        for settings in ({'epsilon': 0}, {'momentum': 0}, {'momentum': True}, {'other': 1}):
            with self.assertRaises(ValueError):
                validate_config({**base, 'batch_norm': settings})
        with self.assertRaises(ValueError):
            MLP([3, 2], batch_norm={})
        with self.assertRaises(ValueError):
            MLP([3, 4, 2], batch_norm={}, tamano_lote=1)


if __name__ == '__main__':
    unittest.main()
