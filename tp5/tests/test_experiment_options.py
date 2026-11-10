"""Numerical contracts for loss, code activation and cosine continuation."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from src.autoencoder import Autoencoder
from src.dataset import load_dataset
from tp3.shared.mlp import MLP


class ExperimentOptionsTests(unittest.TestCase):
    def test_gradients_for_both_losses_and_hidden_activations(self):
        x = np.array([[.2, .8, .1], [.7, .3, .9]])
        y = np.array([[0., 1.], [1., 0.]])
        for loss in ('half_squared_error', 'bce'):
            for linear in (None, 0):
                with self.subTest(loss=loss, linear=linear):
                    model = MLP([3, 2, 2], salida='logistica', beta=.7,
                                loss=loss, linear_hidden_layer=linear, semilla=17)
                    gradients = model.backprop(x, y)
                    for parameter, gradient in zip(model.parametros, gradients):
                        for index in np.ndindex(parameter.shape):
                            original = parameter[index]
                            parameter[index] = original + 1e-6
                            plus = model.costo(x, y)
                            parameter[index] = original - 1e-6
                            minus = model.costo(x, y)
                            parameter[index] = original
                            self.assertAlmostEqual(gradient[index], (plus-minus)/2e-6, places=7)

    def test_bce_corrects_confidently_wrong_saturated_outputs(self):
        model = MLP([1, 2], salida='logistica', loss='bce')
        model.pesos[0][:] = 0
        model.biases[0][:] = [-1000., 1000.]
        x, y = np.ones((1, 1)), np.array([[1., 0.]])
        self.assertEqual(model.costo(x, y), 4000.)
        np.testing.assert_array_equal(model.backprop(x, y)[-1], [-2., 2.])

    def test_cosine_and_linear_bce_survive_save_and_resume(self):
        x = load_dataset().vectors
        options = dict(architecture=[35, 3, 2, 3, 35], seed=17, loss='bce',
                       latent_activation='linear', optimizer={'name': 'adam', 'learning_rate': .002},
                       lr_scheduler={'name': 'cosine', 'total_epochs': 8, 'min_lr': 0.})
        uninterrupted = Autoencoder(**options)
        uninterrupted.train(x, 8)
        chunked = Autoencoder(**options)
        chunked.train(x, 3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.npz'
            chunked.save(path)
            restored = Autoencoder.load(path)
            restored.train(x, 5)
            self.assertEqual(restored.configuration, uninterrupted.configuration)
            self.assertEqual(restored.history['learning_rate'], uninterrupted.history['learning_rate'])
            np.testing.assert_array_equal(restored.reconstruct(x), uninterrupted.reconstruct(x))
            np.testing.assert_array_equal(restored.decode(restored.encode(x)), restored.reconstruct(x))
            expected = .002*(1+np.cos(np.pi*np.arange(8)/8))/2
            np.testing.assert_allclose(restored.history['learning_rate'], expected, atol=1e-18)

    def test_options_preserve_initial_weights(self):
        baseline = Autoencoder(seed=42)
        for loss in ('half_squared_error', 'bce'):
            for latent in ('tanh', 'linear'):
                alternative = Autoencoder(seed=42, loss=loss, latent_activation=latent)
                for before, after in zip(baseline._mlp.parametros, alternative._mlp.parametros):
                    np.testing.assert_array_equal(before, after)
