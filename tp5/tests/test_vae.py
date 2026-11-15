import tempfile
from pathlib import Path
import unittest

import numpy as np

from src.vae import VAE


def _model(kl_weight=0.7, seed=3):
    return VAE(encoder=[6, 5, 4], latent=2, kl_weight=kl_weight, seed=seed, batch_size=4)


class VAETests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = rng.random((7, 6))
        self.eps = rng.standard_normal((7, 2))

    def test_analytic_gradients_match_finite_differences(self):
        model = _model()
        for parameter in model.parameters:  # leave the zero-bias symmetric point
            parameter += np.random.default_rng(1).normal(0, 0.3, parameter.shape)
        _, _, grads = model.gradients(self.x, self.eps)

        def objective():
            recon, kl = model.terms(self.x, self.eps)
            return recon + model.kl_weight * kl

        step = 1e-6
        for parameter, grad in zip(model.parameters, grads):
            numeric = np.zeros_like(parameter)
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + step
                up = objective()
                parameter[index] = original - step
                down = objective()
                parameter[index] = original
                numeric[index] = (up - down) / (2 * step)
            np.testing.assert_allclose(grad, numeric, rtol=1e-5, atol=1e-7)

    def test_kl_is_zero_at_the_prior_and_positive_elsewhere(self):
        zeros = np.zeros((3, 2))
        np.testing.assert_allclose(VAE._kl(zeros, zeros), 0.0)
        self.assertTrue((VAE._kl(zeros + 1.0, zeros) > 0).all())
        self.assertTrue((VAE._kl(zeros, zeros - 2.0) > 0).all())

    def test_shapes_determinism_and_decreasing_loss(self):
        first, second = _model(), _model()
        before = sum(np.multiply((1, first.kl_weight), first.terms(self.x)))
        for _ in range(200):
            first.train_epoch(self.x)
            second.train_epoch(self.x)
        after = sum(np.multiply((1, first.kl_weight), first.terms(self.x)))
        self.assertLess(after, before)
        mu, logvar = first.encode(self.x)
        self.assertEqual(mu.shape, (7, 2))
        self.assertEqual(logvar.shape, (7, 2))
        self.assertEqual(first.reconstruct(self.x).shape, (7, 6))
        np.testing.assert_array_equal(first.reconstruct(self.x), second.reconstruct(self.x))
        codes, images = first.sample(5, np.random.default_rng(9))
        self.assertEqual((codes.shape, images.shape), ((5, 2), (5, 6)))
        self.assertTrue(((images > 0) & (images < 1)).all())

    def test_saved_model_reproduces_outputs(self):
        model = _model()
        model.train_epoch(self.x)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.npz'
            model.save(path)
            restored = VAE.load(path)
        np.testing.assert_array_equal(restored.reconstruct(self.x), model.reconstruct(self.x))
        self.assertEqual(restored.configuration, model.configuration)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            VAE(encoder=[6], latent=2)
        with self.assertRaises(ValueError):
            VAE(encoder=[6, 4], latent=0)
        with self.assertRaises(ValueError):
            VAE(encoder=[6, 4], latent=2, kl_weight=-1)
        with self.assertRaises(ValueError):
            _model().encode(np.zeros((2, 5)))


if __name__ == '__main__':
    unittest.main()
