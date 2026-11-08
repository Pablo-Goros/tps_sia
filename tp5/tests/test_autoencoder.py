import tempfile
from pathlib import Path
import unittest
import numpy as np
from src.dataset import load_dataset
from src.autoencoder import Autoencoder


class AutoencoderTests(unittest.TestCase):
    def test_public_model_roundtrip_and_continued_training(self):
        x = load_dataset().vectors
        model = Autoencoder(seed=42)
        model.train(x, 2)
        self.assertEqual(model.encode(x).shape, (32, 2))
        self.assertEqual(model.reconstruct(x).shape, (32, 35))
        np.testing.assert_allclose(model.decode(model.encode(x)), model.reconstruct(x))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.npz'
            model.save(path)
            restored = Autoencoder.load(path)
            np.testing.assert_allclose(restored.encode(x), model.encode(x))
            np.testing.assert_allclose(restored.reconstruct(x), model.reconstruct(x))
            restored.train(x, 2)
            uninterrupted = Autoencoder(seed=42)
            uninterrupted.train(x, 4)
            np.testing.assert_allclose(restored.reconstruct(x), uninterrupted.reconstruct(x))
            self.assertTrue(np.isfinite(restored.reconstruct(x)).all())
