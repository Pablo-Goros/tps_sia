import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.autoencoder import Autoencoder
from src.dataset import load_dataset
from src.explore import visualize_latent


class LatentTests(unittest.TestCase):
    def test_saved_model_yields_reproducible_labeled_coordinates_without_training(self):
        dataset = load_dataset()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            model_path = folder / 'model.npz'
            model = Autoencoder(seed=19)
            model.save(model_path)
            before = model_path.read_bytes()
            result = visualize_latent(model_path, folder / 'latent')
            self.assertEqual(len(result['points']), 32)
            self.assertEqual([p['label'] for p in result['points']], dataset.labels)
            self.assertEqual([p['code'] for p in result['points']], list(range(96, 128)))
            self.assertEqual(result['points'][-1]['label'], '0x7f DEL')
            coordinates = np.array([p['coordinates'] for p in result['points']])
            self.assertEqual(coordinates.shape, (32, 2))
            np.testing.assert_allclose(coordinates, model.encode(dataset.vectors), atol=1e-14)
            self.assertEqual(result['model']['sha256'], hashlib.sha256(before).hexdigest())
            self.assertEqual(result['dataset']['sha256'], dataset.sha256)
            self.assertEqual(model_path.read_bytes(), before)
            self.assertTrue((folder / 'latent' / 'latent.png').is_file())
            self.assertEqual(json.loads((folder / 'latent' / 'latent.json').read_text()), result)
            self.assertEqual(result, visualize_latent(model_path, folder / 'repeated'))
            self.assertEqual(Autoencoder.load(model_path).history['epocas_corridas'], 0)
