import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.autoencoder import Autoencoder
from src.dataset import load_dataset, DEFAULT_DATASET
from src.explore import visualize_latent, candidate_distances, explore_candidates, record_review


class LatentTests(unittest.TestCase):
    def test_saved_coordinates_regenerate_interpolations_and_occupied_grid(self):
        dataset = load_dataset()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            model_path = folder / 'model.npz'
            model = Autoencoder(seed=19)
            model.save(model_path)
            config = {'pairs': [[96, 127]], 'steps': 3, 'grid_size': 3}
            result = explore_candidates(model_path, folder / 'candidates', config)
            self.assertEqual(len(result['candidates']), 12)
            coordinates = np.array([item['coordinates'] for item in result['candidates']])
            predictions = Autoencoder.load(model_path).decode(coordinates)
            np.testing.assert_allclose(predictions, [p['continuous'] for p in result['candidates']])
            codes = model.encode(dataset.vectors)
            np.testing.assert_allclose(coordinates[:3][[0, 2]], codes[[0, 31]])
            np.testing.assert_allclose(coordinates[1], (codes[0] + codes[31])/2)
            np.testing.assert_allclose(coordinates[3:].min(axis=0), codes.min(axis=0))
            np.testing.assert_allclose(coordinates[3:].max(axis=0), codes.max(axis=0))
            self.assertTrue(np.all(np.abs(coordinates) <= 1))
            self.assertEqual(result['review']['status'], 'pending')
            self.assertEqual(result['config'], config)
            self.assertEqual([p['min_hamming'] for p in result['candidates']], candidate_distances(predictions, dataset.vectors))
            self.assertTrue((folder / 'candidates' / 'interpolations.png').is_file())
            self.assertTrue((folder / 'candidates' / 'grid.png').is_file())
            self.assertEqual(json.loads((folder / 'candidates' / 'candidates.json').read_text()), result)
            self.assertEqual(Autoencoder.load(model_path).history['epocas_corridas'], 0)
            record_path = folder / 'candidates' / 'candidates.json'
            review = record_review(record_path, model_path, None, 'No hay letra adecuada en esta prueba')
            self.assertEqual(review['status'], 'no_suitable_letter')
            self.assertIsNone(review['selected_id'])
            selected = next(p for p in result['candidates'] if p['min_hamming'] > 0)
            moved_record = json.loads(record_path.read_text(encoding='utf-8'))
            moved_record['dataset']['path'] = str(folder / 'obsolete' / 'font.h')
            record_path.write_text(json.dumps(moved_record), encoding='utf-8')
            review = record_review(record_path, model_path, selected['id'],
                                   'Revisión sintética del flujo, no evidencia académica',
                                   dataset_path=DEFAULT_DATASET)
            self.assertEqual(review['coordinates'], selected['coordinates'])
            self.assertEqual(review['model_sha256'], result['model']['sha256'])
            self.assertEqual(review['selected_id'], selected['id'])
            recorded = record_path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'already exists'):
                explore_candidates(model_path, folder / 'candidates', config)
            self.assertEqual(record_path.read_bytes(), recorded)
            other_model = folder / 'other.npz'
            Autoencoder(seed=20).save(other_model)
            with self.assertRaisesRegex(ValueError, 'model'):
                record_review(record_path, other_model, selected['id'], 'No corresponde al modelo')
            invalid_dataset = json.loads(record_path.read_text(encoding='utf-8'))
            invalid_dataset['dataset']['sha256'] = '0' * 64
            record_path.write_text(json.dumps(invalid_dataset), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'dataset'):
                record_review(record_path, model_path, None, 'Dataset diferente', DEFAULT_DATASET)

    def test_candidate_novelty_uses_all_original_patterns_and_inclusive_threshold(self):
        dataset = load_dataset()
        copy = dataset.vectors[17].copy()
        copy[copy == 1] = 0.5
        novel = np.zeros(35)
        distances = candidate_distances(np.array([copy, novel]), dataset.vectors)
        self.assertEqual(distances[0], 0)
        self.assertEqual(distances[1], 3)

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
