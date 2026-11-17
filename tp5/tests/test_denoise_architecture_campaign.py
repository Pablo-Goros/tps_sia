"""Validate the authorized grid and reject incompatible/corrupted reuse."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest

from src.dataset import load_dataset
from src.denoise_architecture_campaign import variants, verify_reuse
from src.denoise_campaign import read_json
from src.denoise_training import train_run
from src.noise import make_collection


class ArchitectureCampaignTests(unittest.TestCase):
    def test_grid_keeps_the_agreed_depths_and_all_four_code_widths(self):
        root = Path(__file__).resolve().parents[1]
        config = read_json(root/'configs/denoising-architectures.json')
        original = read_json(root/'results/denoising/manifest.json')['variants']['winner']
        resolved = variants(config, original)
        self.assertEqual(len(resolved), 12)
        self.assertEqual(resolved['current-d10']['architecture'],
                         [35, 40, 20, 10, 10, 10, 20, 40, 35])
        self.assertEqual(resolved['simple-d20']['architecture'], [35, 40, 20, 20, 20, 40, 35])
        self.assertEqual(resolved['deeper-d5']['architecture'],
                         [35, 40, 30, 20, 10, 5, 10, 20, 30, 40, 35])
        for model in resolved.values():
            self.assertEqual({k: v for k, v in model.items() if k != 'architecture'},
                             {k: v for k, v in original.items() if k != 'architecture'})
        modified = copy.deepcopy(config)
        modified['max_epochs'] = 3001
        with self.assertRaises(ValueError):
            variants(modified, original)

    def test_reuse_recalculates_validation_and_rejects_protocol_changes_or_corruption(self):
        clean = load_dataset().vectors
        validation = make_collection(clean, [0, .05, .1, .2], 2, 123)
        config = {'variant': 'old', 'seed': 42, 'max_epochs': 2, 'evaluate_every': 1,
                  'model': {'architecture': [35, 8, 5, 8, 35], 'loss': 'bce'},
                  'training_levels': [0, .05, .1, .2], 'selection_levels': [.05, .1, .2]}
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)/'run'
            with contextlib.redirect_stdout(io.StringIO()):
                row = train_run(config, clean, validation, directory)
            desired = {**config, 'variant': 'renamed'}
            checked = verify_reuse(directory, desired, row, validation)
            self.assertEqual(checked['validation_recomputed'], row['best']['score'])
            wrong = {**desired, 'training_levels': [0, .1]}
            with self.assertRaisesRegex(ValueError, 'protocol'):
                verify_reuse(directory, wrong, row, validation)
            path = directory/'best_model.npz'
            path.write_bytes(path.read_bytes()+b'corrupt')
            with self.assertRaisesRegex(ValueError, 'hash'):
                verify_reuse(directory, desired, row, validation)
