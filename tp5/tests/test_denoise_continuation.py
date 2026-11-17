"""A saved denoising continuation must match uninterrupted training exactly."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.autoencoder import Autoencoder
from src.dataset import load_dataset
from src.denoise_continuation import continue_run, read_archive_config
from src.denoise_training import train_run
from src.noise import make_collection


class ContinuationTests(unittest.TestCase):
    def test_absolute_noise_epochs_and_adam_state_match_uninterrupted_training(self):
        clean = load_dataset().vectors
        validation = make_collection(clean, [0, .05, .1, .2], 2, 123)
        config = {'variant': 'smoke', 'seed': 42, 'max_epochs': 2, 'evaluate_every': 1,
                  'model': {'architecture': [35, 8, 5, 8, 35], 'loss': 'bce',
                            'optimizer': {'name': 'adam', 'learning_rate': .002}},
                  'training_levels': [0, .05, .1, .2], 'selection_levels': [.05, .1, .2]}
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            train_run(config, clean, validation, root/'initial')
            resumed = continue_run(root/'initial', root/'continued', clean, validation, 6)
            full = train_run({**config, 'max_epochs': 6}, clean, validation, root/'full')
            self.assertEqual(resumed['epochs_completed'], 6)
            self.assertEqual(resumed['additional_epochs_completed'], 4)
            self.assertEqual(resumed['best']['score'], full['best']['score'])
            self.assertEqual(resumed['best']['epoch'], full['best']['epoch'])
            np.testing.assert_array_equal(
                Autoencoder.load(root/'continued/last_model.npz').reconstruct(clean),
                Autoencoder.load(root/'full/last_model.npz').reconstruct(clean))
            with np.load(root/'continued/last_model.npz', allow_pickle=False) as a, \
                    np.load(root/'full/last_model.npz', allow_pickle=False) as b:
                for key in a.files:
                    if key.startswith(('w_', 'b_', 'optimizer_')):
                        np.testing.assert_array_equal(a[key], b[key])
                self.assertEqual(read_archive_config(a)['rng'], read_archive_config(b)['rng'])
                self.assertEqual(read_archive_config(a)['optimizer_state'], read_archive_config(b)['optimizer_state'])
            with self.assertRaisesRegex(ValueError, 'new'):
                continue_run(root/'initial', root/'continued', clean, validation, 6)
            with self.assertRaisesRegex(ValueError, 'Target'):
                continue_run(root/'initial', root/'invalid', clean, validation, 2)
