import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from src.autoencoder import Autoencoder
from src.continue_training import continue_run
from src.dataset import load_dataset
from src.train import run_training


class ContinuationTests(unittest.TestCase):
    def test_reload_and_rate_change_preserve_trajectory_and_source(self):
        data = load_dataset()
        for rate in (0.002, 0.001):
            with self.subTest(rate=rate), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                config = {'architecture': [35, 20, 2, 20, 35], 'seed': 42,
                          'optimizer': {'name': 'rmsprop', 'learning_rate': 0.002},
                          'batch_size': 32, 'max_epochs': 2, 'evaluate_every': 2,
                          'stop_on_tolerance': True}
                (root/'config.json').write_text(json.dumps(config))
                run_training(root/'config.json', root/'source')
                before = {p.name: p.read_bytes() for p in (root/'source').iterdir()}
                # Reference continues in memory, with a single train call per phase.
                reference = Autoencoder(architecture=config['architecture'], seed=42,
                                        optimizer=config['optimizer'])
                reference.train(data.vectors, 2)
                reference.set_learning_rate(rate)
                reference.train(data.vectors, 2)
                result = continue_run(root/'source', root/'continued', rate, 2, 2,
                                      time.perf_counter()+60, data)
                actual = Autoencoder.load(root/'continued/model.npz')
                np.testing.assert_array_equal(actual.reconstruct(data.vectors), reference.reconstruct(data.vectors))
                self.assertEqual(actual.history['updates'], reference.history['updates'])
                self.assertEqual(actual.history['learning_rate'], [0.002, 0.002, rate, rate])
                self.assertEqual(result['manifest']['additional_epochs_completed'], 2)
                self.assertEqual(before, {p.name: p.read_bytes() for p in (root/'source').iterdir()})
                with self.assertRaises(ValueError):
                    continue_run(root/'source', root/'continued', rate, 2, 2,
                                 time.perf_counter()+60, data)

    def test_expired_budget_performs_no_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = {'architecture': [35, 20, 2, 20, 35], 'seed': 42,
                      'optimizer': {'name': 'rmsprop', 'learning_rate': 0.002},
                      'batch_size': 32, 'max_epochs': 1, 'evaluate_every': 1,
                      'stop_on_tolerance': True}
            (root/'config.json').write_text(json.dumps(config))
            run_training(root/'config.json', root/'source')
            result = continue_run(root/'source', root/'continued', 0.001, 2, 1,
                                  time.perf_counter()-1, load_dataset())
            self.assertEqual(result['manifest']['additional_epochs_completed'], 0)
            self.assertEqual(result['manifest']['stop_reason'], 'time_budget')
