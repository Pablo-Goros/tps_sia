"""End-to-end development artifacts and exact runner resumption on synthetic data."""
import copy
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from tps_sia.tp3.shared import experiments
from tps_sia.tp3.shared.analysis import comparison_rows, generate_comparison, seed_summary
from tps_sia.tp3.shared.digit_dataset import cargar, particionar
from tps_sia.tp3.shared.metrics import evaluate_model
from tps_sia.tp3.shared.mlp import MLP


def synthetic_csv(path):
    with Path(path).open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['image', 'label'])
        for digit in (0, 5, 8):
            for sample in range(10):
                image = [0.0] * 784
                image[digit] = 0.2 + sample * 0.01
                writer.writerow([str(image), digit])


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        synthetic_csv(self.base / 'digits.csv')
        self.config = {'dataset': str(self.base / 'digits.csv'), 'cache': str(self.base / 'cache.npz'),
                       'architecture': [784, 4, 10], 'epochs': 3, 'batch_size': 8,
                       'weight_logging': {'enabled': True, 'every_updates': 1,
                                          'selected_weights': [[0, 0, 0]]}}

    def test_artifacts_reload_and_analysis(self):
        r = experiments.run(self.config, self.base / 'runs')
        directory = Path(r['run_directory'])
        for name in ('manifest.json', 'results.json', 'history.csv', 'split.npz', 'best_model.npz', 'checkpoint.npz', 'weights.jsonl'):
            self.assertTrue((directory / name).is_file(), name)
        X, y = cargar(self.config['dataset'], self.config['cache'])
        _, _, Xv, yv, train, val = particionar(X, y, return_indices=True)
        model = MLP.cargar(directory / 'best_model.npz')
        self.assertEqual(evaluate_model(model, Xv, yv), r['metrics']['validation'])
        with np.load(directory / 'split.npz') as saved:
            np.testing.assert_array_equal(saved['train'], train)
            np.testing.assert_array_equal(saved['validation'], val)
        self.assertEqual(r['chosen_epoch'], model.epochs_completed)
        self.assertEqual(sum(r['dataset']['validation_class_counts']), len(yv))
        with (directory / 'history.csv').open() as file:
            self.assertEqual(len(list(csv.DictReader(file))), 3)
        with patch.object(experiments, 'cargar', side_effect=AssertionError('Analysis must not load data')):
            rows = generate_comparison([directory / 'results.json'], self.base / 'analysis')
        self.assertEqual(len(rows), 1)
        self.assertEqual(seed_summary(rows)[0]['accuracy_std'], 0)
        with self.assertRaises(FileExistsError):
            experiments.run(self.config, self.base / 'runs')
        with self.assertRaises(FileExistsError):
            experiments.run(self.config, self.base / 'runs', resume=True)

    def test_resume_exact_for_all_optimizers(self):
        for optimizer in ('sgd', 'momentum', 'adam'):
            with self.subTest(optimizer=optimizer):
                config = {**self.config, 'optimizer': {'name': optimizer, 'learning_rate': 0.001}}
                full = experiments.run(config, self.base / f'{optimizer}-full')
                paused = experiments.run(config, self.base / f'{optimizer}-paused', pause_after=1)
                self.assertEqual(paused['status'], 'interrupted')
                self.assertIsNone(paused['metrics'])
                resumed = experiments.run(config, self.base / f'{optimizer}-paused', resume=True)
                self.assertEqual(resumed['status'], 'completed')
                a = MLP.cargar(Path(full['run_directory']) / 'checkpoint.npz')
                b = MLP.cargar(Path(resumed['run_directory']) / 'checkpoint.npz')
                for p, q in zip(a.parametros, b.parametros):
                    np.testing.assert_array_equal(p, q)
                self.assertEqual(full['metrics'], resumed['metrics'])
                for key in ('costo', 'accuracy', 'learning_rate', 'epochs', 'updates'):
                    self.assertEqual(full['history'][key], resumed['history'][key])
                full_records = (Path(full['run_directory']) / 'weights.jsonl').read_text()
                resumed_records = (Path(resumed['run_directory']) / 'weights.jsonl').read_text()
                self.assertEqual(full_records, resumed_records)

    def test_source_changes_block_resume(self):
        experiments.run(self.config, self.base / 'runs', pause_after=1)
        with (self.base / 'digits.csv').open('a') as file:
            file.write('\n')
        with self.assertRaises(ValueError):
            experiments.run(self.config, self.base / 'runs', resume=True)

    def test_test_dataset_rejected_before_loading(self):
        for name in ('digits_test.csv', 'DIGITS_TEST.CSV'):
            with patch.object(experiments, 'cargar') as loader, self.assertRaises(ValueError):
                experiments.run({**self.config, 'dataset': str(self.base / name)}, self.base / 'runs')
            loader.assert_not_called()
        test = self.base / 'digits_test.csv'
        synthetic_csv(test)
        alias = self.base / 'alias.csv'
        alias.symlink_to(test)
        with self.assertRaises(ValueError):
            experiments.run({**self.config, 'dataset': str(alias)}, self.base / 'runs')

    def test_invalid_configs(self):
        variants = [{'epochs': True}, {'model_seed': -1}, {'shuffle': 1}, {'batch_size': 0},
                    {'strategy': 'batch'}, {'strategy': 'online'}, {'activation': 'step'},
                    {'architecture': [2, 10]}, {'output': 'logistica'}, {'loss': 'mse'},
                    {'optimizer': {'name': 'adam', 'learning_rate': float('nan')}},
                    {'stopping': {'patience': 0}}, {'stopping': {'monitor': 'test_loss'}},
                    {'weight_logging': {'enabled': 1}}, {'weight_logging': {'selected_weights': [[9, 0, 0]]}},
                    {'preprocessing': {'name': 'unknown'}}, {'validation_fraction': 1}, {'unknown': True}]
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                experiments.validate_config({**self.config, **variant})
        for strategy, size in (('online', 1), ('batch', None), ('mini_batch', 8)):
            experiments.validate_config({**self.config, 'strategy': strategy, 'batch_size': size})

    def test_numerical_failure_excluded(self):
        with patch.object(MLP, '_gradientes', side_effect=FloatingPointError('Injected numerical failure')):
            r = experiments.run(self.config, self.base / 'failed')
        self.assertEqual(r['status'], 'failed')
        self.assertEqual(r['stop_reason'], 'numerical_failure')
        self.assertIsNone(r['metrics'])
        self.assertEqual(comparison_rows([r]), [])
        json.dumps(r, allow_nan=False)

    def test_train_only_standardization(self):
        c = {**self.config, 'preprocessing': {'name': 'standardize'}}
        r = experiments.run(c, self.base / 'standardize')
        X, y = cargar(c['dataset'], c['cache'])
        Xt, _, Xv, yv = particionar(X, y)
        np.testing.assert_array_equal(r['preprocessing']['mean'], Xt.mean(axis=0, dtype=float))
        model = MLP.cargar(Path(r['run_directory']) / 'best_model.npz')
        self.assertEqual(evaluate_model(model, experiments.preprocess(Xv, model.preprocessing), yv), r['metrics']['validation'])

    def test_new_run_from_weights_resets_history(self):
        r = experiments.run(self.config, self.base / 'first')
        source = Path(r['run_directory']) / 'best_model.npz'
        c = {**self.config, 'epochs': 1}
        new = experiments.run(c, self.base / 'second', initial_model=source)
        self.assertEqual(new['initialization']['mode'], 'existing_weights')
        self.assertEqual(new['history']['epochs'], [1])
        self.assertNotEqual(new['config_id'], r['config_id'])
        self.assertEqual(new['history']['updates'], [3])

    def test_resume_from_existing_weights(self):
        r = experiments.run(self.config, self.base / 'source')
        source = Path(r['run_directory']) / 'best_model.npz'
        experiments.run(self.config, self.base / 'warm', initial_model=source, pause_after=1)
        result = experiments.run(self.config, self.base / 'warm', initial_model=source, resume=True)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['history']['epochs'], [1, 2, 3])

    def test_partition_indices_preserve_default(self):
        X, y = cargar(self.config['dataset'], self.config['cache'])
        default = particionar(X, y)
        indexed = particionar(X, y, return_indices=True)
        for original, part in zip(default, indexed[:4]):
            np.testing.assert_array_equal(original, part)
        train, val = indexed[4:]
        self.assertFalse(np.intersect1d(train, val).size)
        np.testing.assert_array_equal(np.sort(np.r_[train, val]), np.arange(len(X)))
        self.assertEqual(len(particionar(X, y, validation_fraction=0.5)[2]), 15)

    def test_figures_from_saved_results(self):
        try:
            import matplotlib
        except ImportError:
            self.skipTest('Matplotlib required for figures')
        from tps_sia.tp3.shared.analysis import generate
        r = experiments.run(self.config, self.base / 'figures')
        path = Path(r['run_directory']) / 'results.json'
        with patch.object(experiments, 'cargar', side_effect=AssertionError('Do not load data')):
            generate(path, self.base / 'plots')
        for filename in ('learning_curves.png', 'confusion_matrix.png', 'confusion_normalized.png',
                         'weight_norms.png', 'selected_weights.png'):
            self.assertGreater((self.base / 'plots' / filename).stat().st_size, 1000)

    def test_summary_spread_and_no_duplicate_seeds(self):
        r = experiments.run(self.config, self.base / 'first')
        other = copy.deepcopy(r)
        other['config']['model_seed'] = 0
        other['run_id'] = 'other'
        other['metrics']['validation']['accuracy'] = 0.9
        rows = comparison_rows([r, other])
        summary = seed_summary(rows)[0]
        self.assertEqual(summary['runs'], 2)
        self.assertAlmostEqual(summary['accuracy_std'], abs(rows[0]['accuracy'] - rows[1]['accuracy']) / 2)
        with self.assertRaises(ValueError):
            seed_summary(rows + [rows[0]])


if __name__ == '__main__':
    unittest.main()
