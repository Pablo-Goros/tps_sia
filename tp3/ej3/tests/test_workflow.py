"""Development workflow checks with synthetic data and training always blocked."""
import copy
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from tps_sia.tp3.shared import experiments as runner
from tps_sia.tp3.shared.metrics import classification_metrics
from tps_sia.tp3.shared.mlp import MLP
from tps_sia.tp3.shared.staged_search import summarize, _job
from tps_sia.tp3.ej3.src.analysis import analyze, render_group
from tps_sia.tp3.ej3.src.experiments import Search, STAGES
from tps_sia.tp3.ej3.src.factor_study import FactorStudy, PARTS, eligible_indices, nested_subsets
from tps_sia.tp3.ej3.src.protocol import CONFIG, Development, bind_output


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fake_rows(jobs, output, identity, root):
    """Write fabricated measurements, never invoke training or prediction."""
    rows = []
    for config, label in jobs:
        config = runner.validate_config(config)
        cid = runner.config_identity(config, path_root=root)
        run_id = f'{cid}-seed-{config["model_seed"]}'
        directory = output / 'runs' / run_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'best_model.npz').write_bytes(b'synthetic placeholder, not a model')
        metrics = classification_metrics(np.arange(10), np.arange(10))
        metrics.update(loss=0.01, cross_entropy=0.01)
        history = {'epocas_corridas': 2, 'epochs': [1, 2], 'validation_epochs': [1, 2],
                   'costo': [0.2, 0.01], 'costo_validacion': [0.3, 0.01],
                   'accuracy': [0.8, 1.0], 'accuracy_validacion': [0.7, 1.0]}
        report = {'run_id': run_id, 'config_id': cid, 'config': runner.recorded_config(config, root),
                  'status': 'completed', 'stop_reason': 'early_stopping', 'chosen_epoch': 2,
                  'history': history, 'metadata': identity, 'parameter_count': 100,
                  'duration_seconds': 1.0, 'metrics': {'train': metrics, 'validation': metrics},
                  'dataset': {'sha256': config['data']['dataset_sha256'],
                              'split_sha256': runner.fingerprint({k: config['data'][k]
                                 for k in ('train_indices', 'validation_indices')}),
                              'train_samples': len(config['data']['train_indices']),
                              'validation_samples': len(config['data']['validation_indices'])},
                  'artifacts': {'best_model': 'best_model.npz'},
                  'model_sha256': runner.sha256_file(directory / 'best_model.npz')}
        save(directory / 'results.json', report)
        rows.append(summarize(report, label))
    return rows


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.addCleanup(patch.stopall)
        patch.object(MLP, 'entrenar', side_effect=AssertionError('Training is prohibited in these checks')).start()
        (self.root / 'data').mkdir()
        self.X = np.zeros((30, 784), dtype=np.float32)
        labels = np.repeat([0, 5, 8], 10)
        for i in range(30):
            self.X[i, i] = 0.5
        self.y = np.eye(10, dtype=np.float32)[labels]
        for name in ('more_digits.csv', 'digits.csv'):
            with (self.root / 'data' / name).open('w', encoding='utf-8', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['image', 'label'])
                for image, label in zip(self.X, labels):
                    if name == 'digits.csv' and label == 8:
                        continue
                    writer.writerow([str(image.tolist()), int(label)])
        train = np.concatenate([np.arange(i, i + 7) for i in (0, 10, 20)])
        validation = np.concatenate([np.arange(i + 7, i + 10) for i in (0, 10, 20)])
        split = self.root / 'ej3/results/split_indices.npz'
        split.parent.mkdir(parents=True)
        np.savez(split, train=train, validation=validation, excluded=np.array([], dtype=np.int64))
        save(split.parent / 'split_manifest.json', {
            'dataset': {'path': 'data/more_digits.csv', 'rows': 30,
                        'sha256': runner.sha256_file(self.root / 'data/more_digits.csv')},
            'indices_file': 'split_indices.npz', 'train_size': 21, 'validation_size': 9,
            'split_sha256': runner.fingerprint({'train': train.tolist(), 'validation': validation.tolist()})})
        save(self.root / 'ej2/baseline.json', {'optimizer': 'sgd', 'learning_rate': 0.01,
                                             'architecture': [784, 4, 10], 'sanity_accuracy': 0.9})
        save(self.root / 'ej2/selection.json', {'config': {'architecture': [784, 4, 10],
                'dataset': '/another/machine/digits.csv', 'cache': '/another/machine/cache.npz',
                'optimizer': {'name': 'momentum', 'learning_rate': 0.1, 'momentum': 0.9}}})
        protocol = json.loads(CONFIG.read_text(encoding='utf-8'))
        protocol['reference_selection'] = 'ej2/selection.json'
        protocol['stages']['architectures']['architectures'] = [[784, 4, 10], [784, 8, 10]]
        self.config = self.root / 'ej3/configs/search.json'
        save(self.config, protocol)
        self.output = self.root / 'ej3/results/search'

    def search(self):
        return Search(self.config, self.output, root=self.root)

    def complete_search(self):
        search = self.search()
        with patch.object(Search, 'run_labeled', lambda s, jobs:
                          fake_rows(jobs, s.output, s.identity, s.development.root)):
            for stage in STAGES:
                search.run_stage(stage)
        return search

    def test_missing_reference_never_falls_back(self):
        (self.root / 'ej2/selection.json').unlink()
        with patch('tps_sia.tp3.ej3.src.protocol.cargar', side_effect=AssertionError('No loading')):
            with self.assertRaises(FileNotFoundError):
                self.search()

    def test_partition_fingerprint_and_sources_are_pinned(self):
        development = Development(self.config, root=self.root)
        self.assertEqual(len(development.train), 21)
        with (self.root / 'data/more_digits.csv').open('a') as file:
            file.write('\n')
        with self.assertRaises(ValueError):
            Development(self.config, root=self.root)

    def test_full_staged_workflow_without_training(self):
        search = self.complete_search()
        selected = json.loads((self.output / 'selection.json').read_text())
        self.assertEqual(selected['seeds'], [42, 0, 1])
        self.assertTrue(selected['validation_target_met'])
        with patch.object(Search, 'run_labeled', side_effect=AssertionError('No rerun')):
            search.run_stage('confirmation')
        with self.assertRaises(ValueError):
            bind_output(self.output, {'different': 'protocol'})

    def test_single_overrides_preserve_budget_and_partition(self):
        search = self.search()
        custom = self.root / 'custom.json'
        save(custom, {'epochs': 200, 'optimizer': {'learning_rate': 0.03}})
        value = search.run_single(custom, seed=7, dry_run=True)['config']
        self.assertEqual(value['epochs'], 200)
        self.assertEqual(value['model_seed'], 7)
        self.assertEqual(value['optimizer']['name'], 'momentum')
        self.assertEqual(len(value['data']['validation_indices']), 9)

    def test_factor_studies_and_saved_only_analysis(self):
        self.complete_search()
        factors = self.root / 'ej3/results/factors'
        study = FactorStudy(self.config, factors, self.output / 'selection.json', root=self.root)
        with patch('tps_sia.tp3.ej3.src.factor_study.execute_jobs', side_effect=
                   lambda jobs, output, identity, root, *args: fake_rows(jobs, output, identity, root)):
            for part in PARTS:
                study.run_part(part)
        dataset = json.loads((factors / 'dataset.json').read_text())
        self.assertEqual(dataset['data']['digits_eligible_samples'], 14)
        with patch.object(runner, 'cargar', side_effect=AssertionError('Analysis must not load data')):
            analyze(self.output, factors, self.root / 'analysis', plots=False)
        self.assertTrue((self.root / 'analysis/factors/coverage/paired_differences.csv').exists())
        self.assertTrue((self.root / 'analysis/search/stage_confirmation/per_class.csv').exists())

    def test_subsets_are_nested_and_keep_classes(self):
        labels = self.y.argmax(axis=1)
        subsets = nested_subsets(np.arange(30), labels, [0.25, 0.5, 1.0], 42)
        self.assertTrue(set(subsets[0.25]) < set(subsets[0.5]) < set(subsets[1.0]))
        self.assertEqual(set(labels[subsets[0.25]]), {0, 5, 8})
        self.assertEqual(len(eligible_indices(self.X, self.X[[0, 5]])), 28)

    def test_explicit_partition_rejects_image_leakage_and_test(self):
        development = Development(self.config, root=self.root)
        c = development.reference
        runner.explicit_partitions(c, self.X, self.y)
        duplicate = self.X.copy()
        duplicate[7] = duplicate[0]
        with self.assertRaises(ValueError):
            runner.explicit_partitions(c, duplicate, self.y)
        c = copy.deepcopy(c)
        c['data']['validation_dataset'] = str(self.root / 'data/digits_test.csv')
        with self.assertRaises(ValueError):
            runner.validate_config(c)

    def test_identity_includes_training_indices_and_validation_source(self):
        d = Development(self.config, root=self.root)
        smaller = d.config(d.reference, data={**d.data, 'train_indices': d.train[:5].tolist()})
        self.assertNotEqual(runner.config_identity(smaller, path_root=self.root),
                            runner.config_identity(d.reference, path_root=self.root))

    def test_reuse_and_resume_dispatch_without_training(self):
        from tps_sia.tp3.shared import staged_search
        search = self.search()
        config = search.development.reference
        rows = fake_rows([(config, 'reference')], search.output, search.identity, self.root)
        path = search.output / 'runs' / rows[0]['run_id'] / 'results.json'
        report = json.loads(path.read_text())
        with patch.object(staged_search, 'run', side_effect=AssertionError('Finished run must be reused')):
            self.assertEqual(_job(config, search.output, search.identity, self.root)['status'], 'completed')
        paused = copy.deepcopy(report)
        paused['status'] = 'interrupted'
        save(path, paused)
        with patch.object(staged_search, 'run', return_value=report) as run:
            _job(config, search.output, search.identity, self.root)
            self.assertTrue(run.call_args.kwargs['resume'])

    def test_cross_source_partition_excludes_common_validation(self):
        self.complete_search()
        study = FactorStudy(self.config, self.root / 'factors', self.output / 'selection.json', root=self.root)
        config = study.variants['dataset'][0][1]
        X, y = runner.cargar(config['dataset'], config['cache'])
        Xt, yt, Xv, yv, train, validation = runner.explicit_partitions(config, X, y)
        self.assertEqual(len(Xt), 14)
        self.assertEqual(len(Xv), 9)
        with (self.root / 'data/more_digits.csv').open('a') as file:
            file.write('\n')
        with self.assertRaises(ValueError):
            runner.data_hashes(config)

    def test_figures_use_saved_synthetic_measurements(self):
        search = self.search()
        with patch.object(Search, 'run_labeled', lambda s, jobs:
                          fake_rows(jobs, s.output, s.identity, s.development.root)):
            value = search.run_stage('controls')
        destination = self.root / 'figures'
        with patch.object(runner, 'cargar', side_effect=AssertionError('No source loading')):
            render_group(value, self.output, destination, plots=True)
        self.assertTrue((destination / 'accuracy.png').exists())
        self.assertTrue((destination / 'learning_curves.png').exists())
        self.assertTrue(list((destination / 'runs').glob('*/confusion_normalized.png')))


if __name__ == '__main__':
    unittest.main()
