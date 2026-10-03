"""Protocol v2: python -m tps_sia.tp3.ej2.tests.test_staged_search.

Selection rules use synthetic results. The smoke test trains real but tiny runs
(2 epochs, 256 train samples of digits.csv) into a temporary directory; it is
a technical check, not an experiment. digits_test.csv is never opened.
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tps_sia.tp3.ej2.src.staged_search import (
    CONFIG, TP3, StagedSearch, _winner_text, best_neighbor, border_extension, choose_activation,
    group_summary, near_tie_pair, protocol_sha256, rank, resolve_near_tie, sequence_neighbor)
from tps_sia.tp3.shared import experiments
from tps_sia.tp3.shared.experiments import config_identity, recorded_config, validate_config
from tps_sia.tp3.shared.tests.test_experiments import alias_to, synthetic_csv


def row(config_id, seed=42, accuracy=0.9, loss=0.3, parameters=100, status='completed', **extra):
    return {'label': config_id, 'run_id': f'{config_id}-seed-{seed}', 'config_id': config_id, 'seed': seed,
            'status': status, 'stop_reason': 'early_stopping', 'converged': True,
            'accuracy': accuracy if status == 'completed' else None,
            'loss': loss if status == 'completed' else None, 'chosen_epoch': 7, 'epochs_run': 17,
            'parameter_count': parameters, 'duration_seconds': 1.0, 'dataset_sha256': 'd',
            'split_sha256': 's', 'config': {}, **extra}


def fake_runner(score):
    """Replace run_labeled: score(config) -> accuracy or None (numerical failure)."""
    calls = []

    def run_labeled(self, jobs):
        rows = []
        for config, label in jobs:
            c = validate_config(config)
            calls.append((label, c['model_seed']))
            accuracy = score(c)
            architecture = c['architecture']
            parameters = sum(a * b + b for a, b in zip(architecture, architecture[1:]))
            portable = {k: v for k, v in recorded_config(c, self.root).items() if k != 'model_seed'}
            rows.append(row(config_identity(c, path_root=self.root), c['model_seed'], accuracy,
                            loss=1 - (accuracy or 0), parameters=parameters,
                            status='completed' if accuracy is not None else 'failed',
                            label=label, activation=c['activation'], architecture=architecture,
                            batch_size=c['batch_size'], optimizer=c['optimizer']['name'],
                            learning_rate=c['optimizer']['learning_rate'], config=portable))
            # Minimal stored report, read only by stage 7 to write selection.json.
            directory = self.output / 'runs' / rows[-1]['run_id']
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'results.json').write_text(json.dumps({
                'dataset': {'source': 'data/digits.csv'}, 'artifacts': {'best_model': 'best_model.npz'},
                'model_sha256': 'synthetic', 'chosen_epoch': 7}), encoding='utf-8')
        return rows
    return run_labeled, calls


class RuleTests(unittest.TestCase):
    def test_sequence_and_border_rule(self):
        self.assertEqual(sequence_neighbor(0.1, +1), 0.3)
        self.assertEqual(sequence_neighbor(0.05, +1), 0.1)
        self.assertEqual(sequence_neighbor(0.0001, -1), 3e-05)
        self.assertEqual(sequence_neighbor(0.0003, -1), 0.0001)
        rates = [0.001, 0.01, 0.05, 0.1]
        rows = [row(str(r), accuracy=0.8 + r, learning_rate=r) for r in rates]
        self.assertEqual(border_extension(rows, rates), ('up', 0.3))
        rows.append(row('0.3', accuracy=0.5, learning_rate=0.3))  # Reevaluated: now inside.
        self.assertIsNone(border_extension(rows, rates + [0.3]))
        rows[-1] = row('0.3', status='failed', learning_rate=0.3)  # A diverged extension also closes it.
        self.assertIsNone(border_extension(rows, rates + [0.3]))
        low = [row(str(r), accuracy=0.9 - r, learning_rate=r) for r in rates]
        self.assertEqual(border_extension(low, rates), ('down', 0.0003))
        self.assertEqual(best_neighbor(rows, rates + [0.3], 0.1), 0.05)

    def test_activation_tie_prefers_tanh(self):
        rows = [row('t1', accuracy=0.949, activation='tanh'), row('t2', accuracy=0.90, activation='tanh'),
                row('r1', accuracy=0.951, activation='relu'), row('r2', accuracy=0.93, activation='relu')]
        winner, decision = choose_activation(rows, 0.003, 'tanh')
        self.assertEqual(winner['config_id'], 't1')
        self.assertTrue(decision['tie'])
        rows[2]['accuracy'] = 0.96
        winner, decision = choose_activation(rows, 0.003, 'tanh')
        self.assertEqual(winner['config_id'], 'r1')
        self.assertFalse(decision['tie'])

    def test_near_tie_detection_and_improvement_rule(self):
        ranked = rank([row('a', accuracy=0.950), row('b', accuracy=0.948), row('c', accuracy=0.90)])
        self.assertEqual([r['config_id'] for r in near_tie_pair(ranked, 0.003)], ['a', 'b'])
        self.assertIsNone(near_tie_pair(rank([row('a', accuracy=0.95), row('b', accuracy=0.94)]), 0.003))
        noisy = [row('a', s, acc) for s, acc in ((42, 0.950), (0, 0.940), (1, 0.960))]
        noisy += [row('b', s, acc) for s, acc in ((42, 0.948), (0, 0.946), (1, 0.950))]
        summary = group_summary(noisy, [42, 0, 1])
        self.assertEqual(summary[0]['config_id'], 'a')
        winner, evidence = resolve_near_tie(summary, 'b', 2.0)
        self.assertEqual(winner, 'b')  # Not distinguishable: the incumbent stays.
        self.assertFalse(evidence['comparison']['significant'])
        self.assertAlmostEqual(evidence['comparison']['sigma'], ((0.01 ** 2 + 0.002 ** 2) / 2) ** 0.5)
        clear = [row('a', s, acc) for s, acc in ((42, 0.950), (0, 0.951), (1, 0.952))]
        clear += [row('b', s, acc) for s, acc in ((42, 0.948), (0, 0.940), (1, 0.941))]
        winner, evidence = resolve_near_tie(group_summary(clear, [42, 0, 1]), 'b', 2.0)
        self.assertEqual(winner, 'a')
        self.assertTrue(evidence['comparison']['significant'])

    def test_stage_7_tiebreak_order(self):
        seeds = [42, 0, 1, 2, 3]
        rows = []
        for config_id, accuracy, loss, parameters in (('z', 0.95, 0.2, 100), ('y', 0.95, 0.2, 100),
                                                      ('x', 0.95, 0.2, 200), ('w', 0.95, 0.1, 300),
                                                      ('v', 0.96, 0.9, 900)):
            rows += [row(config_id, seed, accuracy, loss, parameters) for seed in seeds]
        self.assertEqual([s['config_id'] for s in group_summary(rows, seeds)], ['v', 'w', 'y', 'z', 'x'])

    def test_incomplete_groups_rejected(self):
        seeds = [42, 0, 1]
        complete = [row('a', s) for s in seeds]
        for invalid in (complete[:2], complete + [row('a', 0)], complete + [row('b', 42)]):
            with self.assertRaises(ValueError):
                group_summary(invalid, seeds)
        mixed = complete + [row('b', 42, 0.99), row('b', 0, status='failed'), row('b', 1, 0.99)]
        summary = group_summary(mixed, seeds)
        self.assertEqual([s['status'] for s in summary], ['completed', 'failed'])
        other_split = [row('c', s) for s in seeds]
        other_split[1]['split_sha256'] = 'other'
        with self.assertRaises(ValueError):
            group_summary(other_split, seeds)


class StageFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / 'v2'

    def score(self, c):
        """Synthetic landscape with known optimum per optimizer."""
        optimizer, rate = c['optimizer']['name'], c['optimizer']['learning_rate']
        if optimizer == 'momentum' and rate >= 0.1:
            return None  # Diverges: closes momentum's upper border.
        peak = {'sgd': 0.3, 'momentum': 0.05, 'rmsprop': 3e-05, 'adam': 1e9}[optimizer]
        import math
        distance = abs(math.log10(rate) - math.log10(peak))
        base = {'sgd': 0.93, 'momentum': 0.95, 'rmsprop': 0.94, 'adam': 0.90}[optimizer]
        bonus = 0.002 if c['activation'] == 'relu' else 0
        width = c['architecture'][1] / 100000
        return base + bonus + width - 0.01 * distance - 0.001 * (c['batch_size'] != 32)

    def test_border_extension_rounds_and_cap(self):
        run_labeled, calls = fake_runner(self.score)
        with patch.object(StagedSearch, 'run_labeled', run_labeled):
            search = StagedSearch(CONFIG, self.output)
            stage_0 = search.run_stage(0)
            self.assertEqual(stage_0['winner']['activation'], 'tanh')  # 0.2 pp < 0.3 pp
            stage_1 = search.run_stage(1)
        extensions = [(x['optimizer'], x['direction'], x['added_learning_rate']) for x in stage_1['extensions']]
        self.assertIn(('sgd', 'up', 0.3), extensions)
        self.assertIn(('sgd', 'up', 1.0), extensions)
        self.assertIn(('momentum', 'up', 0.1), extensions)
        self.assertIn(('rmsprop', 'down', 3e-05), extensions)
        self.assertIn(('rmsprop', 'down', 1e-05), extensions)
        self.assertEqual(stage_1['best']['sgd']['learning_rate'], 0.3)
        self.assertEqual(stage_1['best']['momentum']['learning_rate'], 0.05)
        self.assertEqual(stage_1['best']['rmsprop']['learning_rate'], 3e-05)
        adam = [x for x in extensions if x[0] == 'adam']
        self.assertEqual(len(adam), 4)  # Cap of the pre-registered border rule.
        self.assertIn('adam', stage_1['unresolved_borders'])
        self.assertEqual(stage_1['grids']['sgd'], [0.001, 0.01, 0.05, 0.1, 0.3, 1.0])
        # Every seed is 42 before stage 2, and identical runs of stage 0 are requested again (reused).
        self.assertEqual({seed for _, seed in calls}, {42})

    def test_full_flow_until_stage_6(self):
        run_labeled, calls = fake_runner(self.score)
        with patch.object(StagedSearch, 'run_labeled', run_labeled):
            search = StagedSearch(CONFIG, self.output)
            with self.assertRaises(ValueError):
                search.run_stage(2)  # Requires stage 1.
            for stage in range(7):
                result = search.run_stage(stage)
            stage_2 = json.loads((self.output / 'stage_2.json').read_text())
            self.assertEqual(stage_2['winner']['optimizer'], 'momentum')
            self.assertEqual([r['optimizer'] for r in stage_2['top']], ['momentum', 'rmsprop'])
            stage_3 = json.loads((self.output / 'stage_3.json').read_text())
            # 512 vs 256 differ by 0.256 pp < 0.3 pp: seeds 0 and 1 for both.
            near = stage_3['decision']['near_tie']
            self.assertIsNotNone(near)
            self.assertEqual(sorted(r['seed'] for r in near['extra_runs']), [0, 0, 1, 1])
            self.assertEqual([len(g['seeds']) for g in near['groups']], [3, 3])
            self.assertEqual(result['architectures'][0], json.loads(
                (self.output / 'stage_4.json').read_text())['winner']['architecture'])
            self.assertEqual(len(result['ranking']), 8)  # 2 optimizers x 2 rates x 2 architectures
            self.assertTrue(all(g['seeds'] == [0, 1, 42] for g in result['ranking']))
            self.assertEqual(result['learning_rates']['momentum']['learning_rates'], [0.05, 0.01])
            stage_7 = search.run_stage(7)
            stage_0 = json.loads((self.output / 'stage_0.json').read_text())
            groups = stage_7['ranking']
            self.assertEqual(len(groups), 4)  # Top-3 of stage 6 plus the stage-0 control.
            self.assertIn(stage_0['control']['config_id'], [g['config_id'] for g in groups])
            self.assertTrue(all(g['seeds'] == [0, 1, 2, 3, 42] for g in groups))
            self.assertEqual(groups[0]['config_id'], result['ranking'][0]['config_id'])
            self.assertIn('winner_vs_control', stage_7['evidence'])
            selection = json.loads((self.output / 'selection.json').read_text())
            self.assertEqual(selection['config_id'], groups[0]['config_id'])
            self.assertEqual(selection['config']['dataset'], 'data/digits.csv')
            self.assertEqual(selection['retraining_epochs'], 7)
            self.assertTrue(selection['candidate_model'].startswith('runs/'))
            self.assertEqual(selection['search_sha256'], search.sha)
            # The command summary works for every stage, including stage 1 (one best rate per optimizer).
            for stage in range(8):
                text = _winner_text(json.loads((self.output / f'stage_{stage}.json').read_text()))
                self.assertNotEqual(text, 'none', f'stage {stage}')
            stage_1_text = _winner_text(json.loads((self.output / 'stage_1.json').read_text()))
            self.assertIn('sgd 0.3', stage_1_text)
            self.assertIn('momentum 0.05', stage_1_text)
            # A changed protocol cannot reuse previous decisions.
            changed = Path(self.temp.name) / 'search_v2.json'
            protocol = json.loads(CONFIG.read_text(encoding='utf-8'))
            protocol['stages']['3']['widths'] = [64, 128]
            changed.write_text(json.dumps(protocol), encoding='utf-8')
            with self.assertRaises(ValueError):
                StagedSearch(changed, self.output).run_stage(3)


class PortabilityAndIsolationTests(unittest.TestCase):
    def test_protocol_hash_ignores_line_endings(self):
        with TemporaryDirectory() as tmp:
            text = CONFIG.read_bytes().replace(b'\r\n', b'\n')
            (Path(tmp) / 'lf.json').write_bytes(text)
            (Path(tmp) / 'crlf.json').write_bytes(text.replace(b'\n', b'\r\n'))
            self.assertEqual(protocol_sha256(Path(tmp) / 'lf.json'), protocol_sha256(Path(tmp) / 'crlf.json'))
            self.assertEqual(protocol_sha256(CONFIG), protocol_sha256(Path(tmp) / 'lf.json'))

    def test_config_id_ignores_absolute_location(self):
        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            configs = []
            for root in (Path(a) / 'machine-one' / 'tp3', Path(b) / 'other' / 'place' / 'tp3'):
                (root / 'data').mkdir(parents=True)
                synthetic_csv(root / 'data' / 'digits.csv')
                config = {'dataset': str(root / 'data' / 'digits.csv'),
                          'cache': str(root / 'ej2' / 'cache' / 'digits.npz'), 'architecture': [784, 4, 10]}
                configs.append((validate_config(config), root))
            (first, root_a), (second, root_b) = configs
            self.assertNotEqual(config_identity(first), config_identity(second))  # v1 ids keep absolute paths
            self.assertEqual(config_identity(first, path_root=root_a), config_identity(second, path_root=root_b))
            self.assertEqual(recorded_config(first, root_a)['dataset'], 'data/digits.csv')
            reports = [experiments.run({**c, 'epochs': 1}, root / 'runs', path_root=root) for c, root in configs]
            self.assertEqual(reports[0]['config_id'], reports[1]['config_id'])
            self.assertEqual(reports[0]['config'], reports[1]['config'])
            for r, root in zip(reports, (root_a, root_b)):
                self.assertEqual(r['dataset']['source'], 'data/digits.csv')
                self.assertEqual(r['config']['cache'], 'ej2/cache/digits.npz')
                self.assertEqual(r['run_directory'], f'runs/{r["run_id"]}')
                manifest = json.loads((root / r['run_directory'] / 'manifest.json').read_text())
                self.assertEqual(manifest['identity']['config']['dataset'], 'data/digits.csv')

    def test_staged_search_rejects_digits_test(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'data').mkdir()
            synthetic_csv(root / 'data' / 'digits_test.csv')  # Synthetic stand-in, never the real file.
            protocol = json.loads(CONFIG.read_text(encoding='utf-8'))
            for dataset in ('data/digits_test.csv', 'data/DIGITS_TEST.csv'):
                protocol['base']['dataset'] = dataset
                path = root / 'search.json'
                path.write_text(json.dumps(protocol), encoding='utf-8')
                with patch.object(experiments, 'cargar') as loader, self.assertRaises(ValueError):
                    StagedSearch(path, root / 'v2', root=root).run_stage(0)
                loader.assert_not_called()
            alias = root / 'data' / 'digits.csv'
            protocol['base']['dataset'] = 'data/digits.csv'
            path.write_text(json.dumps(protocol), encoding='utf-8')
            with alias_to(alias, root / 'data' / 'digits_test.csv'), \
                    patch.object(experiments, 'cargar') as loader, self.assertRaises(ValueError):
                StagedSearch(path, root / 'v2', root=root).run_stage(0)
            loader.assert_not_called()


class SmokeTest(unittest.TestCase):
    """Real but tiny runs of stages 0 and 1: technical check, not an experiment."""

    def test_stage_0_and_1_smoke(self):
        original = experiments.particionar

        def small_train(*args, **kwargs):
            Xt, yt, Xv, yv, train, validation = original(*args, **kwargs)
            return Xt[:256], yt[:256], Xv, yv, train[:256], validation

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = json.loads(CONFIG.read_text(encoding='utf-8'))
            protocol['base']['epochs'] = 2
            config = root / 'search_v2_smoke.json'
            config.write_text(json.dumps(protocol), encoding='utf-8')
            with patch.object(experiments, 'particionar', small_train):
                search = StagedSearch(config, root / 'v2')
                stage_0 = search.run_stage(0)
                stage_1 = search.run_stage(1)
                again = search.run_stage(0)
            self.assertEqual(again, stage_0)
            runs = list((root / 'v2' / 'runs').glob('*/results.json'))
            rows = stage_0['runs'] + [r for group in stage_1['runs'].values() for r in group]
            self.assertEqual(len(runs), len({r['run_id'] for r in rows}))
            report = json.loads(runs[0].read_text(encoding='utf-8'))
            self.assertEqual(report['config']['dataset'], 'data/digits.csv')
            self.assertEqual(report['dataset']['source'], 'data/digits.csv')
            self.assertEqual(report['dataset']['train_samples'], 256)
            self.assertEqual(report['metadata']['search_sha256'], search.sha)
            self.assertNotIn(json.dumps(str(TP3))[1:-1], runs[0].read_text(encoding='utf-8'))
            self.assertTrue(all(r['stop_reason'] in ('max_epochs', 'numerical_failure') for r in rows))
            self.assertFalse(any(r['converged'] for r in rows))  # 2 epochs: reported as not converged.
            try:
                import matplotlib  # noqa: F401
            except ImportError:
                return
            from tps_sia.tp3.ej2.src.search_analysis import generate_v2
            with patch.object(experiments, 'cargar', side_effect=AssertionError('Analysis must not load data')):
                generate_v2(root / 'v2', root / 'analysis')
            for name in ('comparison_v2.md', 'stage_0_accuracy.png', 'stage_0_curves.png',
                         'stage_1_rates.png', 'stage_1.csv'):
                self.assertTrue((root / 'analysis' / name).is_file(), name)


if __name__ == '__main__':
    unittest.main()
