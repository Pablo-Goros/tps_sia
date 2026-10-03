"""Appendix checks with synthetic data only (digits.csv and digits_test.csv are never read).

    python -m unittest -v tps_sia.tp3.ej2.tests.test_appendix
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tps_sia.tp3.ej2.src import appendix
from tps_sia.tp3.ej2.src.appendix import compare_relu, execute, factor_summary, factor_variants
from tps_sia.tp3.ej2.src.staged_search import CONFIG, load_protocol, materialize
from tps_sia.tp3.shared import experiments
from tps_sia.tp3.shared.experiments import run
from tps_sia.tp3.shared.tests.test_experiments import synthetic_csv

SOURCES = [Path(appendix.__file__), Path(appendix.__file__).with_name('appendix_report.py')]


def row(config_id, seed, accuracy, loss=0.2, parameters=100, **config):
    return {'label': config_id, 'run_id': f'{config_id}-seed-{seed}', 'config_id': config_id, 'seed': seed,
            'status': 'completed', 'accuracy': accuracy, 'loss': loss, 'chosen_epoch': 5,
            'parameter_count': parameters, 'duration_seconds': 1.0, 'converged': True,
            'dataset_sha256': 'd', 'split_sha256': 's', 'config': config}


class VariantTests(unittest.TestCase):
    def test_factors_match_the_protocol_stages(self):
        protocol = load_protocol(CONFIG)
        widths, incumbent = factor_variants(protocol, 'width', 128)
        self.assertEqual([v for v, _ in widths], [[784, w, 10] for w in (32, 64, 128, 256, 512)])
        self.assertEqual(incumbent, [784, 128, 10])
        depth, incumbent = factor_variants(protocol, 'depth', 128)
        self.assertEqual(depth, [([784, 128, 10], 'hidden-128'), ([784, 128, 64, 10], 'hidden-128-64'),
                                 ([784, 128, 64, 32, 10], 'hidden-128-64-32')])
        self.assertEqual(incumbent, [784, 128, 10])
        batches, incumbent = factor_variants(protocol, 'batch_size', 128)
        self.assertEqual([v for v, _ in batches], [16, 32, 64, 128])
        self.assertEqual(incumbent, 32)


class RuleTests(unittest.TestCase):
    def test_relu_comparison_uses_the_stage_7_rule(self):
        seeds = [42, 0, 1, 2, 3]
        tanh = [0.970, 0.972, 0.974, 0.971, 0.973]
        relu = [0.975, 0.977, 0.976, 0.978, 0.979]
        rows = ([row('t', s, a, activation='tanh') for s, a in zip(seeds, tanh)]
                + [row('r', s, a, activation='relu') for s, a in zip(seeds, relu)])
        groups, comparison = compare_relu(rows, seeds, 2.0)
        self.assertEqual(comparison['better'], 'r')
        sigma = ((0.0015811 ** 2 + 0.0015811 ** 2) / 2) ** 0.5
        self.assertAlmostEqual(comparison['sigma'], sigma, places=6)
        self.assertAlmostEqual(comparison['threshold'], 2 * comparison['sigma'])
        self.assertTrue(comparison['significant'])            # 0.5 pp > 2 sigma (0.32 pp)
        close = [row('r', s, a + 0.001, activation='relu') for s, a in zip(seeds, tanh)]
        _, comparison = compare_relu(rows[:5] + close, seeds, 2.0)
        self.assertFalse(comparison['significant'])

    def test_factor_summary_reports_change_from_single_seed(self):
        seeds = [42, 0, 1]
        values = {'a': ([784, 32, 10], (0.95, 0.93, 0.94)), 'b': ([784, 128, 10], (0.94, 0.95, 0.96))}
        rows = [row(cid, s, acc, architecture=arch) for cid, (arch, accs) in values.items()
                for s, acc in zip(seeds, accs)]
        original = [r for r in rows if r['seed'] == 42]
        summary = factor_summary(rows, original, [784, 128, 10], 'architecture', seeds, 2.0)
        by_id = {g['config_id']: g for g in summary['groups']}
        self.assertAlmostEqual(by_id['a']['difference_vs_seed_42'], 0.94 - 0.95)
        self.assertAlmostEqual(by_id['b']['difference_vs_seed_42'], 0.95 - 0.94)
        self.assertEqual(summary['best_seed_42_config_id'], 'a')   # Seed 42 alone picks 'a'...
        self.assertEqual(summary['best_3_seeds_config_id'], 'b')   # ...three seeds pick 'b'.
        self.assertFalse(summary['same_winner'])
        self.assertIsNone(summary['best_vs_incumbent'])            # The best is the incumbent.


class ExecutionTests(unittest.TestCase):
    """Tiny real runs on a synthetic CSV: reuse by config_id and isolation from test."""

    def test_reuse_by_config_id_without_writing_v2(self):
        protocol = load_protocol(CONFIG)
        base = {**protocol['base'], 'architecture': [784, 4, 10], 'epochs': 1, 'checkpoint_every': 1}
        real_loader = experiments.cargar

        def guarded_loader(path, cache):
            if Path(path).name.casefold() == 'digits_test.csv':
                raise AssertionError('digits_test.csv must not be loaded by the appendix.')
            return real_loader(path, cache)

        with TemporaryDirectory() as tmp, patch.object(experiments, 'cargar', guarded_loader):
            root = Path(tmp)
            (root / 'data').mkdir()
            synthetic_csv(root / 'data' / 'digits.csv')
            v2_runs, output = root / 'v2' / 'runs', root / 'appendix'
            run(materialize(base, root, 42), v2_runs, path_root=root)
            jobs = [(materialize(base, root, seed), 'tiny') for seed in (42, 0)]
            metadata = {'appendix': 'test'}
            rows = execute(jobs, v2_runs, output, metadata, root)
            self.assertEqual([r['origin'] for r in rows], ['v2', 'appendix_new'])
            self.assertEqual(len(list(v2_runs.iterdir())), 1)       # Nothing written to v2.
            self.assertEqual(len(list((output / 'runs').iterdir())), 1)
            again = execute(jobs, v2_runs, output, metadata, root)
            self.assertEqual([r['origin'] for r in again], ['v2', 'appendix_reused'])
            self.assertEqual([r['accuracy'] for r in again], [r['accuracy'] for r in rows])
            # Same id with a different stored configuration is an error, not a silent reuse.
            stored = next(v2_runs.iterdir()) / 'results.json'
            report = json.loads(stored.read_text(encoding='utf-8'))
            report['config']['batch_size'] = 999
            stored.write_text(json.dumps(report), encoding='utf-8')
            with self.assertRaises(ValueError):
                execute(jobs[:1], v2_runs, output, metadata, root)


class IsolationTests(unittest.TestCase):
    def test_sources_do_not_mention_the_test_file(self):
        for source in SOURCES:
            self.assertFalse('digits_test' in source.read_text(encoding='utf-8'), source.name)

    def test_import_does_not_load_final_evaluation(self):
        code = ('import sys, tps_sia.tp3.ej2.src.appendix, tps_sia.tp3.ej2.src.appendix_report; '
                'print(sorted(m for m in sys.modules if "final_evaluation" in m))')
        root = Path(appendix.__file__).resolve().parents[4]
        result = subprocess.run([sys.executable, '-c', code], cwd=root, capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), '[]')


if __name__ == '__main__':
    unittest.main(verbosity=2)
