"""Selection uses common seeds and provenance, and rate comparisons stay controlled."""
import copy
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tps_sia.tp3.ej2.src.experiments import CONFIG, load_search
from tps_sia.tp3.ej2.src.search import _job, confirm_groups, rank, validate_rates
from tps_sia.tp3.shared.experiments import DEFAULTS, run
from tps_sia.tp3.shared.tests.test_experiments import synthetic_csv


def report(config_id, seed, accuracy, loss=0.2, parameters=100):
    return {'config_id': config_id, 'run_id': f'{config_id}-{seed}', 'status': 'completed',
            'config': {'model_seed': seed}, 'parameter_count': parameters,
            'chosen_epoch': 10, 'duration_seconds': 2.0,
            'dataset': {'sha256': 'dataset', 'split_sha256': 'split'},
            'metrics': {'validation': {'accuracy': accuracy, 'loss': loss,
                                       'macro_f1': accuracy, 'balanced_accuracy': accuracy}}}


class SearchTests(unittest.TestCase):
    def test_mean_ranks_configurations_instead_of_best_seed(self):
        runs = [report('a', 42, 0.99), report('a', 0, 0.8), report('a', 1, 0.8)]
        runs += [report('b', seed, 0.9) for seed in (42, 0, 1)]
        summary = confirm_groups(runs, [42, 0, 1])
        self.assertEqual(summary[0]['config_id'], 'b')
        self.assertGreater(summary[1]['accuracy_std'], 0)

    def test_missing_duplicate_and_failed_seeds_are_rejected(self):
        runs = [report('a', seed, 0.9) for seed in (42, 0, 1)]
        for invalid in (runs[:2], runs + [runs[0]], runs + [report('b', 42, 0.99)]):
            with self.assertRaises(ValueError):
                confirm_groups(invalid, [42, 0, 1])
        runs[1]['status'] = 'failed'
        with self.assertRaises(ValueError):
            confirm_groups(runs, [42, 0, 1])

    def test_incomparable_partitions_are_rejected(self):
        runs = [report('a', seed, 0.9) for seed in (42, 0, 1)]
        runs[1]['dataset']['split_sha256'] = 'other'
        with self.assertRaises(ValueError):
            confirm_groups(runs, [42, 0, 1])

    def test_ties_and_failures(self):
        runs = [report('large', 42, 0.9, parameters=200), report('small', 42, 0.9),
                report('loss', 42, 0.9, loss=0.1)]
        self.assertEqual([r['config_id'] for r in rank(runs)], ['loss', 'small', 'large'])
        runs[2]['status'] = 'failed'
        self.assertEqual(rank(runs)[0]['config_id'], 'small')

    def test_rate_stage_enforces_controlled_comparisons(self):
        search = load_search(CONFIG)
        validate_rates(search, CONFIG.parent)
        changed = copy.deepcopy(search)
        changed['candidates'][1]['overrides']['split_seed'] = 99
        with self.assertRaises(ValueError):
            validate_rates(changed, CONFIG.parent)
        changed = copy.deepcopy(search)
        changed['candidates'][1]['overrides']['optimizer']['learning_rate'] = 0.5
        with self.assertRaises(ValueError):
            validate_rates(changed, CONFIG.parent)
        changed = copy.deepcopy(search)
        changed['candidates'][0]['name'], changed['candidates'][3]['name'] = (
            changed['candidates'][3]['name'], changed['candidates'][0]['name'])
        with self.assertRaises(ValueError):
            validate_rates(changed, CONFIG.parent)

    def test_parallel_runs_reuse_results_and_reject_changed_model(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic_csv(root / 'digits.csv')
            config = copy.deepcopy(DEFAULTS)
            config.update(dataset=str(root / 'digits.csv'), cache=str(root / 'cache.npz'),
                          architecture=[784, 4, 10], epochs=1)
            # Populate the shared cache before launching independent workers.
            from tps_sia.tp3.shared.digit_dataset import cargar
            cargar(config['dataset'], config['cache'])
            with ProcessPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(_job, dict(config, model_seed=seed), 'candidate', root, 'protocol')
                           for seed in (42, 0)]
                reports = [future.result() for future in futures]
            self.assertTrue(all(r['status'] == 'completed' for r in reports))
            self.assertEqual(reports[0], _job(config, 'candidate', root, 'protocol'))
            best = Path(reports[0]['run_directory']) / 'best_model.npz'
            with best.open('ab') as file:
                file.write(b'changed')
            with self.assertRaises(ValueError):
                _job(config, 'candidate', root, 'protocol')

    def test_interrupted_run_resumes_without_replacing_completed_runs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic_csv(root / 'digits.csv')
            config = copy.deepcopy(DEFAULTS)
            config.update(dataset=str(root / 'digits.csv'), cache=str(root / 'cache.npz'),
                          architecture=[784, 4, 10], epochs=2)
            partial = run(config, root / 'runs', pause_after=1,
                          metadata={'candidate': 'candidate', 'protocol_sha256': 'protocol'})
            self.assertEqual(partial['status'], 'interrupted')
            resumed = _job(config, 'candidate', root, 'protocol')
            self.assertEqual(resumed['status'], 'completed')
            self.assertEqual(resumed['history']['epocas_corridas'], 2)


if __name__ == '__main__':
    unittest.main()
