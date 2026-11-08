import json
from pathlib import Path
import tempfile
import unittest

from src.compare import run_comparison


class ComparisonTests(unittest.TestCase):
    def test_short_stage_preserves_runs_and_selects_traceable_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            variant = {'name': 'base', 'architecture': [35, 20, 2, 20, 35],
                       'optimizer': {'name': 'adam', 'learning_rate': 0.001},
                       'batch_size': 32, 'max_epochs': 1, 'evaluate_every': 1,
                       'stop_on_tolerance': False}
            protocol = {'stage': 'synthetic-test', 'approval': 'automated short consistency test',
                        'seeds': [1, 2], 'selection': 'max_mean_mse', 'variants': [variant]}
            config = root / 'protocol.json'
            config.write_text(json.dumps(protocol), encoding='utf-8')
            summary = run_comparison(config, root / 'comparison')
            self.assertEqual(len(summary['runs']), 2)
            self.assertEqual(summary['budget']['max_epochs_total'], 2)
            group = summary['aggregates'][0]
            self.assertEqual(group['completed'], 2)
            self.assertEqual(group['failed'], 0)
            self.assertEqual(summary['selected_variant'], 'base')
            self.assertEqual(group['max_pixel_errors_mean'],
                             sum(row['max_pixel_errors'] for row in summary['runs']) / 2)
            chosen = summary['selected']
            self.assertTrue(Path(chosen['model']['path']).is_file())
            self.assertIn(chosen['run_id'], [row['run_id'] for row in summary['runs']])
            for name in ['runs.csv', 'aggregates.csv', 'comparison.png', 'summary.json', 'protocol.json']:
                self.assertTrue((root / 'comparison' / name).is_file(), name)
            protocol['reuse_runs'] = {row['run_id']: str(root / 'comparison' / row['run_id'])
                                      for row in summary['runs']}
            config.write_text(json.dumps(protocol), encoding='utf-8')
            reused = run_comparison(config, root / 'reused')
            self.assertEqual(reused['selected']['model'], chosen['model'])
            self.assertEqual(reused['budget']['max_epochs_total'], 0)
            self.assertTrue(all(row['reused'] for row in reused['runs']))

    def test_failed_runs_remain_visible_and_are_not_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = {'stage': 'invalid-config-test', 'approval': 'short consistency test',
                        'seeds': [1], 'selection': 'max_mean_mse', 'variants': [
                            {'name': 'bad', 'architecture': [35, 2, 35],
                             'optimizer': {'name': 'unknown'}, 'batch_size': 32,
                             'max_epochs': 1, 'evaluate_every': 1, 'stop_on_tolerance': False}]}
            config = root / 'protocol.json'
            config.write_text(json.dumps(protocol), encoding='utf-8')
            summary = run_comparison(config, root / 'comparison')
            self.assertEqual(summary['runs'][0]['status'], 'failed')
            self.assertIn('error', summary['runs'][0])
            self.assertEqual(summary['aggregates'][0]['failed'], 1)
            self.assertIsNone(summary['selected'])


if __name__ == '__main__':
    unittest.main()
