"""Search configuration and exercise command integration without course datasets."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from tps_sia.tp3.ej2.src.experiments import CONFIG, candidate_config, load_search
from tps_sia.tp3.shared.tests.test_experiments import synthetic_csv


class ExerciseExperimentTests(unittest.TestCase):
    def test_predefined_candidates_and_paths(self):
        search = load_search(CONFIG)
        self.assertEqual(len(search['candidates']), 11)
        self.assertEqual(search['seeds'], [42, 0, 1])
        c = candidate_config(search, 'baseline', 42, CONFIG.parent)
        self.assertEqual(Path(c['dataset']).name, 'digits.csv')
        self.assertEqual(c['architecture'], [784, 128, 10])
        self.assertEqual(c['optimizer'], {'name': 'sgd', 'learning_rate': 0.01})
        self.assertEqual(c['epochs'], 30)
        with self.assertRaises(ValueError):
            candidate_config(search, 'baseline', 999, CONFIG.parent)

    def test_commands_and_test_isolation(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic_csv(root / 'digits.csv')
            search = copy.deepcopy(load_search(CONFIG))
            search['base'].update(dataset='digits.csv', cache='cache.npz', architecture=[784, 4, 10], epochs=1)
            search['candidates'] = search['candidates'][:1]
            path = root / 'search.json'
            path.write_text(json.dumps(search))
            command = [sys.executable, '-m', 'tps_sia.tp3.ej2.src.experiments', '--config', str(path), '--output-dir', str(root / 'runs')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            reports = list((root / 'runs').glob('*/results.json'))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
            self.assertEqual(report['metadata']['search'], search)
            self.assertEqual(report['metadata']['candidate'], 'baseline')
            search['base']['dataset'] = 'digits_test.csv'
            path.write_text(json.dumps(search))
            with self.assertRaises(ValueError):
                load_search(path)


if __name__ == '__main__':
    unittest.main()
