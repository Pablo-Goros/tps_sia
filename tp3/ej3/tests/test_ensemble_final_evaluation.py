import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tps_sia.tp3.shared.experiments import sha256_file
from tps_sia.tp3.shared.mlp import MLP
from tps_sia.tp3.ej3.src import ensemble_final_evaluation as evaluator


class EnsembleFinalTests(unittest.TestCase):
    def test_dry_run_never_reads_test_or_creates_output(self):
        with TemporaryDirectory() as temp:
            output = Path(temp) / 'final'
            with patch.object(evaluator, 'frozen_ensemble', return_value=({}, {'members': [1, 2]})), \
                    patch.object(evaluator, 'cargar', side_effect=AssertionError('Test read')):
                r = evaluator.evaluate('config', 'ensemble', output, dry_run=True)
            self.assertFalse(r['test_loaded'])
            self.assertFalse(output.exists())

    def test_synthetic_evaluation_freezes_before_test_and_refuses_overwrite(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'data').mkdir()
            (root / 'data/digits_test.csv').write_text('synthetic test fixture')
            source = root / 'selection.json'
            source.write_text('{}')
            ensemble = root / 'ensemble.json'
            ensemble.write_text('{}')
            X = np.array([[0., 1.], [1., 0.], [1., 1.]])
            y = np.eye(10)[[0, 1, 2]]
            members, predictions = [], []
            for seed in (0, 1):
                model = MLP([2, 4, 10], semilla=seed)
                path = root / f'model-{seed}.npz'
                model.guardar(path)
                members.append({'model': path.name, 'model_sha256': sha256_file(path),
                                'preprocessing': {'name': 'identity'}, 'weight': 0.5})
                predictions.append(model.predecir(X))
            frozen = {'members': members, 'ensemble_manifest_sha256': sha256_file(ensemble),
                      'source_selection_sha256': sha256_file(source)}
            manifest = {'source_selection': source.name, 'validation': {'accuracy': 0.5}}
            output = root / 'final'
            def loader(*args):
                self.assertTrue((output / 'frozen_ensemble.json').exists())
                return X, y
            with patch.object(evaluator, 'frozen_ensemble', return_value=(manifest, frozen)), \
                    patch.object(evaluator, 'cargar', side_effect=loader):
                result = evaluator.evaluate('config', ensemble, output, root=root)
            with np.load(output / 'predictions.npz') as saved:
                np.testing.assert_allclose(saved['probabilities'], np.mean(predictions, axis=0))
                self.assertEqual(result['correct'], int(np.sum(saved['predicted'] == saved['actual'])))
            with patch.object(evaluator, 'cargar', side_effect=AssertionError('Test reread')):
                with self.assertRaises(FileExistsError):
                    evaluator.evaluate('config', ensemble, output, root=root)

    def test_changed_weights_are_rejected_before_loading_models(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'selection.json'
            source.write_text('{}')
            confirmation = {'ranking': [{'config_id': 'a', 'config': {}, 'status': 'completed'}]}
            (root / 'stage_confirmation.json').write_text(json.dumps(confirmation))
            manifest = {'protocol': 'ej3-ensemble-validation-v1', 'identity': {},
                        'source_selection': source.name, 'source_selection_sha256': sha256_file(source),
                        'top_configurations': 1, 'config_ids': ['a'],
                        'members': [{'config_id': 'a', 'seed': 42, 'weight': 0.9}]}
            ensemble = root / 'ensemble.json'
            ensemble.write_text(json.dumps(manifest))
            development = SimpleNamespace(identity={}, protocol={'confirmation_seeds': [42]})
            with patch.object(evaluator, 'Development', return_value=development), \
                    patch.object(evaluator, 'frozen_candidate', return_value=(None, {}, None)), \
                    patch.object(evaluator.MLP, 'cargar', side_effect=AssertionError('Model loaded')):
                with self.assertRaisesRegex(ValueError, 'membership or equal weights'):
                    evaluator.frozen_ensemble('config', ensemble, root=root)


if __name__ == '__main__':
    unittest.main()
