"""Control de entrenamiento: python3 -m unittest tps_sia.tp3.shared.tests.test_training_state."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from tps_sia.tp3.shared.mlp import MLP
from tps_sia.tp3.shared.optimizers import SGD, Momentum, Adam


X = np.random.default_rng(12).normal(size=(11, 3))
Y = np.eye(2)[(X[:, 0] > 0).astype(int)]


def equal(test, a, b):
    if isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        test.assertEqual(set(a), set(b))
        for key in a:
            if key != 'tiempo_segundos':
                equal(test, a[key], b[key])
    elif isinstance(a, list):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            equal(test, x, y)
    else:
        test.assertEqual(a, b)


class TrainingStateTests(unittest.TestCase):
    def model(self, optimizer=SGD):
        return MLP([3, 5, 2], semilla=7, tamano_lote=4, optimizador=optimizer())

    def test_exact_resume_all_optimizers_and_logging(self):
        for optimizer in (SGD, Momentum, Adam):
            with self.subTest(optimizer=optimizer), TemporaryDirectory() as directory:
                path = Path(directory)
                full, split = self.model(optimizer), self.model(optimizer)
                options = dict(X_val=X, y_val=Y, patience=20, min_delta=1e-5)
                full.entrenar(X, Y, epocas=7, **options)
                split.preprocessing = {'name': 'scale', 'divisor': 255.0}
                split.entrenar(X, Y, epocas=7, pause_after=3, checkpoint_path=path/'checkpoint.npz',
                               weight_log_path=path/'weights.jsonl', selected_weights=[(0, 0, 1)], **options)
                self.assertEqual(split.historia.stop_reason, 'interrupted')
                restored = MLP.cargar(path/'checkpoint.npz')
                self.assertEqual(restored.preprocessing, split.preprocessing)
                restored.entrenar(X, Y, epocas=4, X_val=X, y_val=Y)
                equal(self, full._snapshot(), restored._snapshot())
                equal(self, full.best_state, restored.best_state)
                self.assertEqual(restored.historia.epochs, list(range(1, 8)))
                self.assertEqual(restored.historia.updates, [3*i for i in range(1, 8)])
                records = [json.loads(line) for line in (path/'weights.jsonl').read_text().splitlines()]
                self.assertEqual([row['update'] for row in records], list(range(1, 22)))
                self.assertEqual(records[-1]['epoch'], 7)
                self.assertEqual(records[-1]['selected_weights'][0]['index'], [0, 0, 1])

    def test_validation_does_not_change_parameters_optimizer_or_rng(self):
        a, b = self.model(Adam), self.model(Adam)
        a.entrenar(X, Y, epocas=4)
        b.entrenar(X, Y, epocas=4, X_val=X, y_val=1-Y)
        equal(self, a.parametros, b.parametros)
        equal(self, a.optimizador.export_state(), b.optimizador.export_state())
        equal(self, a.rng.bit_generator.state, b.rng.bit_generator.state)

    def test_best_and_last_are_separate_and_best_can_resume(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            model = self.model(Adam)
            options = dict(X_val=X, y_val=1-Y, patience=2, min_delta=100.)
            model.entrenar(X, Y, epocas=10, checkpoint_path=path/'checkpoint.npz',
                           best_model_path=path/'best_model.npz', **options)
            self.assertEqual(model.historia.stop_reason, 'early_stopping')
            self.assertEqual(model.epochs_completed, 3)
            self.assertEqual(model.early_stopping['best_epoch'], 1)
            best, last = MLP.cargar(path/'best_model.npz'), MLP.cargar(path/'checkpoint.npz')
            self.assertEqual(best.epochs_completed, 1)
            self.assertEqual(last.epochs_completed, 3)
            reference = self.model(Adam)
            reference.entrenar(X, Y, epocas=1, **options)
            reference.historia.stop_reason = None
            equal(self, best._snapshot(), reference._snapshot())
            best.entrenar(X, Y, epocas=1, **options)
            reference.entrenar(X, Y, epocas=1, **options)
            equal(self, best._snapshot(), reference._snapshot())
            before = last._snapshot()
            last.guardar_best(path/'export.npz')
            equal(self, before, last._snapshot())
            np.testing.assert_array_equal(MLP.cargar(path/'export.npz').predecir(X), model_from_state(model).predecir(X))

    def test_stop_reasons_and_accuracy_monitor(self):
        model = self.model()
        model.entrenar(X, Y, epocas=2, epsilon=100.)
        self.assertEqual(model.historia.stop_reason, 'training_epsilon')
        model = self.model()
        model.entrenar(X, Y, epocas=3, X_val=X, y_val=Y, monitor='validation_accuracy', patience=1, min_delta=1.)
        self.assertEqual(model.historia.stop_reason, 'early_stopping')
        self.assertEqual(model.early_stopping['best_value'], model.historia.accuracy_validacion[0])

    def test_min_delta_controls_patience_but_exports_actual_best(self):
        model = self.model()
        with patch.object(model, '_costo', side_effect=[0.6, 0.5, 0.5, 0.49, 0.4, 0.48]):
            model.entrenar(X, Y, epocas=10, X_val=X, y_val=Y, patience=2, min_delta=0.1)
        self.assertEqual(model.historia.stop_reason, 'early_stopping')
        self.assertEqual(model.early_stopping['best_epoch'], 3)
        self.assertEqual(model.early_stopping['best_value'], 0.48)
        self.assertEqual(model.early_stopping['bad_epochs'], 2)

    def test_mid_epoch_interrupt_and_numerical_failure_rollback(self):
        for failure, reason in ((KeyboardInterrupt(), 'interrupted'), (FloatingPointError(), 'numerical_failure')):
            with self.subTest(reason=reason), TemporaryDirectory() as directory:
                model = self.model(Adam)
                reference = self.model(Adam)
                original = model._gradientes
                calls = 0
                def fail(X, y):
                    nonlocal calls
                    calls += 1
                    if calls == 5:
                        raise failure
                    return original(X, y)
                with patch.object(model, '_gradientes', side_effect=fail):
                    model.entrenar(X, Y, epocas=3, checkpoint_path=Path(directory)/'state.npz')
                self.assertEqual(model.historia.stop_reason, reason)
                self.assertEqual(model.epochs_completed, 1)
                restored = MLP.cargar(Path(directory)/'state.npz')
                restored.entrenar(X, Y, epocas=2)
                reference.entrenar(X, Y, epocas=3)
                equal(self, restored._snapshot(), reference._snapshot())

    def test_periodic_checkpoints_and_atomic_failure(self):
        model = self.model()
        with TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.npz'
            original = model.guardar
            epochs = []
            def save(path):
                epochs.append(model.epochs_completed)
                original(path)
            with patch.object(model, 'guardar', side_effect=save):
                model.entrenar(X, Y, epocas=5, checkpoint_path=path, checkpoint_every=2)
            self.assertEqual(epochs, [2, 4, 5])
            contents = path.read_bytes()
            with patch('tps_sia.tp3.shared.mlp.np.savez_compressed', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    model.guardar(path)
            self.assertEqual(path.read_bytes(), contents)
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])

    def test_legacy_versions_migrate_and_resume(self):
        for version in (1, 2):
            with self.subTest(version=version), TemporaryDirectory() as directory:
                path = Path(directory)/'legacy.npz'
                model = self.model()
                model.entrenar(X, Y, epocas=2)
                model.guardar(path)
                with np.load(path, allow_pickle=False) as data:
                    arrays = {key: data[key] for key in data.files}
                config = json.loads(str(arrays['configuracion']))
                config['version'] = version
                config.pop('training_state')
                for key in ('epochs', 'validation_epochs', 'updates', 'learning_rate', 'stop_reason'):
                    config['historia'].pop(key)
                if version == 1:
                    config.pop('optimizer_state')
                arrays['configuracion'] = json.dumps(config)
                np.savez(path, **arrays)
                legacy = MLP.cargar(path)
                np.testing.assert_array_equal(legacy.predecir(X), model.predecir(X))
                self.assertIsNone(legacy.seed)
                legacy.guardar(path)
                migrated = MLP.cargar(path)
                migrated.entrenar(X, Y, epocas=1)
                self.assertEqual(migrated.historia.epochs, [1, 2, 3])
                self.assertEqual(migrated.historia.learning_rate[:2], [None, None])

    def test_invalid_options_do_not_train(self):
        for options in ({'patience': 0}, {'patience': 2}, {'min_delta': -1},
                        {'monitor': 'test_accuracy'}, {'checkpoint_every': 0},
                        {'selected_weights': [(8, 0, 0)]}, {'pause_after': True}, {'checkpoint_path': 'same.npz', 'best_model_path': 'same.npz'}):
            model = self.model()
            before = model._snapshot()
            with self.assertRaises(ValueError):
                model.entrenar(X, Y, **options)
            equal(self, before, model._snapshot())


def model_from_state(model):
    import copy
    result = copy.deepcopy(model)
    result._restore(model.best_state)
    return result


if __name__ == '__main__':
    unittest.main()
