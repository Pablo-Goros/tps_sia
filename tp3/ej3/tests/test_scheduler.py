from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from tps_sia.tp3.shared.mlp import MLP
from tps_sia.tp3.shared.optimizers import Momentum
from tps_sia.tp3.shared.schedulers import plateau_step, validate_scheduler


class SchedulerTests(unittest.TestCase):
    def test_plateau_reduces_next_rate_and_respects_floor(self):
        config = validate_scheduler({'patience': 2, 'min_lr': 0.00375})
        state = {'best': None, 'bad_epochs': 0, 'reductions': 0, 'current_lr': 0.015}
        rate = 0.015
        values = []
        for loss in [1, 1, 1, 1, 1, 1, 1]:
            rate = plateau_step(config, state, loss, rate)
            values.append(rate)
        self.assertEqual(values, [0.015, 0.015, 0.0075, 0.0075, 0.00375, 0.00375, 0.00375])
        self.assertEqual(state['reductions'], 2)
        plateau_step(config, state, 0.9, rate)
        self.assertEqual(state['bad_epochs'], 0)

    def test_exact_resume_with_momentum_augmentation_and_best_checkpoint(self):
        rng = np.random.default_rng(2)
        X = rng.uniform(0, 1, (8, 784))
        y = np.eye(10)[np.arange(8)]
        def model():
            return MLP([784, 4, 10], semilla=9, tamano_lote=4,
                       optimizador=Momentum(0.015, 0.9),
                       augmentation={'name': 'translation', 'max_shift': 1},
                       lr_scheduler={'patience': 2, 'min_delta': 100, 'min_lr': 0.001875})
        full, partial = model(), model()
        full.entrenar(X, y, epocas=8, X_val=X, y_val=y, patience=None)
        with TemporaryDirectory() as temp:
            checkpoint, best = Path(temp)/'checkpoint.npz', Path(temp)/'best.npz'
            partial.entrenar(X, y, epocas=8, X_val=X, y_val=y, patience=None,
                             pause_after=4, checkpoint_path=checkpoint, best_model_path=best)
            resumed = MLP.cargar(checkpoint)
            resumed.entrenar(X, y, epocas=4, X_val=X, y_val=y, patience=None,
                             checkpoint_path=checkpoint, best_model_path=best)
            for a, b in zip(full.parametros, resumed.parametros):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(full.historia.learning_rate, resumed.historia.learning_rate)
            self.assertEqual(full.scheduler_state, resumed.scheduler_state)
            self.assertEqual(full.early_stopping, resumed.early_stopping)
            self.assertEqual(full.rng.bit_generator.state, resumed.rng.bit_generator.state)
            self.assertEqual(full.optimizador.configuracion(), resumed.optimizador.configuracion())
            chosen = MLP.cargar(best)
            self.assertEqual(chosen.epochs_completed, resumed.early_stopping['best_epoch'])
            self.assertEqual(chosen.scheduler_state['current_lr'], chosen.optimizador.learning_rate)

    def test_invalid_settings_and_missing_validation(self):
        for settings in ({'factor': 1}, {'patience': True}, {'min_lr': -1},
                         {'min_delta': float('nan')}, {'unknown': 1}):
            with self.assertRaises(ValueError):
                validate_scheduler(settings)
        with self.assertRaises(ValueError):
            MLP([2, 3, 2], eta=0.001, lr_scheduler={})
        model = MLP([2, 3, 2], eta=0.015, lr_scheduler={})
        with self.assertRaisesRegex(ValueError, 'requires validation'):
            model.entrenar(np.zeros((2, 2)), np.eye(2), epocas=1)


if __name__ == '__main__':
    unittest.main()
