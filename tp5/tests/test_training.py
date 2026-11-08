import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.dataset import load_dataset
from src.autoencoder import Autoencoder
from src.metrics import reconstruction_metrics
from src.train import run_training, evaluate_model


class TrainingTests(unittest.TestCase):
    def test_short_public_workflow_and_reload_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'run'
            config = {'architecture': [35,20,2,20,35], 'seed': 42,
                      'optimizer': {'name': 'adam', 'learning_rate': 0.001},
                      'batch_size': 32, 'max_epochs': 4, 'evaluate_every': 2,
                      'stop_on_tolerance': True}
            config_path = Path(folder) / 'config.json'
            config_path.write_text(json.dumps(config))
            result = run_training(config_path, output)
            restored = Autoencoder.load(output / 'model.npz')
            data = load_dataset()
            self.assertEqual(result['metrics'], reconstruction_metrics(data.vectors, restored.reconstruct(data.vectors)))
            for name in ['manifest.json','history.json','metrics.json','curves.png','reconstructions.png','predictions.npz']:
                self.assertTrue((output/name).is_file(), name)
            self.assertEqual(result['manifest']['stop_reason'], 'max_epochs')
            self.assertEqual(result['manifest']['epochs_completed'], 4)
            self.assertEqual(result['manifest']['effective_model']['output'], 'logistica')
            self.assertEqual(result['manifest']['effective_model']['optimizer']['beta1'], 0.9)
            history = json.loads((output/'history.json').read_text())
            self.assertEqual([point['epoch'] for point in history['evaluations']], [0,2,4])
            evaluated = evaluate_model(output/'model.npz', Path(folder)/'evaluation')
            self.assertEqual(evaluated, result['metrics'])
            np.testing.assert_allclose(np.load(output/'predictions.npz')['codes'], restored.encode(data.vectors))
