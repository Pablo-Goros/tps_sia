"""Public model contracts for denoising and persistence."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from src.autoencoder import Autoencoder
from src.dataset import load_dataset
from src.explore import visualize_latent

class DenoiseModelTests(unittest.TestCase):
    def test_clean_compatibility_and_separate_targets_survive_roundtrip(self):
        clean = load_dataset().vectors
        noisy = 1-clean
        for width in (2, 10):
            with self.subTest(width=width), tempfile.TemporaryDirectory() as folder:
                a = Autoencoder([35, 8, width, 8, 35], seed=17, loss='bce')
                b = Autoencoder([35, 8, width, 8, 35], seed=17, loss='bce')
                a.train(noisy, 2, targets=clean)
                b.train(noisy, 1, targets=clean)
                path = Path(folder)/'model.npz'
                b.save(path)
                b = Autoencoder.load(path)
                b.train(noisy, 1, targets=clean)
                np.testing.assert_array_equal(a.reconstruct(clean), b.reconstruct(clean))
                np.testing.assert_array_equal(b.decode(b.encode(clean)), b.reconstruct(clean))
                self.assertEqual(b.encode(clean).shape, (32, width))
                self.assertEqual(a.history['updates'], b.history['updates'])
                with self.assertRaises(ValueError):
                    b.train(noisy, 1, targets=clean[:1])
                if width == 10:
                    with self.assertRaisesRegex(ValueError, 'two-dimensional'):
                        visualize_latent(path, Path(folder)/'explore')
        a, b = Autoencoder(seed=17), Autoencoder(seed=17)
        a.train(clean, 2)
        b.train(clean, 2, targets=clean)
        np.testing.assert_array_equal(a.reconstruct(clean), b.reconstruct(clean))

    def test_short_training_saves_last_and_best_without_restarting(self):
        from src.noise import make_collection
        from src.denoise_training import train_run
        import json
        clean = load_dataset().vectors
        validation = make_collection(clean, [0,.05,.1,.2], 2, 123)
        config = {'variant': 'smoke', 'seed': 42, 'max_epochs': 2, 'evaluate_every': 1,
                  'model': {'architecture': [35,8,10,8,35], 'loss': 'bce'},
                  'training_levels': [0,.05,.1,.2], 'selection_levels': [.05,.1,.2]}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'run'
            result = train_run(config, clean, validation, output)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['epochs_completed'], 2)
            history = json.loads((output/'history.json').read_text())
            self.assertEqual([v['epoch'] for v in history['validation']], [0,1,2])
            self.assertTrue((output/'last_model.npz').exists())
            self.assertEqual(Autoencoder.load(output/'best_model.npz').history['epocas_corridas'], result['best']['epoch'])
            with self.assertRaises(ValueError):
                train_run(config, clean, validation, output)

    def test_campaign_records_interruption_without_selection_or_retry(self):
        import io
        import json
        from contextlib import redirect_stdout
        from src.denoise import prepare_campaign
        from src.denoise_campaign import train_campaign, sha256
        from src.train import write_json

        class InterruptOutput(io.StringIO):
            def write(self, message):
                if 'epoch=0' in message:
                    raise KeyboardInterrupt()
                return super().write(message)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = Autoencoder([35,4,2,4,35], seed=17)
            model.save(root/'model.npz')
            source = dict(architecture=[35,4,2,4,35], seed=17,
                          optimizer={'name': 'adam', 'learning_rate': .001}, batch_size=32)
            write_json(root/'config.json', source)
            write_json(root/'manifest.json', {'config': source, 'effective_model': model.configuration,
                       'model': {'sha256': sha256(root/'model.npz')},
                       'dataset': {'sha256': load_dataset().sha256}})
            write_json(root/'selection.json', {'model': {'path': 'model.npz', 'sha256': sha256(root/'model.npz')},
                       'source_config': 'config.json', 'configuration': source})
            protocol = json.loads((Path(__file__).resolve().parents[1]/'configs/denoising.json').read_text())
            protocol['source_selection'] = 'selection.json'
            write_json(root/'protocol.json', protocol)
            campaign = root/'campaign'
            prepare_campaign(root/'protocol.json', campaign, tp5_root=root)
            with redirect_stdout(InterruptOutput()), self.assertRaises((RuntimeError, KeyboardInterrupt)):
                train_campaign(campaign)
            status = json.loads((campaign/'training.json').read_text())
            self.assertEqual(status['status'], 'interrupted')
            self.assertEqual(status['epochs_completed'], 0)
            self.assertFalse((campaign/'selection.json').exists())
            with self.assertRaises(ValueError):
                train_campaign(campaign)
