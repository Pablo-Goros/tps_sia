"""Campaign preparation through its public interface, without training."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.autoencoder import Autoencoder
from src.dataset import DEFAULT_DATASET, load_dataset
from src.train import write_json
from src.denoise import prepare_campaign


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = load_dataset()
        self.config = json.loads((Path(__file__).resolve().parents[1]/'configs/denoising.json').read_text())

    def selection(self, architecture, optimizer):
        source = self.root/'source'
        source.mkdir(exist_ok=True)
        config = dict(architecture=architecture, optimizer=optimizer, seed=17,
                      batch_size=32, max_epochs=3000, evaluate_every=100,
                      stop_on_tolerance=True, loss='bce', latent_activation='tanh', lr_scheduler=None)
        model = Autoencoder(architecture=architecture, optimizer=optimizer, loss='bce', seed=17)
        model.save(source/'model.npz')
        digest = hashlib.sha256((source/'model.npz').read_bytes()).hexdigest()
        write_json(source/'config.json', config)
        write_json(source/'manifest.json', {'config': config, 'effective_model': model.configuration,
                   'dataset': {'sha256': self.dataset.sha256}, 'model': {'sha256': digest}})
        write_json(source/'selection.json', {'paths_relative_to': '.', 'model': {'path': 'source/model.npz', 'sha256': digest},
                   'source_config': 'source/config.json', 'configuration': config})
        self.config['source_selection'] = 'source/selection.json'
        write_json(self.root/'campaign.json', self.config)
        return model

    def test_preparation_derives_variants_and_freezes_reference(self):
        for architecture, optimizer in [([35, 5, 2, 5, 35], {'name': 'adam', 'learning_rate': .003}),
                                         ([35, 8, 2, 8, 35], {'name': 'rmsprop', 'learning_rate': .007})]:
            with self.subTest(optimizer=optimizer):
                original = self.selection(architecture, optimizer)
                output = self.root/optimizer['name']
                manifest = prepare_campaign(self.root/'campaign.json', output, tp5_root=self.root)
                self.assertEqual(manifest['variants']['winner']['architecture'], architecture)
                self.assertEqual(manifest['variants']['latent-10']['architecture'], [35, architecture[1], 10, architecture[1], 35])
                self.assertEqual(manifest['variants']['winner']['optimizer'], optimizer)
                self.assertEqual(manifest['variants']['winner']['loss'], 'bce')
                frozen = (output/'reference/model.npz').read_bytes()
                (self.root/'source/model.npz').write_bytes(b'changed source')
                self.assertEqual((output/'reference/model.npz').read_bytes(), frozen)
                restored = Autoencoder.load(output/'reference/model.npz')
                self.assertEqual(restored.configuration, original.configuration)
                self.assertEqual(manifest, json.loads((output/'manifest.json').read_text()))
                self.assertEqual((output/'reference/font.h').read_bytes(), DEFAULT_DATASET.read_bytes())
                self.assertFalse((output/'validation.npz').exists())

    def test_invalid_reference_leaves_no_campaign_and_existing_campaign_is_preserved(self):
        self.selection([35, 5, 2, 5, 35], {'name': 'adam', 'learning_rate': .003})
        output = self.root/'campaign'
        selection_path = self.root/'source/selection.json'
        selection = json.loads(selection_path.read_text())
        selection['model']['sha256'] = '0'*64
        write_json(selection_path, selection)
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            prepare_campaign(self.root/'campaign.json', output, tp5_root=self.root)
        self.assertFalse(output.exists())
        output.mkdir()
        marker = output/'keep.txt'
        marker.write_text('preserve')
        with self.assertRaisesRegex(ValueError, 'existing'):
            prepare_campaign(self.root/'campaign.json', output, tp5_root=self.root)
        self.assertEqual(marker.read_text(), 'preserve')

    def test_reference_must_match_dataset_and_model_configuration(self):
        for field in ('dataset', 'architecture', 'optimizer', 'latent_activation'):
            with self.subTest(field=field):
                self.selection([35, 5, 2, 5, 35], {'name': 'adam', 'learning_rate': .003})
                manifest_path = self.root/'source/manifest.json'
                manifest = json.loads(manifest_path.read_text())
                if field == 'dataset':
                    manifest['dataset']['sha256'] = '0'*64
                else:
                    config_path = self.root/'source/config.json'
                    config = json.loads(config_path.read_text())
                    config[field] = {'architecture': [35, 7, 2, 7, 35],
                                     'optimizer': {'name': 'adam', 'learning_rate': .009},
                                     'latent_activation': 'linear'}[field]
                    write_json(config_path, config)
                    manifest['config'] = config
                    selection_path = self.root/'source/selection.json'
                    selection = json.loads(selection_path.read_text())
                    selection['configuration'] = config
                    write_json(selection_path, selection)
                write_json(manifest_path, manifest)
                output = self.root/field
                with self.assertRaises(ValueError):
                    prepare_campaign(self.root/'campaign.json', output, tp5_root=self.root)
                self.assertFalse(output.exists())

    def test_protocol_rejects_boolean_realization_count(self):
        self.selection([35, 5, 2, 5, 35], {'name': 'adam', 'learning_rate': .003})
        self.config['noise']['clean_realizations_per_character'] = True
        write_json(self.root/'campaign.json', self.config)
        with self.assertRaises(ValueError):
            prepare_campaign(self.root/'campaign.json', self.root/'invalid', tp5_root=self.root)

    def test_protocol_rejects_unapproved_training_budget(self):
        self.selection([35, 5, 2, 5, 35], {'name': 'adam', 'learning_rate': .003})
        self.config['max_epochs'] = 3001
        write_json(self.root/'campaign.json', self.config)
        with self.assertRaisesRegex(ValueError, 'protocol'):
            prepare_campaign(self.root/'campaign.json', self.root/'over-budget', tp5_root=self.root)
        self.assertFalse((self.root/'over-budget').exists())
