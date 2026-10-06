"""Exercise 3 protocol, references and immutable development partitions."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from tps_sia.tp3.shared.digit_dataset import cargar
from tps_sia.tp3.shared.experiments import (
    fingerprint, portable_path, sha256_file, validate_config, validate_data, write_json,
)
from tps_sia.tp3.shared.staged_search import materialize, protocol_sha256

TP3 = Path(__file__).resolve().parents[2]
CONFIG = TP3 / 'ej3/configs/search.json'
OUTPUT = TP3 / 'ej3/results/search'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}.')


def seeds(values, name):
    if not isinstance(values, list) or not values or len(set(values)) != len(values):
        raise ValueError(f'{name} must be a nonempty list of unique seeds.')
    for value in values:
        positive_integer(value, name, 0)


def load_protocol(path):
    p = read_json(path)
    required = {'schema_version', 'protocol', 'reference_selection', 'baseline', 'dataset',
                'cache', 'split_manifest', 'primary_seed', 'confirmation_seeds',
                'budget', 'ranking', 'stages', 'factor_study', 'scope'}
    if not isinstance(p, dict) or set(p) != required or p['schema_version'] != 1:
        raise ValueError('Invalid ej3 protocol fields or version.')
    if not isinstance(p['protocol'], str) or not p['protocol']:
        raise ValueError('A protocol name is required.')
    positive_integer(p['primary_seed'], 'primary_seed', 0)
    seeds(p['confirmation_seeds'], 'confirmation_seeds')
    if p['primary_seed'] not in p['confirmation_seeds']:
        raise ValueError('Confirmation must include the primary seed.')
    if len(p['confirmation_seeds']) < 3:
        raise ValueError('Confirmation requires at least three common seeds.')
    expected_ranking = ['validation accuracy descending', 'validation loss ascending',
                        'parameter count ascending', 'config_id ascending']
    if p['ranking'] != expected_ranking:
        raise ValueError('Unsupported ranking rule.')
    if set(p['budget']) != {'epochs', 'stopping', 'checkpoint_every'}:
        raise ValueError('Invalid training budget.')
    positive_integer(p['budget']['epochs'], 'epochs')
    positive_integer(p['budget']['checkpoint_every'], 'checkpoint_every')
    if p['budget']['stopping'].get('monitor') != 'validation_loss':
        raise ValueError('Checkpoints must be selected by validation_loss.')
    stage_keys = {'rates': 'learning_rates', 'architectures': 'architectures',
                  'batches': 'batch_sizes', 'confirmation': 'finalists'}
    if set(p['stages']) not in (set(stage_keys), set(stage_keys) | {'joint'}):
        raise ValueError('Invalid search stages.')
    for stage, key in stage_keys.items():
        spec = p['stages'][stage]
        if set(spec) != {key}:
            raise ValueError(f'Invalid {stage} settings.')
        if stage == 'confirmation':
            positive_integer(spec[key], 'finalists')
        elif not isinstance(spec[key], list) or not spec[key]:
            raise ValueError(f'{key} must be a nonempty list.')
    for rate in p['stages']['rates']['learning_rates']:
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not np.isfinite(rate) or rate <= 0:
            raise ValueError('Learning rates must be finite and positive.')
    for architecture in p['stages']['architectures']['architectures']:
        if (not isinstance(architecture, list) or len(architecture) < 3
                or architecture[0] != 784 or architecture[-1] != 10):
            raise ValueError('Architectures require 784 inputs, hidden layers and 10 outputs.')
        for width in architecture:
            positive_integer(width, 'layer width')
    for size in p['stages']['batches']['batch_sizes']:
        positive_integer(size, 'batch_size', 2)
    if 'joint' in p['stages']:
        joint = p['stages']['joint']
        required_joint = {'learning_rates', 'batch_sizes'}
        allowed_joint = required_joint | {'architecture', 'architectures', 'l2_values', 'translation_shifts'}
        if (not required_joint <= set(joint) <= allowed_joint
                or len({'architecture', 'architectures'} & set(joint)) != 1):
            raise ValueError('Invalid joint search settings.')
        variants = joint_architectures(joint)
        if not isinstance(variants, list) or not variants:
            raise ValueError('Joint architectures must be a nonempty list.')
        for architecture in variants:
            if (not isinstance(architecture, list) or len(architecture) < 3
                    or architecture[0] != 784 or architecture[-1] != 10):
                raise ValueError('Joint architecture requires 784 inputs and 10 outputs.')
            for width in architecture:
                positive_integer(width, 'joint layer width')
        if len({tuple(a) for a in variants}) != len(variants):
            raise ValueError('Joint architectures must be unique.')
        for key in ('learning_rates', 'batch_sizes'):
            values = joint[key]
            if not isinstance(values, list) or not values:
                raise ValueError(f'Joint {key} must be a nonempty list.')
            for value in values:
                if key == 'batch_sizes':
                    positive_integer(value, 'joint batch_size', 2)
                elif (isinstance(value, bool) or not isinstance(value, (int, float))
                      or not np.isfinite(value) or value <= 0):
                    raise ValueError('Joint learning rates must be finite and positive.')
            if len(set(values)) != len(values):
                raise ValueError(f'Joint {key} must be unique.')
        if 'l2_values' in joint:
            values = joint['l2_values']
            if (not isinstance(values, list) or not values
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not np.isfinite(v) or v < 0 for v in values)
                    or len(set(values)) != len(values) or 0 not in values):
                raise ValueError('L2 grid requires unique finite nonnegative values and a zero control.')
        if 'translation_shifts' in joint:
            values = joint['translation_shifts']
            if (not isinstance(values, list) or not values
                    or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < 28 for v in values)
                    or len(set(values)) != len(values) or 0 not in values):
                raise ValueError('Translation grid requires unique integer shifts 0..27 and a zero control.')
    factors = p['factor_study']
    if set(factors) != {'digits_dataset', 'digits_cache', 'subset_fractions', 'subset_seed', 'seeds'}:
        raise ValueError('Invalid factor study settings.')
    seeds(factors['seeds'], 'factor seeds')
    positive_integer(factors['subset_seed'], 'subset_seed', 0)
    fractions = factors['subset_fractions']
    if (not isinstance(fractions, list) or not fractions
            or any(isinstance(f, bool) or not isinstance(f, (int, float))
                   or not np.isfinite(f) or not 0 < f <= 1 for f in fractions)
            or fractions != sorted(set(fractions)) or fractions[-1] != 1.0):
        raise ValueError('Subset fractions must increase uniquely and finish at 1.0.')
    return p


def joint_architectures(spec):
    """Old single-architecture protocols retain their configuration and identities."""
    return spec['architectures'] if 'architectures' in spec else [spec['architecture']]


def bind_output(output, identity):
    """Changing protocol, sources or references requires another directory."""
    output = Path(output)
    path = output / 'protocol.json'
    if path.exists():
        if read_json(path) != identity:
            raise ValueError('Output belongs to a different protocol; choose a new --output-dir.')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Use an empty output directory or one with this protocol.json.')
        output.mkdir(parents=True, exist_ok=True)
        write_json(path, identity)


class Development:
    def __init__(self, config=CONFIG, reference_selection=None, root=TP3):
        self.root = Path(root).resolve()
        self.path = Path(config).resolve()
        self.protocol = load_protocol(self.path)
        p = self.protocol
        reference = (Path(reference_selection).resolve() if reference_selection is not None
                     else self.root / p['reference_selection'])
        if not reference.exists():
            raise FileNotFoundError(
                f'Ej2 selection not found: {reference}. Supply --reference-selection explicitly; '
                'v1 is not substituted automatically.')
        selection = read_json(reference)
        if not isinstance(selection.get('config'), dict):
            raise ValueError('The ej2 selection must contain its experiment config.')
        baseline_path = self.root / p['baseline']
        baseline = read_json(baseline_path)
        baseline.pop('sanity_accuracy', None)
        if isinstance(baseline.get('optimizer'), str):
            baseline['optimizer'] = {'name': baseline['optimizer'],
                                     'learning_rate': baseline.pop('learning_rate')}
        self.dataset = (self.root / p['dataset']).resolve()
        self.cache = (self.root / p['cache']).resolve()
        # Reject reserved paths before loading any source.
        validate_config({'dataset': str(self.dataset), 'cache': str(self.cache)})
        manifest_path = self.root / p['split_manifest']
        manifest = read_json(manifest_path)
        if (self.root / manifest['dataset']['path']).resolve() != self.dataset:
            raise ValueError('Split manifest refers to a different dataset.')
        digest = sha256_file(self.dataset)
        if digest != manifest['dataset']['sha256']:
            raise ValueError('CSV differs from split manifest. Regenerate the development split first.')
        split_path = manifest_path.parent / manifest['indices_file']
        with np.load(split_path, allow_pickle=False) as split:
            train, validation, excluded = (split[k] for k in ('train', 'validation', 'excluded'))
        for values in (train, validation, excluded):
            if values.ndim != 1 or values.dtype.kind not in 'iu':
                raise ValueError('Split indices must be one-dimensional integer arrays.')
        self.X, self.y = cargar(self.dataset, self.cache)
        if sha256_file(self.dataset) != digest:
            raise ValueError('Dataset changed during loading.')
        joined = np.concatenate((train, validation, excluded))
        if (len(joined) != len(self.X) or len(np.unique(joined)) != len(self.X)
                or np.any(joined < 0) or np.any(joined >= len(self.X))):
            raise ValueError('Split must cover every source row exactly once.')
        if (len(self.X) != manifest['dataset']['rows'] or len(train) != manifest['train_size']
                or len(validation) != manifest['validation_size']):
            raise ValueError('Split sizes differ from the manifest.')
        split_digest = fingerprint({'train': train.tolist(), 'validation': validation.tolist()})
        if split_digest != manifest['split_sha256']:
            raise ValueError('Split indices differ from the manifest fingerprint.')
        self.train, self.validation = train, validation
        self.data = {'train_indices': train.tolist(), 'validation_indices': validation.tolist(),
                     'validation_dataset': str(self.dataset), 'validation_cache': str(self.cache),
                     'dataset_sha256': digest, 'validation_sha256': digest}
        validate_data(self.data, str(self.dataset), str(self.cache))
        self.baseline = self.config(baseline)
        self.reference = self.config(selection['config'])
        # Validate every variant before the first experiment, not halfway through a stage.
        for rate in p['stages']['rates']['learning_rates']:
            self.config(self.reference, optimizer={**self.reference['optimizer'], 'learning_rate': rate})
        for architecture in p['stages']['architectures']['architectures']:
            self.config(self.reference, architecture=architecture)
        if 'joint' in p['stages']:
            spec = p['stages']['joint']
            for rate in spec['learning_rates']:
                for size in spec['batch_sizes']:
                    for strength in spec.get('l2_values', [0]):
                        for shift in spec.get('translation_shifts', [0]):
                            for architecture in joint_architectures(spec):
                                self.config(self.reference, architecture=architecture,
                                            strategy='mini_batch', batch_size=size,
                                            optimizer={**self.reference['optimizer'], 'learning_rate': rate},
                                            **({'l2': strength} if strength else {}),
                                            **({'augmentation': {'name': 'translation', 'max_shift': shift}}
                                               if shift else {}))
        self.identity = {'schema_version': 1, 'protocol': p['protocol'],
                         'primary_seed': p['primary_seed'],
                         'search_sha256': protocol_sha256(self.path),
                         'reference_selection': portable_path(reference, self.root),
                         'reference_sha256': sha256_file(reference),
                         'baseline_sha256': sha256_file(baseline_path),
                         'dataset_sha256': digest, 'split_sha256': split_digest,
                         'split_manifest_sha256': sha256_file(manifest_path)}

    def config(self, source, seed=None, **changes):
        config = copy.deepcopy(source)
        # Old selections may store absolute paths from another operating system.
        config.update(dataset=str(self.dataset), cache=str(self.cache), data=copy.deepcopy(self.data))
        config.update(copy.deepcopy(self.protocol['budget']))
        config.update(copy.deepcopy(changes))
        config = materialize(config, self.root,
                             self.protocol['primary_seed'] if seed is None else seed)
        return validate_config(config)
