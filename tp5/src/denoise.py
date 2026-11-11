"""Prepare an immutable reference for the TP5 denoising campaign."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from .autoencoder import Autoencoder
from .dataset import DEFAULT_DATASET, load_dataset
from .train import write_json

TP5_ROOT = Path(__file__).resolve().parents[1]
# The approved experimental contract is independent of user-editable input.
APPROVED_PROTOCOL = {
    'seeds': [42, 43, 44], 'batch_size': 32, 'max_epochs': 3000,
    'evaluate_every': 100, 'stop_on_tolerance': False,
    'noise': {'type': 'bit_flip', 'training_levels': [0, .05, .10, .20],
              'evaluation_levels': [0, .05, .10, .20, .30],
              'selection_levels': [.05, .10, .20],
              'validation_seed': 20261110, 'evaluation_seed': 20261111,
              'realizations_per_character': 100, 'clean_realizations_per_character': 1}}



def _read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare_campaign(config_path, output, dataset_path=DEFAULT_DATASET, *, tp5_root=TP5_ROOT):
    """Validate a selected AE and snapshot it; no noise generation or training.

    Relative source_selection resolves from tp5_root. Selection paths resolve
    from paths_relative_to, itself relative to tp5_root (legacy default: '.').
    All paths in the resulting campaign manifest are relative to the campaign.
    """
    config = _read_json(Path(config_path))
    if (not isinstance(config, dict) or set(config) != set(APPROVED_PROTOCOL) | {'source_selection'}
            or not isinstance(config.get('source_selection'), str)
            or not config['source_selection'].strip()
            or json.dumps({k: v for k, v in config.items() if k != 'source_selection'}, sort_keys=True) !=
               json.dumps(APPROVED_PROTOCOL, sort_keys=True)):
        raise ValueError('Campaign must follow the approved B1 protocol; only source_selection may vary')
    output = Path(output)
    if output.exists():
        raise ValueError('Campaign output must be new; refusing to overwrite an existing directory')
    root = Path(tp5_root).resolve()
    selection_path = root/config['source_selection']
    selection = _read_json(selection_path)
    base = selection.get('paths_relative_to', '.')
    if not isinstance(base, str) or not base.strip():
        raise ValueError('paths_relative_to must be a nonempty source TP5 directory')
    base = root/base
    model_path = base/selection['model']['path']
    source_config_path = base/selection['source_config']
    manifest_path = source_config_path.parent/'manifest.json'
    source_config = _read_json(source_config_path)
    source_manifest = _read_json(manifest_path)
    dataset = load_dataset(dataset_path)
    files = {'selection.json': selection_path.read_bytes(),
             'config.json': source_config_path.read_bytes(),
             'manifest.json': manifest_path.read_bytes(),
             'model.npz': model_path.read_bytes(),
             'font.h': Path(dataset.path).read_bytes()}
    digest = _sha256(files['model.npz'])
    if digest != selection['model']['sha256'] or digest != source_manifest['model']['sha256']:
        raise ValueError('Selected model SHA256 disagrees with selection or source manifest')
    if dataset.sha256 != source_manifest['dataset']['sha256']:
        raise ValueError('Dataset SHA256 disagrees with the selected training dataset')
    if source_config != selection.get('configuration', source_config) or source_config != source_manifest['config']:
        raise ValueError('Selected configuration disagrees with source config or manifest')

    # Load and verify the captured bytes, so the validated model is exactly the
    # one published even if the source changes during preparation.
    with tempfile.TemporaryDirectory() as temporary:
        captured = Path(temporary)/'model.npz'
        captured.write_bytes(files['model.npz'])
        model = Autoencoder.load(captured)
    expected_model = Autoencoder(architecture=source_config['architecture'],
                                seed=source_config['seed'], optimizer=source_config['optimizer'],
                                batch_size=source_config['batch_size'],
                                loss=source_config.get('loss', 'half_squared_error'),
                                latent_activation=source_config.get('latent_activation', 'tanh'),
                                lr_scheduler=source_config.get('lr_scheduler'))
    effective = model.configuration
    if effective != expected_model.configuration or effective != source_manifest['effective_model']:
        raise ValueError('Selected model configuration disagrees with source configuration or manifest')
    if (effective['latent_activation'] != 'tanh' or effective['lr_scheduler'] is not None
            or effective['l2'] != 0 or effective['augmentation'] is not None
            or effective['batch_norm'] is not None or effective['beta'] != 1):
        raise ValueError('Source incompatible: require tanh code, constant rate and no extra regularization')
    winner = {key: source_config[key] for key in ('architecture', 'optimizer')}
    winner.update(loss=effective['loss'], latent_activation='tanh', lr_scheduler=None,
                  batch_size=config['batch_size'])
    alternative = {**winner, 'architecture': list(winner['architecture'])}
    alternative['architecture'][len(alternative['architecture'])//2] = 10
    if alternative == winner:
        raise ValueError('Variants must differ in central width')
    result = {'schema_version': 1, 'status': 'prepared', 'paths_relative_to': '.',
              'protocol': config, 'variants': {'winner': winner, 'latent-10': alternative},
              'reference': {name: {'path': f'reference/{name}', 'sha256': _sha256(raw)}
                            for name, raw in files.items()},
              'dataset': {'path': 'reference/font.h', 'sha256': dataset.sha256},
              'source': {'selection': str(selection_path.resolve()), 'tp5_root': str(base.resolve())},
              'budget': {'runs': 6, 'max_total_epochs': 18000, 'epochs_completed': 0}}
    # Stage on the destination filesystem and publish only a complete campaign.
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.denoising-', dir=output.parent))
    try:
        (staging/'reference').mkdir()
        for name, raw in files.items():
            (staging/'reference'/name).write_bytes(raw)
        write_json(staging/'config.json', config)
        write_json(staging/'manifest.json', result)
        if output.exists():
            raise ValueError('Campaign output appeared during preparation')
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--config', required=True, type=Path)
    prepare.add_argument('--output', required=True, type=Path)
    prepare.add_argument('--dataset', default=DEFAULT_DATASET, type=Path)
    args = parser.parse_args()
    try:
        result = prepare_campaign(args.config, args.output, args.dataset)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
