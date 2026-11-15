"""Train the VAE campaign of exercise 2: datasets x KL weights x seeds.

    python -m src.vae_train --config configs/vae.json --output results/vae

Completed runs (those with metrics.json) are skipped, so the command can be
repeated after an interruption; an interrupted run restarts from scratch.
``--max-runs N`` trains at most N pending runs per invocation and ``--seed S``
restricts it to one seed, so seeds can be trained by parallel processes.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from src.digits import load_digits
from src.vae import VAE

KEYS = {'datasets', 'encoder', 'latent', 'optimizer', 'batch_size', 'epochs', 'kl_weights', 'seeds'}


def load_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if set(config) != KEYS:
        raise ValueError(f'Configuration must contain exactly {sorted(KEYS)}')
    if type(config['epochs']) is not int or config['epochs'] < 1:
        raise ValueError('epochs must be a positive integer')
    return config


def run_name(dataset, kl_weight, seed):
    return f"{dataset}-lambda-{str(float(kl_weight)).replace('.', 'p')}-seed-{seed}"


def evaluate(model, x):
    """Deterministic metrics through the mean code."""
    recon, kl = model.terms(x)
    mu, logvar = model.encode(x)
    mse = float(np.mean((model.reconstruct(x) - x) ** 2))
    return {'reconstruction_bce': recon, 'kl': kl, 'mse_per_pixel': mse,
            'mu_mean': mu.mean(axis=0).tolist(), 'mu_std': mu.std(axis=0).tolist(),
            'sigma_mean': float(np.exp(0.5 * logvar).mean())}


def train_run(config, dataset, kl_weight, seed, x_train, x_test, folder):
    folder.mkdir(parents=True, exist_ok=True)
    model = VAE(encoder=config['encoder'], latent=config['latent'], kl_weight=kl_weight,
                seed=seed, optimizer=config['optimizer'], batch_size=config['batch_size'])
    started = time.time()
    for epoch in range(1, config['epochs'] + 1):
        recon, kl = model.train_epoch(x_train)
        test_recon, test_kl = model.terms(x_test)
        model.history.append({'epoch': epoch, 'train_reconstruction_bce': recon, 'train_kl': kl,
                              'test_reconstruction_bce': test_recon, 'test_kl': test_kl})
    metrics = {'dataset': dataset, 'kl_weight': float(kl_weight), 'seed': seed,
               'epochs': config['epochs'], 'seconds': time.time() - started,
               'train_samples': int(x_train.shape[0]), 'test_samples': int(x_test.shape[0]),
               'numpy': np.__version__,
               'train': evaluate(model, x_train), 'test': evaluate(model, x_test)}
    model.save(folder / 'model.npz')
    (folder / 'history.json').write_text(json.dumps(model.history, indent=1), encoding='utf-8')
    (folder / 'config.json').write_text(json.dumps(model.configuration, indent=1), encoding='utf-8')
    (folder / 'metrics.json').write_text(json.dumps(metrics, indent=1), encoding='utf-8')
    return metrics


def write_summary(output, rows):
    fields = ['dataset', 'kl_weight', 'seed', 'epochs', 'train_bce', 'train_kl',
              'test_bce', 'test_kl', 'test_mse_per_pixel', 'test_sigma_mean', 'seconds']
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for m in rows:
            writer.writerow({'dataset': m['dataset'], 'kl_weight': m['kl_weight'], 'seed': m['seed'],
                             'epochs': m['epochs'],
                             'train_bce': m['train']['reconstruction_bce'], 'train_kl': m['train']['kl'],
                             'test_bce': m['test']['reconstruction_bce'], 'test_kl': m['test']['kl'],
                             'test_mse_per_pixel': m['test']['mse_per_pixel'],
                             'test_sigma_mean': m['test']['sigma_mean'], 'seconds': m['seconds']})


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', default='configs/vae.json')
    parser.add_argument('--output', default='results/vae')
    parser.add_argument('--max-runs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    stored = output / 'protocol.json'
    if stored.exists() and json.loads(stored.read_text(encoding='utf-8')) != config:
        raise SystemExit('Output holds a different protocol; use a new directory')
    stored.write_text(json.dumps(config, indent=1), encoding='utf-8')
    x_test, _ = load_digits('test')
    rows, trained, complete = [], 0, True
    for dataset in config['datasets']:
        x_train, _ = load_digits(dataset)
        for kl_weight in config['kl_weights']:
            for seed in config['seeds']:
                folder = output / run_name(dataset, kl_weight, seed)
                if args.seed is not None and seed != args.seed:
                    complete = complete and (folder / 'metrics.json').exists()
                    continue
                if (folder / 'metrics.json').exists():
                    metrics = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
                    print('skip', folder.name, flush=True)
                elif args.max_runs is not None and trained >= args.max_runs:
                    print('pending runs remain; repeat the command to continue', flush=True)
                    return
                else:
                    trained += 1
                    metrics = train_run(config, dataset, kl_weight, seed, x_train, x_test, folder)
                    print(f"done {folder.name} test_bce={metrics['test']['reconstruction_bce']:.2f} "
                          f"test_kl={metrics['test']['kl']:.2f} {metrics['seconds']:.0f}s", flush=True)
                rows.append(metrics)
    if not complete:
        print('runs of other seeds are pending', flush=True)
        return
    if args.seed is not None:  # collect every run, not only this seed's
        rows = [json.loads((output / run_name(d, w, s) / 'metrics.json').read_text(encoding='utf-8'))
                for d in config['datasets'] for w in config['kl_weights'] for s in config['seeds']]
    write_summary(output, rows)
    print('campaign complete', flush=True)


if __name__ == '__main__':
    main()
