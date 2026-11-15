"""Latent-width study of exercise 2: same VAE, wider codes (no 2-D plots).

    python -m src.vae_latent train --latent 16 --seed 42
    python -m src.vae_latent report

Every run reuses the protocol of configs/vae.json (more_digits, KL weight 1)
changing only the code width. Two-dimensional runs come from results/vae.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.digits import load_digits
from src.vae import VAE
from src.vae_explore import SAMPLE_SEED, _save, _show, _tile
from src.vae_train import load_config, run_name, train_run

DATASET, KL_WEIGHT = 'more_digits', 1.0
STUDY = {4: [42], 8: [42], 16: [42, 43, 44], 32: [42]}  # 2 -> seeds of results/vae
ACTIVE_NATS = 0.01


def _folder(output, latent, seed):
    return Path(output) / f'latent-{latent}-seed-{seed}'


def train(args):
    config = dict(load_config(args.config), latent=args.latent)
    folder = _folder(args.output, args.latent, args.seed)
    if (folder / 'metrics.json').exists():
        print('skip', folder.name)
        return
    metrics = train_run(config, DATASET, KL_WEIGHT, args.seed, load_digits(DATASET)[0],
                        load_digits('test')[0], folder)
    print(f"done {folder.name} test_bce={metrics['test']['reconstruction_bce']:.2f} "
          f"test_kl={metrics['test']['kl']:.2f} {metrics['seconds']:.0f}s")


def _measure(model, x_test, y_test, centroids):
    recon = model.reconstruct(x_test)
    mu, logvar = model.encode(x_test)
    kl_per_dim = (0.5 * (np.exp(logvar) + mu ** 2 - 1.0 - logvar)).mean(axis=0)
    nearest = ((recon[:, None, :] - centroids[None]) ** 2).sum(axis=2).argmin(axis=1)
    bce, kl = model.terms(x_test)
    return {'test_bce': bce, 'test_kl': kl, 'mse_per_pixel': float(((recon - x_test) ** 2).mean()),
            'recognizable': float((nearest == y_test).mean()),
            'active_dimensions': int((kl_per_dim > ACTIVE_NATS).sum()),
            'kl_per_dimension': np.sort(kl_per_dim)[::-1].tolist()}


def report(args):
    output = Path(args.output)
    figures = output / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    x_train, y_train = load_digits(DATASET)
    x_test, y_test = load_digits('test')
    centroids = np.array([x_train[y_train == d].mean(axis=0) for d in range(10)])
    campaign = Path(args.campaign)
    seeds_2d = json.loads((campaign / 'protocol.json').read_text(encoding='utf-8'))['seeds']
    paths = {2: [campaign / run_name(DATASET, KL_WEIGHT, s) / 'model.npz' for s in seeds_2d]}
    paths.update({k: [_folder(output, k, s) / 'model.npz' for s in seeds] for k, seeds in STUDY.items()})
    rows, models = [], {}
    for latent in sorted(paths):
        runs = []
        for path in paths[latent]:
            model = VAE.load(path)
            models.setdefault(latent, model)  # first seed is the representative
            runs.append(_measure(model, x_test, y_test, centroids))
        stat = lambda key: [float(np.mean([r[key] for r in runs])), float(np.std([r[key] for r in runs]))]
        rows.append({'latent': latent, 'seeds': len(runs), 'test_bce': stat('test_bce'),
                     'test_kl': stat('test_kl'), 'mse_per_pixel': stat('mse_per_pixel'),
                     'recognizable': stat('recognizable'),
                     'active_dimensions': [r['active_dimensions'] for r in runs],
                     'kl_per_dimension_first_seed': runs[0]['kl_per_dimension']})
    originals = float((((x_test[:, None, :] - centroids[None]) ** 2).sum(axis=2).argmin(axis=1) == y_test).mean())

    # Reconstruction error and negative ELBO against the code width.
    widths = [r['latent'] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    bce = np.array([r['test_bce'][0] for r in rows])
    kl = np.array([r['test_kl'][0] for r in rows])
    axes[0].plot(widths, bce, 'o-', label='BCE de reconstrucción')
    axes[0].plot(widths, bce + kl, 's--', label='BCE + KL (−ELBO)')
    axes[0].set_ylabel('nats por imagen (test)')
    axes[1].plot(widths, [100 * r['recognizable'][0] for r in rows], 'o-', label='reconstrucciones')
    axes[1].axhline(100 * originals, color='gray', ls=':', label='imágenes originales')
    axes[1].set_ylabel('% con centroide más cercano correcto')
    for ax in axes:
        ax.set_xscale('log', base=2)
        ax.set_xticks(widths)
        ax.set_xticklabels(widths)
        ax.set_xlabel('dimensión del latente')
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f'{DATASET}.csv, λ = 1 (media de semillas; 4, 8 y 32 con una sola semilla)', fontsize=9)
    _save(fig, figures / 'latent-width.png')

    # Same test examples through the 2-D and 16-D representatives.
    extremes = json.loads((campaign / 'figures' / 'summary.json').read_text())['extremes_test_indices'][DATASET]
    fig, axes = plt.subplots(6, 10, figsize=(10, 6.6))
    for block, case in enumerate(('mediana', 'peor')):
        for column, index in enumerate(extremes[case]):
            images = ((x_test[index], f'{case} — original'),
                      (models[2].reconstruct(x_test[[index]])[0], 'latente 2D'),
                      (models[16].reconstruct(x_test[[index]])[0], 'latente 16D'))
            for offset, (image, title) in enumerate(images):
                ax = axes[3 * block + offset, column]
                _show(ax, image)
                if column == 0:
                    ax.set_title(title, fontsize=8, loc='left')
    fig.suptitle(f'Mismos ejemplos de test: latente 2D vs 16D ({DATASET}, λ = 1, semilla 42)', fontsize=9)
    fig.tight_layout()
    _save(fig, figures / 'reconstructions-2d-vs-16d.png')

    # Prior samples for every width, and 16-D samples next to their nearest training image.
    rng = np.random.default_rng(SAMPLE_SEED)
    fig, axes = plt.subplots(len(models), 1, figsize=(9.6, 0.85 * len(models)))
    for ax, (latent, model) in zip(axes, sorted(models.items())):
        ax.imshow(_tile(model.sample(16, rng)[1], 16), cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(f'{latent}D', rotation=0, ha='right', va='center', fontsize=8)
    fig.suptitle(f'16 muestras z ~ N(0,I) por dimensión del latente ({DATASET}, λ = 1, semilla 42)', fontsize=9)
    _save(fig, figures / 'samples-by-width.png')

    _, images = models[16].sample(10, np.random.default_rng(SAMPLE_SEED + 1))
    distances = ((images ** 2).sum(axis=1)[:, None] - 2 * images @ x_train.T + (x_train ** 2).sum(axis=1)[None, :])
    nearest = distances.argmin(axis=1)
    rms = np.sqrt(np.maximum(distances[np.arange(10), nearest], 0) / 784)
    fig, axes = plt.subplots(2, 10, figsize=(9, 2.3))
    for index in range(10):
        _show(axes[0, index], images[index])
        _show(axes[1, index], x_train[nearest[index]], f'RMS {rms[index]:.2f}')
    fig.suptitle(f'Arriba: muestras nuevas del VAE 16D. Abajo: imagen de {DATASET}.csv más cercana', fontsize=9)
    _save(fig, figures / 'generated-16d.png')

    (figures / 'summary.json').write_text(json.dumps(
        {'dataset': DATASET, 'kl_weight': KL_WEIGHT, 'active_threshold_nats': ACTIVE_NATS,
         'originals_recognizable': originals, 'generated_16d_rms': rms.tolist(),
         'generated_16d_nearest_train_index': nearest.tolist(), 'rows': rows}, indent=1), encoding='utf-8')
    print(f'originales reconocibles: {100 * originals:.1f} %')
    print('| Latente | Semillas | BCE test | KL test | BCE + KL | MSE/píxel | Reconocibles | Dimensiones activas |')
    print('| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |')
    for r in rows:
        spread = lambda key, f: f"{f % r[key][0]}" + (f" ± {f % r[key][1]}" if r['seeds'] > 1 else '')
        print(f"| {r['latent']} | {r['seeds']} | {spread('test_bce', '%.1f')} | {spread('test_kl', '%.1f')} | "
              f"{r['test_bce'][0] + r['test_kl'][0]:.1f} | {r['mse_per_pixel'][0]:.4f} | "
              f"{100 * r['recognizable'][0]:.1f} % | {', '.join(map(str, r['active_dimensions']))} de {r['latent']} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'report'):
        command = sub.add_parser(name)
        command.add_argument('--output', default='results/vae-latent')
        command.add_argument('--config', default='configs/vae.json')
        command.add_argument('--campaign', default='results/vae')
        if name == 'train':
            command.add_argument('--latent', type=int, required=True)
            command.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    (train if args.command == 'train' else report)(args)


if __name__ == '__main__':
    main()
