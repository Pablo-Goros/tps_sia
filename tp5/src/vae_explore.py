"""Figures and tables of exercise 2 from the saved VAE campaign (no training).

    python -m src.vae_explore --campaign results/vae --output results/vae/figures
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
from src.vae_train import run_name

SAMPLE_SEED = 2026
PRIOR_RADIUS = 2.45  # a 2-D standard normal keeps 95 % of its mass inside


def _show(ax, vector, title=None):
    ax.imshow(np.asarray(vector).reshape(28, 28), cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=7)


def _tile(images, columns):
    """Paste (N,784) images into one array, row by row."""
    rows = int(np.ceil(len(images) / columns))
    canvas = np.zeros((rows * 28, columns * 28))
    for index, image in enumerate(images):
        r, c = divmod(index, columns)
        canvas[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = image.reshape(28, 28)
    return canvas


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _label(kl_weight):
    return f'λ = {kl_weight:g}'


def dataset_figure(output):
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.2))
    for ax, name in zip(axes, ('digits', 'more_digits')):
        x, y = load_digits(name)
        picks = [x[np.flatnonzero(y == d)[:3]] if (y == d).any() else np.zeros((3, 784)) for d in range(10)]
        ax.imshow(np.hstack([_tile(p, 1) for p in picks]), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        counts = ', '.join(f'{d}:{int((y == d).sum())}' for d in range(10))
        ax.set_title(f'{name}.csv — {len(x)} imágenes 28×28 ({counts})', fontsize=8)
    _save(fig, output / 'dataset.png')


def curves_figure(campaign, config, output):
    datasets = config['datasets']
    fig, axes = plt.subplots(len(datasets), 2, figsize=(10, 3.6 * len(datasets)), squeeze=False)
    for row, dataset in enumerate(datasets):
        for kl_weight in config['kl_weights']:
            histories = [json.loads((campaign / run_name(dataset, kl_weight, s) / 'history.json').read_text())
                         for s in config['seeds']]
            epochs = [h['epoch'] for h in histories[0]]
            for col, key in enumerate(('test_reconstruction_bce', 'test_kl')):
                values = np.array([[h[key] for h in history] for history in histories])
                line, = axes[row, col].plot(epochs, values.mean(axis=0), label=_label(kl_weight))
                axes[row, col].fill_between(epochs, values.min(axis=0), values.max(axis=0),
                                            color=line.get_color(), alpha=0.2)
        axes[row, 0].set_ylabel(f'{dataset}\nBCE de reconstrucción (test)')
        axes[row, 1].set_ylabel('KL(q(z|x) || N(0,I)) (test)')
        axes[row, 1].set_yscale('log')
        for ax in axes[row]:
            ax.set_xlabel('época')
            ax.grid(alpha=0.3)
            ax.legend()
    fig.suptitle('Media de 3 semillas (banda: mínimo–máximo)', fontsize=10)
    _save(fig, output / 'curves.png')


def latent_figure(models, dataset, x_test, y_test, output):
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.8))
    angles = np.linspace(0, 2 * np.pi, 200)
    for ax, (kl_weight, model) in zip(axes, models.items()):
        mu, _ = model.encode(x_test)
        scatter = ax.scatter(mu[:, 0], mu[:, 1], c=y_test, cmap='tab10', vmin=-0.5, vmax=9.5, s=4)
        ax.plot(PRIOR_RADIUS * np.cos(angles), PRIOR_RADIUS * np.sin(angles), 'k--', lw=1,
                label='95 % de N(0,I)')
        ax.set_title(f'{_label(kl_weight)} — media μ(x) de test')
        ax.set_xlabel('z₁')
        ax.set_ylabel('z₂')
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(loc='upper right', fontsize=7)
    fig.colorbar(scatter, ax=axes, ticks=range(10), label='dígito', fraction=0.02)
    fig.suptitle(f'Espacio latente — entrenado con {dataset}.csv (semilla 42)')
    _save(fig, output / f'latent-{dataset}.png')


def grid_figure(models, dataset, output, size=15, limit=2.5):
    axis = np.linspace(-limit, limit, size)
    codes = np.array([[a, b] for b in axis[::-1] for a in axis])
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5.2))
    for ax, (kl_weight, model) in zip(axes, models.items()):
        ax.imshow(_tile(model.decode(codes), size), cmap='gray', vmin=0, vmax=1,
                  extent=(-limit, limit, -limit, limit))
        ax.set_title(_label(kl_weight))
        ax.set_xlabel('z₁')
        ax.set_ylabel('z₂')
    fig.suptitle(f'Decodificación de una grilla en [-{limit}, {limit}]² — {dataset}.csv (semilla 42)')
    _save(fig, output / f'grid-{dataset}.png')


def samples_figure(models, dataset, output, count=16):
    z = np.random.default_rng(SAMPLE_SEED).standard_normal((count, 2))
    fig, axes = plt.subplots(len(models), 1, figsize=(count * 0.6, 0.85 * len(models)))
    for ax, (kl_weight, model) in zip(axes, models.items()):
        ax.imshow(_tile(model.decode(z), count), cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(_label(kl_weight), rotation=0, ha='right', va='center', fontsize=8)
    fig.suptitle(f'Mismos 16 códigos z ~ N(0,I) decodificados — {dataset}.csv (semilla 42)', fontsize=9)
    _save(fig, output / f'samples-{dataset}.png')


def reconstruction_figure(all_models, x_test, y_test, output):
    picks = [int(np.flatnonzero(y_test == d)[0]) for d in range(10)]
    rows = [('original (test)', x_test[picks])]
    for dataset, models in all_models.items():
        for kl_weight, model in models.items():
            rows.append((f'{dataset}\n{_label(kl_weight)}', model.reconstruct(x_test[picks])))
    fig, axes = plt.subplots(len(rows), 1, figsize=(7.5, 0.85 * len(rows)))
    for ax, (name, images) in zip(axes, rows):
        ax.imshow(_tile(images, 10), cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha='right', va='center', fontsize=7)
    fig.suptitle('Reconstrucción por el código medio de dígitos no vistos (semilla 42)', fontsize=9)
    _save(fig, output / 'reconstructions.png')
    return picks


def extremes_figure(model, dataset, x_test, y_test, output):
    """Per digit: best, median and worst reconstructed test example (by BCE)."""
    _, mu, _ = model._encode(x_test)
    bce = model._bce_from_pre(x_test, model._decode(mu)[1])
    cases = (('mejor', 0.0), ('mediana', 0.5), ('peor', 1.0))
    record = {}
    fig, axes = plt.subplots(2 * len(cases), 10, figsize=(10, 2.25 * len(cases)))
    for digit in range(10):
        indices = np.flatnonzero(y_test == digit)
        ranked = indices[np.argsort(bce[indices])]
        for row, (name, quantile) in enumerate(cases):
            pick = int(ranked[int(round(quantile * (len(ranked) - 1)))])
            record.setdefault(name, []).append(pick)
            _show(axes[2 * row, digit], x_test[pick], f'{name} — original' if digit == 0 else None)
            _show(axes[2 * row + 1, digit], model.reconstruct(x_test[[pick]])[0], f'BCE {bce[pick]:.0f}')
    fig.suptitle(f'Reconstrucción mejor, mediana y peor de cada dígito de test — {dataset}.csv, '
                 f'λ = {model.kl_weight:g} (semilla {model.seed})', fontsize=9)
    fig.tight_layout()
    _save(fig, output / f'reconstructions-extremes-{dataset}.png')
    return record


def generated_figure(model, dataset, x_train, output, count=10):
    """Exercise 2.c: prior samples next to their closest training image."""
    z, images = model.sample(count, np.random.default_rng(SAMPLE_SEED + 1))
    distances = ((images ** 2).sum(axis=1)[:, None] - 2 * images @ x_train.T
                 + (x_train ** 2).sum(axis=1)[None, :])
    nearest = distances.argmin(axis=1)
    rms = np.sqrt(np.maximum(distances[np.arange(count), nearest], 0) / 784)
    fig, axes = plt.subplots(2, count, figsize=(count * 0.9, 2.4))
    for index in range(count):
        _show(axes[0, index], images[index], f'z=({z[index, 0]:.1f},{z[index, 1]:.1f})')
        _show(axes[1, index], x_train[nearest[index]], f'RMS {rms[index]:.2f}')
    fig.suptitle(f'Arriba: muestras nuevas z ~ N(0,I), λ = 1. Abajo: imagen de {dataset}.csv más cercana',
                 fontsize=9)
    _save(fig, output / f'generated-{dataset}.png')
    return {'codes': z.tolist(), 'nearest_train_index': nearest.tolist(), 'rms_distance': rms.tolist()}


def interpolation_figure(model, dataset, x_test, y_test, output, pairs=((0, 6), (3, 7), (1, 8)), steps=10):
    fig, axes = plt.subplots(len(pairs), 1, figsize=(steps * 0.75, 0.9 * len(pairs)))
    for ax, (a, b) in zip(axes, pairs):
        ends = x_test[[np.flatnonzero(y_test == a)[0], np.flatnonzero(y_test == b)[0]]]
        mu, _ = model.encode(ends)
        weights = np.linspace(0, 1, steps)[:, None]
        ax.imshow(_tile(model.decode((1 - weights) * mu[0] + weights * mu[1]), steps),
                  cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(f'{a} → {b}', rotation=0, ha='right', va='center', fontsize=8)
    fig.suptitle(f'Interpolación lineal entre códigos medios — {dataset}.csv, λ = 1 (semilla 42)', fontsize=9)
    _save(fig, output / f'interpolation-{dataset}.png')


def summary(campaign, config, x_test, y_test):
    """Mean and population std over seeds; BCE of the 8 against other digits."""
    rows = []
    eight = y_test == 8
    for dataset in config['datasets']:
        for kl_weight in config['kl_weights']:
            runs, bce8, bce_rest = [], [], []
            for seed in config['seeds']:
                folder = campaign / run_name(dataset, kl_weight, seed)
                runs.append(json.loads((folder / 'metrics.json').read_text())['test'])
                model = VAE.load(folder / 'model.npz')
                bce8.append(model.terms(x_test[eight])[0])
                bce_rest.append(model.terms(x_test[~eight])[0])
            stat = lambda values: [float(np.mean(values)), float(np.std(values))]
            rows.append({'dataset': dataset, 'kl_weight': kl_weight,
                         'test_bce': stat([r['reconstruction_bce'] for r in runs]),
                         'test_kl': stat([r['kl'] for r in runs]),
                         'mu_std': stat([np.mean(r['mu_std']) for r in runs]),
                         'sigma_mean': stat([r['sigma_mean'] for r in runs]),
                         'test_bce_digit_8': stat(bce8), 'test_bce_other_digits': stat(bce_rest)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--campaign', default='results/vae')
    parser.add_argument('--output', default='results/vae/figures')
    args = parser.parse_args()
    campaign, output = Path(args.campaign), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((campaign / 'protocol.json').read_text(encoding='utf-8'))
    if config['latent'] != 2:
        raise SystemExit('These figures require a two-dimensional code')
    seed = config['seeds'][0]
    x_test, y_test = load_digits('test')
    dataset_figure(output)
    curves_figure(campaign, config, output)
    all_models, generated, extremes = {}, {}, {}
    for dataset in config['datasets']:
        models = {w: VAE.load(campaign / run_name(dataset, w, seed) / 'model.npz') for w in config['kl_weights']}
        all_models[dataset] = models
        latent_figure(models, dataset, x_test, y_test, output)
        grid_figure(models, dataset, output)
        samples_figure(models, dataset, output)
        vae = models[max(config['kl_weights'])]
        generated[dataset] = generated_figure(vae, dataset, load_digits(dataset)[0], output)
        interpolation_figure(vae, dataset, x_test, y_test, output)
        extremes[dataset] = extremes_figure(vae, dataset, x_test, y_test, output)
    picks = reconstruction_figure(all_models, x_test, y_test, output)
    rows = summary(campaign, config, x_test, y_test)
    (output / 'summary.json').write_text(json.dumps(
        {'representative_seed': seed, 'sample_seed': SAMPLE_SEED, 'reconstruction_test_indices': picks,
         'extremes_test_indices': extremes, 'generated': generated, 'aggregates': rows}, indent=1), encoding='utf-8')
    print('| Entrenamiento | λ | BCE test | KL test | desvío de μ | σ media | BCE del 8 | BCE resto |')
    print('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for r in rows:
        cell = lambda key: f"{r[key][0]:.2f} ± {r[key][1]:.2f}"
        print(f"| {r['dataset']} | {r['kl_weight']:g} | {cell('test_bce')} | {cell('test_kl')} | "
              f"{cell('mu_std')} | {r['sigma_mean'][0]:.3f} | {cell('test_bce_digit_8')} | "
              f"{cell('test_bce_other_digits')} |")


if __name__ == '__main__':
    main()
