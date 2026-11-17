"""Exploratory extension of the 1.b campaign: continue the six saved runs.

    python -m src.denoise_extension run --campaign results/denoising --output results/denoising-extended-6000
    python -m src.denoise_extension figures --extension results/denoising-extended-6000 --campaign results/denoising

Each run restarts from its saved last state (epoch = the approved budget) and
trains up to a larger budget with the same noise sequence, optimizer, batch and
validation collection. This is NOT part of the approved protocol: the published
campaign is only read (its SHA256 hashes are checked before and after) and every
output goes to a new directory.
"""
import argparse
import csv
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .autoencoder import Autoencoder
from .dataset import load_dataset
from .denoise_campaign import load_collection, read_json, verified_campaign
from .denoise_evaluation import grouped_metrics, write_csv
from .denoise_training import select_runs, validation_score
from .noise import training_inputs
from .train import write_json

DEFAULT_END = 6000
METRICS = ('mean_pixel_errors', 'exact_fraction', 'within_tolerance_fraction')


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Derived documentation of the campaign (report and figures) may be edited at any time.
DOCUMENTATION = ('report.md', 'figures')


def tree_hashes(root, exclude=()):
    """SHA256 of every file below ``root`` (minus top-level ``exclude`` names), keyed by relative POSIX path."""
    root = Path(root)
    return {p.relative_to(root).as_posix(): _sha256(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.relative_to(root).parts[0] not in exclude}


def protocol_text(start, end):
    return f"""# Extensión exploratoria de 1.b: de {start} a {end} épocas

Estado: exploratorio. NO forma parte del protocolo aprobado. Los resultados publicados
(`{start}` épocas) en `results/denoising/` siguen siendo los principales.

Regla fijada ANTES de ejecutar:

- Se parte del último estado guardado (`last_model.npz`, época {start}) de las seis corridas y se
  continúa hasta la época {end}, con el ruido de la época absoluta (semilla, época), como en el
  protocolo original. Mismos niveles de entrenamiento, mismo lote, mismo optimizador.
- Validación cada 100 épocas con la misma colección y los mismos niveles de selección. El mejor
  checkpoint de cada corrida es el de menor puntaje en TODO el rango 0–{end} (si ninguna época
  nueva supera al mejor de {start}, se conserva el de {start}).
- Variante ganadora y corrida representativa: misma regla que el protocolo original.
- La colección final se evalúa una sola vez, después de fijar la selección. No se ajusta nada con
  ella. Se informa el resultado sea cual sea.
- Los archivos de `results/denoising/` no se modifican (se verifica por SHA256 antes y después).
- Entorno de esta extensión: Python {platform.python_version()} / NumPy {np.__version__}. La
  numérica de las épocas nuevas puede diferir levemente de la que daría otro entorno; el arranque
  es idéntico porque parte de los mismos estados publicados.
"""


def _mean_range(rows, variant, key, level):
    values = [float(row[key]) for row in rows if row['model'].startswith(variant + '-seed-')
              and row['character_index'] in (None, '') and abs(float(row['level']) - level) < 1e-9]
    if len(values) != 3:
        raise ValueError(f'Expected three seeds for {variant} at level {level}')
    return float(np.mean(values)), float(np.min(values)), float(np.max(values))


def extend_campaign(campaign, output, end_epoch=DEFAULT_END, log=print):
    """Continue the saved runs of ``campaign`` up to ``end_epoch`` in a new ``output`` directory."""
    campaign, output = Path(campaign), Path(output)
    manifest = verified_campaign(campaign)
    protocol = manifest['protocol']
    start, every = protocol['max_epochs'], protocol['evaluate_every']
    if type(end_epoch) is not int or end_epoch <= start or (end_epoch - start) % every:
        raise ValueError(f'end_epoch must exceed {start} in steps of {every}')
    ledger = read_json(campaign/'training.json')
    variants, seeds = list(manifest['variants']), protocol['seeds']
    if ledger['status'] != 'completed' or ledger['epochs_completed'] != len(variants)*len(seeds)*start:
        raise ValueError('The extension requires the completed approved campaign')
    if output.exists():
        raise ValueError('Extension output must be new; refusing to overwrite an existing directory')

    before = tree_hashes(campaign, DOCUMENTATION)
    output.mkdir(parents=True)
    (output/'PROTOCOL.md').write_text(protocol_text(start, end_epoch), encoding='utf-8')
    write_json(output/'campaign-hashes-before.json', before)
    stream = (output/'run.log').open('w', encoding='utf-8')

    def say(text):
        log(text)
        stream.write(text + '\n')
        stream.flush()

    noise = protocol['noise']
    levels, selection_levels = noise['training_levels'], noise['selection_levels']
    clean = load_dataset(campaign/manifest['dataset']['path']).vectors
    validation = load_collection(campaign, 'validation')
    keep = validation['levels'] <= max(levels) + 1e-12      # the out-of-range level never enters validation
    validation = {key: value[keep] for key, value in validation.items()}
    final = load_collection(campaign, 'final')
    say(f'validation images: {len(validation["inputs"])} | final images: {len(final["inputs"])} | '
        f'python {platform.python_version()} numpy {np.__version__}')

    def score(model):
        rows = grouped_metrics(validation, model.reconstruct(validation['inputs']))
        return validation_score(rows, selection_levels), rows

    results, histories, started = [], {}, time.perf_counter()
    for variant in variants:
        for seed in seeds:
            run_id = f'{variant}-seed-{seed}'
            source = campaign/'runs'/run_id
            run_manifest = read_json(source/'manifest.json')
            if run_manifest['status'] != 'completed' or run_manifest['epochs_completed'] != start:
                raise ValueError(f'{run_id} is not a completed {start}-epoch run')
            history = read_json(source/'history.json')['validation']
            model = Autoencoder.load(source/'last_model.npz')
            if model.history['epocas_corridas'] != start:
                raise ValueError(f'{run_id}: saved state is not at epoch {start}')
            reloaded, _ = score(model)
            if not np.allclose(reloaded, history[-1]['score'], atol=1e-6):
                raise ValueError(f'{run_id}: reloaded state does not reproduce its stored validation')
            run_dir = output/'runs'/run_id
            run_dir.mkdir(parents=True)
            shutil.copy(source/'best_model.npz', run_dir/'best_model.npz')   # kept unless beaten
            best = {'score': list(run_manifest['best']['score']), 'epoch': run_manifest['best']['epoch']}
            original_best, history, t0 = dict(best), list(history), time.perf_counter()
            for epoch in range(start + 1, end_epoch + 1):
                model.train(training_inputs(clean, levels, seed, epoch), 1, targets=clean)
                if model.history['stop_reason'] in ('interrupted', 'numerical_failure'):
                    raise RuntimeError(f'{run_id} stopped: {model.history["stop_reason"]}')
                if epoch % every == 0:
                    current, rows = score(model)
                    history.append({'epoch': epoch, 'score': current, 'levels': rows})
                    if (current, epoch) < (best['score'], best['epoch']):
                        model.save(run_dir/'best_model.npz')
                        best = {'score': current, 'epoch': epoch}
            model.save(run_dir/'last_model.npz')
            write_json(run_dir/'history.json', {'validation': history})
            histories[run_id] = history
            results.append({'run_id': run_id, 'variant': variant, 'seed': seed, 'status': 'completed',
                            'epochs_completed': model.history['epocas_corridas'], 'best': best,
                            f'best_{start}': original_best})
            say(f'{run_id}: best@{start} epoch {original_best["epoch"]} score {original_best["score"][0]:.4f} -> '
                f'best@{end_epoch} epoch {best["epoch"]} score {best["score"][0]:.4f} | {time.perf_counter() - t0:.0f} s')

    selection = select_runs(results, variants, seeds)
    selection['runs'] = results
    write_json(output/f'selection-{end_epoch}.json', selection)
    say(f'selection at {end_epoch}: {selection["variant"]} / {selection["run_id"]}; means: {selection["aggregates"]}')

    # The final collection is evaluated once, after the selection above was fixed.
    extended_rows = []
    for run in results:
        model = Autoencoder.load(output/'runs'/run['run_id']/'best_model.npz')
        extended_rows.extend({'model': run['run_id'], **row}
                             for row in grouped_metrics(final, model.reconstruct(final['inputs'])))
    write_csv(output/f'by-level-{end_epoch}.csv', extended_rows)
    with (campaign/'evaluation/by-level.csv').open(encoding='utf-8') as handle:
        published = [row for row in csv.DictReader(handle) if row['character_index'] == '']
    extended_rows = [{**row, 'character_index': None} for row in extended_rows]
    table = []
    for variant in variants:
        for level in noise['evaluation_levels']:
            for key in METRICS:
                a = _mean_range(published, variant, key, level)
                b = _mean_range(extended_rows, variant, key, level)
                table.append({'variant': variant, 'level': level, 'metric': key,
                              f'mean_{start}': a[0], f'min_{start}': a[1], f'max_{start}': a[2],
                              f'mean_{end_epoch}': b[0], f'min_{end_epoch}': b[1], f'max_{end_epoch}': b[2]})
    write_csv(output/f'comparison-{start}-vs-{end_epoch}.csv', table)

    after = tree_hashes(campaign, DOCUMENTATION)
    if after != before:
        raise RuntimeError('The published campaign changed during the extension')
    write_json(output/'campaign-hashes-after.json', after)
    say(f'published campaign unchanged ({len(before)} files verified by SHA256) | '
        f'total time: {time.perf_counter() - started:.0f} s')
    info = {'start_epoch': start, 'end_epoch': end_epoch, 'variants': variants, 'seeds': seeds,
            'selected_run': selection['run_id'], 'published_files_verified': len(before),
            'versions': {'python': platform.python_version(), 'numpy': np.__version__}}
    write_json(output/'extension.json', info)
    stream.close()
    return info


# ---------------------------------------------------------------- figures
TEAL, BLUE, GRAY, INK = '#008f86', '#368dc4', '#9aa9b8', '#26384a'
LIGHT_TEAL, LIGHT_BLUE = '#7fc7c1', '#a9cce6'
NAMES = {'winner': 'DAE código 2', 'latent-10': 'DAE código 10'}


def _style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14, 'axes.spines.top': False,
                         'axes.spines.right': False, 'text.color': INK, 'axes.labelcolor': INK,
                         'xtick.color': INK, 'ytick.color': INK, 'savefig.facecolor': 'white'})


def _save(figure, directory, name):
    for extension in ('png', 'svg'):
        figure.savefig(Path(directory)/f'{name}.{extension}', dpi=200, bbox_inches='tight')
    plt.close(figure)
    return f'{name}.png'


def _comma(value, decimals=0):
    return f'{value:.{decimals}f}'.replace('.', ',')


def make_figures(extension, campaign, output=None):
    """Two figures from a finished extension: recovery by level and validation curves."""
    extension, campaign = Path(extension), Path(campaign)
    info = read_json(extension/'extension.json')
    start, end = info['start_epoch'], info['end_epoch']
    output = Path(output) if output else extension/'figures'
    if output.exists() and any(output.iterdir()):
        raise ValueError('Figure output must be new or empty')
    output.mkdir(parents=True, exist_ok=True)
    with (extension/f'comparison-{start}-vs-{end}.csv').open(encoding='utf-8') as handle:
        table = list(csv.DictReader(handle))
    levels = sorted({float(row['level']) for row in table})
    trained = max(read_json(campaign/'manifest.json')['protocol']['noise']['training_levels'])
    _style()

    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    axis.axvspan(trained + .05, .32, color=GRAY, alpha=.18, lw=0)
    axis.text(.285, 1.17, 'fuera del rango\nentrenado', ha='center', va='top', fontsize=12)
    series = [('winner', start, LIGHT_BLUE, '--'), ('winner', end, BLUE, '-'),
              ('latent-10', start, LIGHT_TEAL, '--'), ('latent-10', end, TEAL, '-')]
    for variant, epochs, color, style in series:
        values = [float(next(r[f'mean_{epochs}'] for r in table if r['variant'] == variant
                             and r['metric'] == 'exact_fraction' and abs(float(r['level']) - p) < 1e-9)) for p in levels]
        axis.plot(levels, values, style, color=color, lw=3 if epochs == end else 2.4, marker='o', ms=7 if epochs == end else 6,
                  label=f'{NAMES[variant]} · {epochs} épocas')
        if epochs == start:
            earlier = dict(zip(levels, values))
        else:
            for p, v in zip(levels, values):
                if p in (.1, .2):          # "3000 → 6000" next to the 6000-epoch point
                    axis.annotate(f'{_comma(earlier[p]*100)} → {_comma(v*100)} %', (p, v),
                                  xytext=(12, 8) if variant == 'latent-10' else (12, -26), textcoords='offset points',
                                  fontsize=13, color=color, weight='bold',
                                  bbox=dict(fc='white', ec='none', alpha=.9, pad=1.5))
    axis.set(xlabel='Probabilidad de invertir cada píxel (p)', ylabel='Imágenes recuperadas exactas', xticks=levels,
             xlim=(-.012, .32), ylim=(0, 1.22), yticks=[0, .25, .5, .75, 1])
    axis.set_xticklabels([_comma(p, 2) if p else '0' for p in levels])
    axis.set_yticklabels(['0 %', '25 %', '50 %', '75 %', '100 %'])
    axis.legend(loc='lower left', frameon=False, fontsize=12)
    axis.grid(axis='y', alpha=.15)
    axis.set_axisbelow(True)
    first = _save(figure, output, 'extension-exact-by-level')

    figure, axis = plt.subplots(figsize=(9.6, 5.4))
    axis.axvline(start, color=GRAY, ls=':', lw=2)
    axis.text(start - 50, 17, f'protocolo original\n({start} épocas)', fontsize=12, va='top', ha='right')
    for variant, color in (('winner', BLUE), ('latent-10', TEAL)):
        curves = [read_json(extension/'runs'/f'{variant}-seed-{s}'/'history.json')['validation'] for s in info['seeds']]
        epochs = [p['epoch'] for p in curves[0]]
        values = np.array([[p['score'][0] for p in curve] for curve in curves])
        axis.fill_between(epochs, values.min(axis=0), values.max(axis=0), color=color, alpha=.22, lw=0)
        axis.plot(epochs, values.mean(axis=0), color=color, lw=3, label=f'{NAMES[variant]} (media de 3 semillas)')
    axis.set_yscale('log')
    axis.set_yticks([1, 2, 3, 5, 10, 20])
    axis.set_yticklabels(['1', '2', '3', '5', '10', '20'])
    axis.minorticks_off()
    axis.set(xlabel='Época', ylabel='Error de validación\n(píxeles erróneos por imagen)', xlim=(0, end + 50), ylim=(.9, 20))
    axis.legend(loc='upper right', frameon=False, fontsize=12)
    axis.grid(alpha=.15)
    second = _save(figure, output, 'extension-validation-curves')
    write_json(output/'figures-manifest.json', {'files': [first, second], 'start_epoch': start, 'end_epoch': end,
                                                 'inputs_sha256': {f'comparison-{start}-vs-{end}.csv': _sha256(
                                                     extension/f'comparison-{start}-vs-{end}.csv')}})
    return [first, second]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--campaign', type=Path, default=Path('results/denoising'))
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--epochs', type=int, default=DEFAULT_END, help='Total epochs to reach (default 6000)')
    figures = sub.add_parser('figures')
    figures.add_argument('--extension', type=Path, required=True)
    figures.add_argument('--campaign', type=Path, default=Path('results/denoising'))
    figures.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    try:
        if args.command == 'run':
            result = extend_campaign(args.campaign, args.output, args.epochs)
        else:
            result = make_figures(args.extension, args.campaign, args.output)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
