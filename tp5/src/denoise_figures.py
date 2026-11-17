"""Figures for exercise 1.b from a saved denoising campaign.

    python -m src.denoise_figures --campaign results/denoising --output results/denoising/figures

Reads only the persisted evaluation (CSV/JSON/NPZ), the final collection and the
validation histories; nothing is trained, no model is loaded and the campaign
is never modified. Example images use the indices frozen in
``evaluation/metrics.json`` before any result was inspected.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .dataset import load_dataset

COLORS = {'noisy-input': '0.45', 'ae-reference': 'tab:orange', 'winner': 'tab:blue', 'latent-10': 'tab:green'}
NAMES = {'noisy-input': 'Entrada ruidosa (sin procesar)', 'ae-reference': 'AE de 1.a (sin ruido al entrenar)',
         'winner': 'DAE código 2 (red de 1.a)', 'latent-10': 'DAE código 10'}
EXAMPLE_CHARACTERS = [0, 1, 8, 16, 24, 31]  # fixed by the campaign policy


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(figure, path):
    figure.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(figure)
    return Path(path).name


def load_campaign(campaign):
    """Collect everything the figures need, failing early on an incomplete campaign."""
    campaign = Path(campaign)
    needed = ['manifest.json', 'selection.json', 'evaluation/metrics.json', 'evaluation/by-character.csv',
              'collections/final.npz', 'reference/font.h']
    missing = [name for name in needed if not (campaign/name).is_file()]
    if missing:
        raise ValueError(f'Incomplete campaign; missing {missing}')
    manifest = _read_json(campaign/'manifest.json')
    selection = _read_json(campaign/'selection.json')
    metrics = _read_json(campaign/'evaluation/metrics.json')
    if metrics['selected_run'] != selection['run_id']:
        raise ValueError('Evaluation and selection disagree on the selected run')
    with open(campaign/'evaluation/by-character.csv', newline='', encoding='utf-8') as stream:
        by_character = [{**row, 'level': float(row['level']), 'character_index': int(row['character_index']),
                         'mean_pixel_errors': float(row['mean_pixel_errors'])} for row in csv.DictReader(stream)]
    return {'campaign': campaign, 'manifest': manifest, 'selection': selection, 'metrics': metrics,
            'by_character': by_character,
            'levels': [float(p) for p in manifest['protocol']['noise']['evaluation_levels']],
            'training_levels': [float(p) for p in manifest['protocol']['noise']['training_levels']]}


def aggregate(rows, models, levels, key):
    """Mean/min/max across ``models`` (one value per model) for each level, using all-character rows."""
    values = np.empty((len(models), len(levels)))
    for i, model in enumerate(models):
        for j, level in enumerate(levels):
            match = [r for r in rows if r['model'] == model and r['character_index'] is None
                     and np.isclose(r['level'], level)]
            if len(match) != 1:
                raise ValueError(f'Expected one summary row for {model} at level {level}')
            values[i, j] = match[0][key]
    return {'mean': values.mean(axis=0), 'min': values.min(axis=0), 'max': values.max(axis=0), 'each': values}


def _groups(data):
    runs = data['selection']['runs']
    return {variant: [r['run_id'] for r in runs if r['variant'] == variant]
            for variant in data['manifest']['variants']}


def _shade_untrained(axis, data):
    high = max(data['training_levels'])
    beyond = [p for p in data['levels'] if p > high]
    if beyond:
        axis.axvspan(high, max(beyond) + 0.01, color='0.92', zorder=0)
        axis.annotate('fuera del\nrango entrenado', xy=(high + 0.005, 1), xycoords=('data', 'axes fraction'),
                      va='top', ha='left', fontsize=7, color='0.4')


def _lines(axis, data, key, ylabel, groups):
    rows, levels = data['metrics']['by_level'], data['levels']
    for name in ('noisy-input', 'ae-reference'):
        axis.plot(levels, aggregate(rows, [name], levels, key)['mean'], 'o--' if name == 'noisy-input' else 's-',
                  color=COLORS[name], label=NAMES[name], markersize=4)
    for variant, runs in groups.items():
        stats = aggregate(rows, runs, levels, key)
        axis.fill_between(levels, stats['min'], stats['max'], color=COLORS[variant], alpha=0.2)
        axis.plot(levels, stats['mean'], 'o-', color=COLORS[variant], label=f'{NAMES[variant]} (media de 3 semillas)',
                  markersize=4)
    axis.set(xlabel='probabilidad de inversión por píxel, p', ylabel=ylabel, xticks=levels)
    axis.grid(alpha=0.3)
    _shade_untrained(axis, data)


def recovery_figure(data, output):
    groups = _groups(data)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    _lines(axes[0], data, 'mean_pixel_errors', 'píxeles erróneos por imagen (media)', groups)
    _lines(axes[1], data, 'exact_fraction', 'fracción de imágenes exactas', groups)
    _lines(axes[2], data, 'within_tolerance_fraction', 'fracción con a lo sumo 1 píxel erróneo', groups)
    axes[0].legend(fontsize=7, loc='upper left')
    figure.suptitle('Recuperación del carácter limpio según el nivel de ruido (colección final; '
                    'banda: mínimo–máximo entre semillas)', fontsize=10)
    figure.tight_layout()
    return _save(figure, output/'recovery-by-level.png')


def change_figure(data, output):
    """How much each model changes the number of wrong pixels, and for how many examples."""
    groups, rows, levels = _groups(data), data['metrics']['by_level'], data['levels']
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    _lines(axes[0], data, 'mean_error_reduction', 'reducción media de errores (entrada − salida)', groups)
    axes[0].axhline(0, color='k', linewidth=0.8)
    axes[0].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2)
    series = [('ae-reference', ['ae-reference']), *groups.items()]
    width = 0.8 / len(series)
    positions = np.arange(len(levels))
    for offset, (name, models) in enumerate(series):
        worse = aggregate(rows, models, levels, 'worsened_fraction')['mean']
        better = aggregate(rows, models, levels, 'improved_fraction')['mean']
        x = positions - 0.4 + width*(offset + 0.5)
        axes[1].bar(x, better, width, color=COLORS[name], label=f'{NAMES[name]}: mejora')
        axes[1].bar(x, -worse, width, color=COLORS[name], alpha=0.45, hatch='//',
                    label=f'{NAMES[name]}: empeora')
    axes[1].axhline(0, color='k', linewidth=0.8)
    axes[1].set(xticks=positions, xticklabels=[f'{p:g}' for p in levels], xlabel='p',
                ylabel='fracción de ejemplos\n(arriba: mejoran; abajo: empeoran)')
    axes[1].legend(fontsize=7, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.17))
    axes[1].grid(alpha=0.3, axis='y')
    figure.suptitle('Efecto sobre cada imagen frente a la entrada ruidosa', fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(figure, output/'error-reduction.png')


def validation_figure(data, output):
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    selected = data['selection']['run_id']
    styles = {42: '-', 43: '--', 44: ':'}
    for run in data['selection']['runs']:
        history = _read_json(data['campaign']/'runs'/run['run_id']/'history.json')['validation']
        epochs = [point['epoch'] for point in history]
        for axis, column in zip(axes, (0, 1)):
            axis.plot(epochs, [point['score'][column] for point in history], styles.get(run['seed'], '-'),
                      color=COLORS[run['variant']], linewidth=2.2 if run['run_id'] == selected else 1.2,
                      label=run['run_id'] + (' (seleccionada)' if run['run_id'] == selected else ''))
            best = run['best']['epoch']
            axis.plot([best], [run['best']['score'][column]], 'o', color=COLORS[run['variant']], markersize=5)
    for axis, label in zip(axes, ('error binario medio (píxeles por imagen)', 'MSE por píxel')):
        axis.set(xlabel='época', ylabel=label, yscale='log')
        axis.grid(alpha=0.3, which='both')
    axes[0].legend(fontsize=7)
    figure.suptitle('Validación: promedio de p = 0,05; 0,10 y 0,20 (punto: mejor checkpoint de cada corrida)',
                    fontsize=10)
    figure.tight_layout()
    return _save(figure, output/'validation-curves.png')


def character_figure(data, labels, output):
    groups = _groups(data)
    positive = [p for p in data['levels'] if p > 0]
    figure, axes = plt.subplots(1, len(groups), figsize=(5.2*len(groups), 9.5), squeeze=False)
    tables = {}
    for variant, runs in groups.items():
        table = np.zeros((32, len(positive)))
        for row in data['by_character']:
            if row['model'] in runs and row['level'] in positive:
                table[row['character_index'], positive.index(row['level'])] += row['mean_pixel_errors']/len(runs)
        tables[variant] = table
    vmax = max(table.max() for table in tables.values())
    for axis, (variant, table) in zip(axes[0], tables.items()):
        image = axis.imshow(table, cmap='magma_r', vmin=0, vmax=vmax, aspect='auto')
        for i in range(32):
            for j in range(len(positive)):
                axis.text(j, i, f'{table[i, j]:.1f}', ha='center', va='center', fontsize=6,
                          color='white' if table[i, j] > 0.6*vmax else 'black')
        axis.set(xticks=range(len(positive)), xticklabels=[f'{p:g}' for p in positive],
                 yticks=range(32), yticklabels=labels, xlabel='p', title=NAMES[variant])
        axis.tick_params(axis='y', labelsize=7)
    figure.colorbar(image, ax=axes[0].tolist(), fraction=0.03, label='píxeles erróneos tras el DAE (media de 3 semillas)')
    figure.suptitle('Error por carácter y nivel de ruido', fontsize=10)
    return _save(figure, output/'by-character.png')


def _glyph(axis, vector, title=None, caption=None, color='black'):
    """Draw a 7x5 glyph; the error count goes below it so it cannot be read as the next row's title."""
    axis.imshow(np.asarray(vector).reshape(7, 5), cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
    axis.set_xticks([])
    axis.set_yticks([])
    if title:
        axis.set_title(title, fontsize=8)
    if caption:
        axis.set_xlabel(caption, fontsize=7, color=color, labelpad=2)


def sample_figures(data, labels, output):
    """One sheet per noise level: original / noisy / AE of 1.a / DAE code 2 / selected DAE code 10."""
    campaign, metrics, selection = data['campaign'], data['metrics'], data['selection']
    final = np.load(campaign/'collections/final.npz', allow_pickle=False)
    groups = _groups(data)
    chosen = {}
    for variant, runs in groups.items():  # same rule as the selection: best validation score, then seed
        chosen[variant] = min((r for r in selection['runs'] if r['run_id'] in runs),
                              key=lambda r: (r['best']['score'], r['seed']))['run_id']
    rows = [('Original', None), ('Entrada ruidosa', 'noisy-input'), (NAMES['ae-reference'], 'ae-reference')]
    rows += [(f'{NAMES[v]}\n{run}', run) for v, run in chosen.items()]
    arrays = {name: np.load(campaign/'evaluation'/f'{name}.npz', allow_pickle=False)
              for _, name in rows if name is not None}
    per_level = len(EXAMPLE_CHARACTERS)
    indices = metrics['example_indices']
    if len(indices) != per_level*len(data['levels']):
        raise ValueError('Unexpected number of frozen example indices')
    names = []
    for k, level in enumerate(data['levels']):
        picked = indices[k*per_level:(k + 1)*per_level]
        for index, character in zip(picked, EXAMPLE_CHARACTERS):
            if not np.isclose(final['levels'][index], level) or final['character_indices'][index] != character:
                raise ValueError('Frozen example index does not match its level/character')
        figure, axes = plt.subplots(len(rows), per_level, figsize=(1.35*per_level + 2.6, 1.75*len(rows)))
        for r, (row_label, name) in enumerate(rows):
            for c, index in enumerate(picked):
                target = final['targets'][index]
                if name is None:
                    _glyph(axes[r, c], target, labels[EXAMPLE_CHARACTERS[c]])
                    continue
                if name == 'noisy-input':
                    image, wrong = final['inputs'][index], int(arrays[name]['input_pixel_errors'][index])
                else:
                    image, wrong = arrays[name]['binary'][index], int(arrays[name]['pixel_errors'][index])
                _glyph(axes[r, c], image, caption=f'{wrong} err', color='tab:red' if wrong > 1 else 'black')
            axes[r, 0].text(-0.15, 0.5, row_label, transform=axes[r, 0].transAxes, ha='right', va='center',
                            fontsize=7)
        figure.suptitle(f'p = {level:g}'
                        + (' (fuera del rango entrenado)' if level > max(data['training_levels']) else '')
                        + ' — primera realización de los caracteres fijados antes de ver resultados; '
                          'rojo: más de 1 píxel erróneo', fontsize=8)
        names.append(_save(figure, output/f'samples-p{level:g}.png'))
    return names, chosen


def make_figures(campaign, output):
    """Write every figure plus a manifest into a new directory; returns the manifest."""
    data = load_campaign(campaign)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Figure output must be new or empty')
    output.mkdir(parents=True, exist_ok=True)
    labels = [label.split(' ', 1)[1] for label in load_dataset(Path(campaign)/'reference/font.h').labels]
    files = [recovery_figure(data, output), change_figure(data, output), validation_figure(data, output),
             character_figure(data, labels, output)]
    samples, chosen = sample_figures(data, labels, output)
    files += samples
    manifest = {'campaign': str(Path(campaign).resolve()), 'files': files, 'sample_models': chosen,
                'example_indices': data['metrics']['example_indices'],
                'inputs_sha256': {name: _sha256(Path(campaign)/name) for name in
                                  ('selection.json', 'evaluation/metrics.json', 'evaluation/by-character.csv',
                                   'collections/final.npz')},
                'versions': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__}}
    (output/'figures-manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--campaign', type=Path, default=Path('results/denoising'))
    parser.add_argument('--output', type=Path, default=None, help='New directory (default: <campaign>/figures)')
    args = parser.parse_args()
    try:
        manifest = make_figures(args.campaign, args.output or args.campaign/'figures')
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print('Figures:', ', '.join(manifest['files']))


if __name__ == '__main__':
    main()
