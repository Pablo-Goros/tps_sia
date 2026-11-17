"""Extra figures for the 1.b slides: validation by run, headline results,
validation curves, improved/worsened examples and a numeric table.

Reads only results/denoising/ (no training, no model loading) and checks the
numbers quoted in report.md before drawing. Complements generate_denoising.py.
"""
from pathlib import Path
import csv, hashlib, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent/'assets'
OUT.mkdir(exist_ok=True)
CAMP = OUT.parents[1]/'results/denoising'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14, 'axes.spines.top': False,
                     'axes.spines.right': False, 'text.color': '#26384a', 'axes.labelcolor': '#26384a',
                     'xtick.color': '#26384a', 'ytick.color': '#26384a', 'savefig.facecolor': 'white'})
TEAL = '#008f86'; BLUE = '#368dc4'; GRAY = '#9aa9b8'; INK = '#26384a'; RED = '#c0392b'
LIGHT_TEAL = '#7fc7c1'; LIGHT_GRAY = '#d9e0e7'; NAVY = '#16263F'; CREAM = '#F5F1EA'
LEVELS = [0, .05, .1, .2, .3]
SEEDS = (42, 43, 44)
NOISY = 'Entrada ruidosa'; AE = 'AE de 1.a'; DAE2 = 'DAE código 2'; DAE10 = 'DAE código 10'


def es(value, decimals=2):
    return f'{value:.{decimals}f}'.replace('.', ',')


def pct(value):
    return f'{int(round(value*100))} %'


def save(fig, name):
    for ext in ('png', 'svg'):
        fig.savefig(OUT/f'{name}.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def read_csv(name):
    with (CAMP/'evaluation'/name).open(encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def check(values, expected, decimals=3):
    assert [round(v, decimals) for v in values] == expected, (values, expected)


by_level = {(r['model'], float(r['level'])): r for r in read_csv('by-level.csv') if r['character_index'] == ''}
agg = {(r['variant'], float(r['level'])): r for r in read_csv('seed-aggregates.csv')}
selection = json.loads((CAMP/'selection.json').read_text(encoding='utf-8'))
runs = {r['run_id']: r for r in selection['runs']}
VARIANTS = {'winner': (DAE2, BLUE), 'latent-10': (DAE10, TEAL)}


def seed_mean(variant, key):
    return [float(np.mean([float(by_level[(f'{variant}-seed-{s}', p)][key]) for s in SEEDS])) for p in LEVELS]


def model_values(model, key):
    return [float(by_level[(model, p)][key]) for p in LEVELS]


# ---------- Checks against the numbers published in report.md ----------
validation = {a['variant']: a['mean_score'][0] for a in selection['aggregates']}
check([validation['winner'], validation['latent-10']], [2.28, 1.23], 2)
assert selection['run_id'] == 'latent-10-seed-44'
assert sum(r['best']['epoch'] == 3000 for r in selection['runs']) == 4
check(seed_mean('latent-10', 'exact_fraction')[1:], [.955, .852, .496, .173])
check(seed_mean('latent-10', 'within_tolerance_fraction')[1:], [.974, .899, .57, .221])
check(model_values('noisy-input', 'exact_fraction')[1:], [.166, .029, .001, 0])
check(model_values('ae-reference', 'worsened_fraction')[1:], [.589, .731, .799, .767])
check(seed_mean('latent-10', 'worsened_fraction')[1:], [.009, .027, .111, .193])
check(seed_mean('winner', 'worsened_fraction')[1:], [.092, .112, .259, .361])
check([float(agg[('winner', 0)]['mean_pixel_errors_mean']), float(agg[('latent-10', 0)]['mean_pixel_errors_mean'])], [.26, 0])

# ---------- 8. Validation error by run ----------
order = [f'{v}-seed-{s}' for v in ('winner', 'latent-10') for s in SEEDS]
x = np.array([0, 1, 2, 3.7, 4.7, 5.7])
scores = [runs[r]['best']['score'][0] for r in order]
fig, ax = plt.subplots(figsize=(9.2, 5.3))
bars = ax.bar(x, scores, width=.72, color=[BLUE]*3+[TEAL]*3)
for bar, score, run in zip(bars, scores, order):
    ax.text(bar.get_x()+bar.get_width()/2, score+.05, es(score), ha='center', va='bottom', fontsize=15, zorder=6,
            weight='bold' if run == selection['run_id'] else 'normal', color=TEAL if run == selection['run_id'] else INK,
            bbox=dict(fc='white', ec='none', pad=1.2))
selected = bars[order.index(selection['run_id'])]
selected.set_edgecolor(INK); selected.set_linewidth(2.6)
ax.annotate('seleccionada', (selected.get_x()+selected.get_width()/2, scores[order.index(selection['run_id'])]+.38),
            ha='center', fontsize=13, color=TEAL, weight='bold')
for variant, (name, color), xs in [('winner', VARIANTS['winner'], (-.5, 2.5)), ('latent-10', VARIANTS['latent-10'], (3.2, 6.2))]:
    ax.hlines(validation[variant], *xs, color=INK, ls='--', lw=1.6)
ax.set_xticks(x); ax.set_xticklabels([f'sem. {s}' for s in SEEDS]*2, fontsize=13)
for center, name, color, variant in [(1, DAE2, BLUE, 'winner'), (4.7, DAE10, TEAL, 'latent-10')]:
    ax.text(center, -.13, f'{name}\nmedia {es(validation[variant])}', ha='center', va='top',
            transform=ax.get_xaxis_transform(), fontsize=16, weight='bold', color=color, linespacing=1.3)
ax.text(-.5, 3.02, 'línea punteada: media de las 3 semillas', fontsize=12.5, va='top')
ax.set(ylabel='Error de validación\n(píxeles erróneos por imagen)', ylim=(0, 3.1), xlim=(-.7, 6.4), yticks=[0, .5, 1, 1.5, 2, 2.5, 3])
ax.set_yticklabels([es(v, 1) for v in (0, .5, 1, 1.5, 2, 2.5, 3)])
ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
save(fig, 'denoising-08-validation-by-run')

# ---------- 9. Headline results: exact images by level ----------
levels = LEVELS[1:]
series = [(NOISY+' (exactas)', model_values('noisy-input', 'exact_fraction')[1:], GRAY),
          (DAE10+': exactas', seed_mean('latent-10', 'exact_fraction')[1:], TEAL),
          (DAE10+': a lo sumo 1 píxel erróneo', seed_mean('latent-10', 'within_tolerance_fraction')[1:], LIGHT_TEAL)]
fig, ax = plt.subplots(figsize=(9.6, 5.4))
width = .26
ax.axvspan(2.5, 3.6, color=GRAY, alpha=.18, lw=0)
ax.text(3.05, 1.0, 'fuera del rango\nentrenado', ha='center', va='top', fontsize=12)
for k, (label, values, color) in enumerate(series):
    xs = np.arange(4)+(k-1)*width
    ax.bar(xs, values, width, color=color, label=label)
    for xi, v in zip(xs, values):
        ax.text(xi, v+.012, pct(v), ha='center', va='bottom', fontsize=12.5, color=INK if k != 1 else TEAL, weight='bold' if k == 1 else 'normal')
ax.set_xticks(range(4)); ax.set_xticklabels([f'p = {es(p)}' for p in levels], fontsize=14)
ax.set(ylabel='Imágenes recuperadas', ylim=(0, 1.42), xlim=(-.55, 3.6), yticks=[0, .25, .5, .75, 1])
ax.set_yticklabels(['0 %', '25 %', '50 %', '75 %', '100 %'])
ax.legend(loc='upper left', frameon=False, fontsize=12.5)
ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
save(fig, 'denoising-09-headline-results')

# ---------- 10. Validation curves ----------
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.axvspan(2500, 3000, color=GRAY, alpha=.18, lw=0)
ax.text(2750, 17, 'últimas\n500 épocas', ha='center', va='top', fontsize=12)
styles = {42: '-', 43: '--', 44: ':'}
for run in order:
    variant, seed = run.rsplit('-seed-', 1)
    history = json.loads((CAMP/'runs'/run/'history.json').read_text(encoding='utf-8'))['validation']
    epochs = [p['epoch'] for p in history]; values = [p['score'][0] for p in history]
    selected = run == selection['run_id']
    ax.plot(epochs, values, styles[int(seed)], color=VARIANTS[variant][1], lw=3.4 if selected else 2.2,
            label=f'{VARIANTS[variant][0]} · sem. {seed}' + (' (seleccionada)' if selected else ''))
    best = runs[run]['best']
    ax.plot([best['epoch']], [best['score'][0]], 'o', color=VARIANTS[variant][1], ms=9, mec='white', mew=1.5, zorder=5)
ax.set_yscale('log'); ax.set_yticks([1, 2, 3, 5, 10, 20]); ax.set_yticklabels(['1', '2', '3', '5', '10', '20'])
ax.minorticks_off()
ax.set(xlabel='Época', ylabel='Error de validación\n(píxeles erróneos por imagen)', xlim=(0, 3050), ylim=(.95, 20))
ax.legend(loc='upper right', bbox_to_anchor=(.82, 1.0), frameon=False, fontsize=11.5, ncol=1)
ax.text(40, 1.08, '● mejor checkpoint de cada corrida', fontsize=12, color=INK)
ax.grid(alpha=.15, which='major')
save(fig, 'denoising-10-validation-curves')

# ---------- 11. Improved, unchanged or worse than the noisy input ----------
fig, axes = plt.subplots(1, 4, figsize=(12.5, 5.2), sharey=True)
models = [(AE, ['ae-reference']), (DAE2, [f'winner-seed-{s}' for s in SEEDS]), (DAE10, [f'latent-10-seed-{s}' for s in SEEDS])]
for ax, p in zip(axes, levels):
    for k, (name, ids) in enumerate(models):
        better = float(np.mean([float(by_level[(m, p)]['improved_fraction']) for m in ids]))
        worse = float(np.mean([float(by_level[(m, p)]['worsened_fraction']) for m in ids]))
        same = 1-better-worse
        assert abs(same-float(np.mean([float(by_level[(m, p)]['unchanged_fraction']) for m in ids]))) < 1e-9
        ax.bar(k, better, .84, color=TEAL)
        ax.bar(k, same, .84, bottom=better, color=LIGHT_GRAY)
        ax.bar(k, worse, .84, bottom=better+same, color=RED)
        if better > .06: ax.text(k, better/2, pct(better), ha='center', va='center', color='white', fontsize=12.5, weight='bold')
        if worse > .06: ax.text(k, better+same+worse/2, pct(worse), ha='center', va='center', color='white', fontsize=12.5, weight='bold')
        elif worse > 0: ax.text(k, 1.015, pct(worse), ha='center', va='bottom', color=RED, fontsize=12.5, weight='bold')
    ax.set_xticks(range(3)); ax.set_xticklabels(['AE\n1.a', 'DAE\ncód. 2', 'DAE\ncód. 10'], fontsize=12)
    ax.set_title(f'p = {es(p)}' + ('\n(fuera de rango)' if p == .3 else ''), fontsize=16, pad=10, color=GRAY if p == .3 else INK, weight='bold')
    ax.set_ylim(0, 1.08); ax.set_yticks([0, .25, .5, .75, 1])
    ax.tick_params(axis='x', length=0)
axes[0].set_yticklabels(['0 %', '25 %', '50 %', '75 %', '100 %']); axes[0].set_ylabel('Ejemplos', fontsize=15)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (TEAL, LIGHT_GRAY, RED)]
fig.legend(handles, ['Menos errores que la entrada', 'Igual', 'Más errores que la entrada'], loc='lower center',
           ncol=3, frameon=False, fontsize=13, bbox_to_anchor=(.5, -.04))
fig.subplots_adjust(wspace=.12, bottom=.2)
save(fig, 'denoising-11-better-or-worse')

# ---------- 12. Results table ----------
columns = ['p = 0', 'p = 0,05', 'p = 0,10', 'p = 0,20', 'p = 0,30*']
fig, ax = plt.subplots(figsize=(11.5, 6.3))
ax.set_xlim(0, 11.5); ax.set_ylim(-.75, 5.6); ax.axis('off')
col_x = [3.15+1.66*i for i in range(5)]


def cell_pixels(variant, p):
    row = agg[(variant, p)]
    low, high = float(row['mean_pixel_errors_min']), float(row['mean_pixel_errors_max'])
    mean = float(row['mean_pixel_errors_mean'])
    return es(mean) if abs(high-low) < 5e-3 else f'{es(mean)}\n[{es(low)}–{es(high)}]'


blocks = [('Píxeles erróneos por imagen',
           [(NOISY, GRAY, [es(v) for v in model_values('noisy-input', 'mean_pixel_errors')]),
            (AE, GRAY, [es(v) for v in model_values('ae-reference', 'mean_pixel_errors')]),
            (DAE2, BLUE, [cell_pixels('winner', p) for p in LEVELS]),
            (DAE10, TEAL, [cell_pixels('latent-10', p) for p in LEVELS])]),
          ('Imágenes recuperadas exactas',
           [(NOISY, GRAY, [pct(v) for v in model_values('noisy-input', 'exact_fraction')]),
            (AE, GRAY, [pct(v) for v in model_values('ae-reference', 'exact_fraction')]),
            (DAE2, BLUE, [pct(v) for v in seed_mean('winner', 'exact_fraction')]),
            (DAE10, TEAL, [pct(v) for v in seed_mean('latent-10', 'exact_fraction')])])]
y = 5.45
ax.add_patch(plt.Rectangle((0, y-.5), 11.5, .5, color=NAVY))
for cx, name in zip(col_x, columns):
    ax.text(cx+.8, y-.25, name, color='white', ha='center', va='center', fontsize=14, weight='bold')
y -= .5
for title, rows in blocks:
    ax.text(.1, y-.28, title, fontsize=14, weight='bold', va='center', color=NAVY)
    y -= .5
    for i, (name, color, cells) in enumerate(rows):
        height = .62 if title.startswith('Píxeles') and name in (DAE2, DAE10) else .42
        ax.add_patch(plt.Rectangle((0, y-height), 11.5, height, color=CREAM if i % 2 == 0 else 'white', lw=0))
        ax.text(.25, y-height/2, name, fontsize=13, va='center', color=color, weight='bold' if name == DAE10 else 'normal')
        for cx, text in zip(col_x, cells):
            ax.text(cx+.8, y-height/2, text, ha='center', va='center', fontsize=12.5 if '\n' not in text else 10.5,
                    color=TEAL if name == DAE10 else INK, weight='bold' if name == DAE10 else 'normal', linespacing=1.1)
        y -= height
    y -= .12
ax.text(.1, y-.1, 'Entre corchetes: mínimo–máximo entre 3 semillas.  * p = 0,30: fuera del rango entrenado.\n3.200 imágenes por nivel con ruido y 32 limpias (evaluación independiente).',
        fontsize=11, color=INK, va='top', linespacing=1.4)
save(fig, 'denoising-12-results-table')

# ---------- Provenance ----------
inputs = ['evaluation/by-level.csv', 'evaluation/seed-aggregates.csv', 'selection.json'] + [f'runs/{r}/history.json' for r in order]
(OUT/'denoising-extra-provenance.json').write_text(json.dumps({
    'campaign': 'tps_sia/tp5/results/denoising',
    'inputs_sha256': {p: hashlib.sha256((CAMP/p).read_bytes()).hexdigest() for p in inputs},
    'selected_run': selection['run_id'], 'validation_mean_pixel_errors': validation,
    'trained_or_loaded_models': False}, ensure_ascii=False, indent=2), encoding='utf-8')
print('Generated 5 extra denoising figures in PNG/SVG; verified the key numbers against results/denoising.')
