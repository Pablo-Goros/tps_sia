"""Report and figures of the ej2 v2 appendix, from saved summaries only (no training)."""
from __future__ import annotations

import json
from pathlib import Path

COLORS = {'bar': '#2a78d6', 'seed_42': '#eb6834', 'ink': '#0b0b0b', 'ink_2': '#52514e',
          'muted': '#898781', 'surface': '#fcfcfb'}
FACTOR_TITLES = {'width': 'Ancho de la capa oculta (etapa 3)', 'depth': 'Profundidad (etapa 4)',
                 'batch_size': 'Tamaño de lote (etapa 5)'}


def pct(value, digits=2):
    return '—' if value is None else f'{100 * value:.{digits}f}'


def value_text(value):
    return '-'.join(map(str, value[1:-1])) if isinstance(value, list) else str(value)


def plot_factor(name: str, factor: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    groups = sorted(factor['groups'], key=lambda g: factor['order'].index(g['config_id']))
    labels = [value_text(g['value']) for g in groups]
    means = [100 * g['accuracy_mean'] for g in groups]
    stds = [100 * g['accuracy_std'] for g in groups]
    originals = [100 * g['seed_42_original']['accuracy'] for g in groups]
    x = range(len(groups))
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True, facecolor=COLORS['surface'])
    ax.set_facecolor(COLORS['surface'])
    ax.bar(x, means, width=0.6, color=COLORS['bar'], yerr=stds, capsize=4,
           error_kw={'ecolor': COLORS['ink_2'], 'elinewidth': 1.2},
           label='Media de 3 semillas (0, 1, 42) ± desvío')
    ax.scatter(x, originals, s=64, marker='D', color=COLORS['seed_42'], edgecolor=COLORS['surface'],
               linewidth=1.5, zorder=3, label='Semilla 42 de la etapa original (exploratorio)')
    low = min(min(m - s for m, s in zip(means, stds)), min(originals))
    high = max(max(m + s for m, s in zip(means, stds)), max(originals))
    ax.set_ylim(max(0, low - 1.0), min(100, high + 1.0))
    ax.set_xticks(list(x), labels)
    ax.set_xlabel(FACTOR_TITLES[name].split(' (')[0], color=COLORS['ink_2'])
    ax.set_ylabel('Accuracy de validación (%)\n(el eje no empieza en 0)', color=COLORS['ink_2'])
    ax.set_title(FACTOR_TITLES[name], color=COLORS['ink'], loc='left')
    ax.tick_params(colors=COLORS['ink_2'])
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(COLORS['muted'])
    ax.grid(axis='y', color=COLORS['muted'], alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=1,
              labelcolor=COLORS['ink_2'])
    fig.savefig(path, dpi=160, facecolor=COLORS['surface'])
    plt.close(fig)


def relu_section(summary: dict) -> list[str]:
    groups = {g['config']['activation']: g for g in summary['groups']}
    comparison = summary['relu_vs_tanh']
    lines = ['## A. ReLU en la configuración final', '',
             'Comprobación posterior, **no selección**: no puede cambiar el ganador ni `selection.json`. '
             'Misma configuración que la seleccionada (`[784,256,10]`, momentum 0,1, lote 32, parada '
             'temprana, partición 42), cambiando sólo la activación; inicialización automática del código '
             'actual (He en las capas ocultas con ReLU; Xavier en la salida y con tanh).', '',
             f"Semillas: {', '.join(map(str, summary['seeds']))}. Corridas reutilizadas de v2: "
             f"{summary['runs_reused_from_v2']}; entrenadas: {summary['runs_trained_or_resumed']}; "
             f"tiempo: {summary['wall_seconds']:.0f} s.", '',
             '| Activación | Accuracy val. (%) | Desvío (pp) | CE val. | Desvío CE | Mejores épocas (semillas 0, 1, 2, 3, 42) | Tiempo medio (s) |',
             '|---|---:|---:|---:|---:|---|---:|']
    for name in ('tanh', 'relu'):
        g = groups[name]
        lines.append(f"| {name}{' (seleccionada)' if name == 'tanh' else ''} | {pct(g['accuracy_mean'])} | "
                     f"{pct(g['accuracy_std'])} | {g['loss_mean']:.4f} | {g['loss_std']:.4f} | "
                     f"{', '.join(map(str, g['chosen_epochs']))} | {g['duration_seconds_mean']:.1f} |")
    if comparison is not None:
        better = 'ReLU' if comparison['better'] == groups['relu']['config_id'] else 'tanh'
        lines += ['', f"Diferencia de medias a favor de {better}: {pct(comparison['difference'])} pp; "
                      f"σ agrupado {pct(comparison['sigma'])} pp; umbral 2σ = {pct(comparison['threshold'])} pp. "
                      f"{'Supera' if comparison['significant'] else 'No supera'} el umbral de la regla de la etapa 7."]
    return lines


def stages_section(summary: dict) -> list[str]:
    lines = ['## B. Etapas 3, 4 y 5 con tres semillas', '',
             'Misma base que el incumbente de esas etapas (tanh, momentum 0,1, `[784,128,10]`, lote 32), '
             'variando un factor a la vez como en `search_v2.json`. Los valores de **semilla 42** son los de '
             'la etapa original: una sola semilla, por lo tanto **exploratorios**. La columna Δ es la media de '
             '3 semillas menos ese valor.', '',
             f"Corridas reutilizadas de v2: {summary['runs_reused_from_v2']}; entrenadas: "
             f"{summary['runs_trained_or_resumed']}; tiempo: {summary['wall_seconds']:.0f} s.", '']
    for name, factor in summary['factors'].items():
        groups = sorted(factor['groups'], key=lambda g: factor['order'].index(g['config_id']))
        lines += [f"### {FACTOR_TITLES[name]}", '',
                  '| Valor | Accuracy 3 semillas (%) | Desvío (pp) | CE media | Semilla 42 original (%) | Δ (pp) | Mejores épocas (semillas 0, 1, 42) |',
                  '|---|---:|---:|---:|---:|---:|---|']
        for g in groups:
            mark = ' (incumbente)' if g['config_id'] == factor['incumbent_config_id'] else ''
            lines.append(f"| {value_text(g['value'])}{mark} | {pct(g['accuracy_mean'])} | {pct(g['accuracy_std'])} | "
                         f"{g['loss_mean']:.4f} | {pct(g['seed_42_original']['accuracy'])} | "
                         f"{'+' if g['difference_vs_seed_42'] >= 0 else ''}{pct(g['difference_vs_seed_42'])} | "
                         f"{', '.join(map(str, g['chosen_epochs']))} |")
        by_id = {g['config_id']: g for g in groups}
        best, first = by_id[factor['best_3_seeds_config_id']], by_id[factor['best_seed_42_config_id']]
        lines += ['', f"Mejor con 3 semillas: {value_text(best['value'])}; mejor con semilla 42 (etapa original): "
                      f"{value_text(first['value'])}. {'Coinciden.' if factor['same_winner'] else '**No coinciden.**'}"]
        comparison = factor['best_vs_incumbent']
        if comparison is not None:
            lines.append(f"Contra el incumbente: +{pct(comparison['difference'])} pp, umbral 2σ = "
                         f"{pct(comparison['threshold'])} pp; {'supera' if comparison['significant'] else 'no supera'} el umbral.")
        if name == 'batch_size':
            lines += ['', 'Nota: con lote 16, tasa 0,1 y momentum 0,9 el entrenamiento es inestable (mejor época 2 '
                          'en la etapa original). La comparación de lotes está confundida con la tasa, que el '
                          'protocolo mantiene fija; no se ajustó nada.']
        lines += ['', f"![{FACTOR_TITLES[name]}](analysis/stages_{name}.png)", '']
    return lines


def write_report(output: Path) -> Path:
    relu = json.loads((output / 'relu' / 'summary.json').read_text(encoding='utf-8'))
    stages = json.loads((output / 'stages_3_5' / 'summary.json').read_text(encoding='utf-8'))
    analysis = output / 'analysis'
    analysis.mkdir(parents=True, exist_ok=True)
    for name, factor in stages['factors'].items():
        plot_factor(name, factor, analysis / f'stages_{name}.png')
    lines = ['# Apéndice de la búsqueda v2 del ejercicio 2', '',
             f"Generado desde `relu/summary.json` y `stages_3_5/summary.json`. Selección vigente: "
             f"`{relu['selected_config_id']}` (no se modifica). El conjunto de test no se usa.", '',
             *relu_section(relu), '', *stages_section(stages)]
    path = output / 'report.md'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path
