"""Tablas y gráficos del paso 7, únicamente desde artefactos guardados."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tps_sia.tp3.shared.analysis import generate, generate_comparison
from .search import confirm_groups


def generate_study(results: Path, output: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    results, output = results.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selection = json.loads((results / 'selection.json').read_text())
    stages = {}
    rendered = set()
    for stage in ('rates', 'architectures', 'confirmation'):
        manifest = json.loads((results / f'{stage}.json').read_text())
        paths = [results / 'runs' / run_id / 'results.json' for run_id in manifest['run_ids']]
        stages[stage] = [json.loads(p.read_text()) for p in paths]
        generate_comparison(paths, output / stage)
        for path in paths:
            if path not in rendered:
                generate(path, output / 'runs' / path.parent.name)
                rendered.add(path)
    summary = confirm_groups(stages['confirmation'], selection['seeds'])
    if summary != selection['evidence']['ranking']:
        raise ValueError('Saved selection differs from confirmation evidence.')

    for stage in ('rates', 'architectures'):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        for r in stages[stage]:
            if r['status'] != 'completed':
                continue
            label = r['metadata']['candidate']
            for ax, key in zip(axes, ('accuracy_validacion', 'costo_validacion')):
                ax.plot(r['history']['validation_epochs'], r['history'][key], label=label)
                ax.set(xlabel='Época', ylabel='Accuracy de validación' if key.startswith('accuracy') else 'Cross-entropy de validación')
                ax.grid(alpha=0.2)
        axes[1].set_yscale('log')
        axes[1].legend(fontsize=7)
        fig.savefig(output / f'{stage}.png', dpi=160)
        plt.close(fig)

    names = []
    for item in summary:
        r = next(r for r in stages['confirmation'] if r['config_id'] == item['config_id'])
        names.append(r['metadata']['candidate'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].bar(names, [s['accuracy_mean'] for s in summary], yerr=[s['accuracy_std'] for s in summary], capsize=5)
    axes[0].set(ylabel='Accuracy de validación (media ± desvío)', ylim=(0.9, 1))
    axes[1].bar(names, [s['duration_seconds_mean'] for s in summary])
    axes[1].set(ylabel='Tiempo medio por corrida (s)')
    for ax in axes:
        ax.tick_params(axis='x', labelrotation=25)
    fig.savefig(output / 'confirmation.png', dpi=160)
    plt.close(fig)

    lines = ['# Comparaciones de desarrollo — Ejercicio 2', '',
             'Resultados de la partición fija de `digits.csv`; test reservado.', '']
    for stage in ('rates', 'architectures'):
        title = {'rates': 'Tasas y optimizadores', 'architectures': 'Arquitecturas'}[stage]
        lines += [f'## {title}', '', '| Candidato | Accuracy | Cross-entropy | Época | Parámetros | Segundos |',
                  '|---|---:|---:|---:|---:|---:|']
        for r in stages[stage]:
            if r['status'] != 'completed':
                lines.append(f'| {r["metadata"]["candidate"]} | fallo | — | — | — | — |')
                continue
            m = r['metrics']['validation']
            lines.append(f'| {r["metadata"]["candidate"]} | {m["accuracy"]:.6f} | {m["loss"]:.6f} | {r["chosen_epoch"]} | {r["parameter_count"]} | {r["duration_seconds"]:.1f} |')
        lines += ['', f'![Curvas de validación]({stage}.png)', '']
    lines += ['## Confirmación con tres semillas', '',
              '| Candidato | Accuracy media ± desvío | CE media ± desvío | Macro-F1 | Balanced accuracy | Épocas elegidas |',
              '|---|---:|---:|---:|---:|---|']
    for name, s in zip(names, summary):
        lines.append(f'| {name} | {s["accuracy_mean"]:.6f} ± {s["accuracy_std"]:.6f} | {s["loss_mean"]:.6f} ± {s["loss_std"]:.6f} | {s["macro_f1_mean"]:.6f} | {s["balanced_accuracy_mean"]:.6f} | {s["chosen_epochs"]} |')
    lines += ['', '![Confirmación](confirmation.png)', '', '## Candidato con semilla predefinida 42', '',
              '| Dígito | Soporte | Precision | Recall | F1 |', '|---|---:|---:|---:|---:|']
    winner = next(r for r in stages['confirmation'] if r['config_id'] == selection['config_id'] and r['config']['model_seed'] == 42)
    def metric(value):
        return 'N/E' if value is None else f'{value:.6f}'
    for row in winner['metrics']['validation']['per_class']:
        lines.append(f'| {row["class"]} | {row["support"]} | {metric(row["precision"])} | {metric(row["recall"])} | {metric(row["f1"])} |')
    lines += ['', f'![Train y validación](runs/{winner["run_id"]}/learning_curves.png)', '',
              f'![Confusión](runs/{winner["run_id"]}/confusion_normalized.png)', '']
    (output / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    return winner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    generate_study(args.results_dir, args.output_dir)


if __name__ == '__main__':
    main()
