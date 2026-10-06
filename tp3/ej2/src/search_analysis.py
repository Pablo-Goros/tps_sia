"""Tablas y gráficos del paso 7 (v1) y de la búsqueda por etapas v2,
únicamente desde artefactos guardados: no carga datasets ni entrena."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tps_sia.tp3.shared.analysis import generate, generate_comparison, plot_accuracy_bars, plot_run_curves
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


STAGE_TITLES = {0: 'Etapa 0 — Activación', 1: 'Etapa 1 — Tasa por optimizador', 2: 'Etapa 2 — Optimizador',
                3: 'Etapa 3 — Ancho', 4: 'Etapa 4 — Profundidad', 5: 'Etapa 5 — Tamaño de lote',
                6: 'Etapa 6 — Cruce chico (3 semillas)', 7: 'Etapa 7 — Confirmación (5 semillas)'}
STOP_TEXT = {'early_stopping': 'parada temprana', 'max_epochs': 'no convergió (presupuesto agotado)',
             'numerical_failure': 'fallo numérico', 'interrupted': 'interrumpida'}


def _stage_rows(stage):
    """Every seed row of a stage: main runs plus near-tie repetitions."""
    runs = stage['runs']
    rows = [r for group in runs.values() for r in group] if isinstance(runs, dict) else list(runs)
    near = (stage.get('decision') or {}).get('near_tie')
    return rows + (near['extra_runs'] if near else [])


def _display_groups(rows):
    """Mean and sample spread over whatever seeds exist (display only)."""
    groups = {}
    for row in rows:
        groups.setdefault(row['config_id'], {})[row['seed']] = row  # reused rows appear once
    entries = []
    for config_id, by_seed in groups.items():
        group = [by_seed[seed] for seed in sorted(by_seed)]
        done = [r for r in group if r['status'] == 'completed']
        accuracy = [r['accuracy'] for r in done]
        entries.append({
            'config_id': config_id, 'label': group[0]['label'], 'seeds': sorted(by_seed),
            'completed': len(done), 'runs': len(group),
            'accuracy_mean': float(np.mean(accuracy)) if done else None,
            'accuracy_std': float(np.std(accuracy, ddof=1)) if len(done) > 1 else None,
            'loss_mean': float(np.mean([r['loss'] for r in done])) if done else None,
            'chosen_epochs': [r['chosen_epoch'] for r in group], 'epochs_run': [r['epochs_run'] for r in group],
            'stop_reasons': sorted({STOP_TEXT.get(r['stop_reason'], str(r['stop_reason'])) for r in group}),
            'parameter_count': group[0]['parameter_count'],
            'duration_seconds_mean': float(np.mean([r['duration_seconds'] for r in group])),
            'seed42_run_id': by_seed[min(by_seed, key=lambda s: (s != 42, s))]['run_id']})
    return entries


def _relevant_ids(stage_number, stage):
    """Config ids whose curves are shown: decision head, ranking head (+ control)."""
    if stage_number == 0:  # Best learning rate of each activation, winner first.
        by_run = {r['run_id']: r['config_id'] for r in stage['runs']}
        ids = [by_run[run_id] for run_id in stage['decision']['best_per_activation'].values()]
        return sorted(ids, key=lambda i: i != stage['winner']['config_id'])
    if stage_number == 1:
        return [r['config_id'] for r in stage['best'].values()]
    if 'decision' in stage:
        return stage['decision']['order'][:2]
    ids = [s['config_id'] for s in stage['ranking'] if s['status'] == 'completed'][:3]
    control = stage.get('control_config_id')
    return ids + ([control] if control and control not in ids else [])


def _fmt(value, digits=4):
    return '—' if value is None else f'{value:.{digits}f}'


def _comparison_text(label, c):
    return (f'- {label}: ventaja {c["difference"] * 100:.2f} pp; 2σ = {c["threshold"] * 100:.2f} pp '
            f'(σ agrupado {c["sigma"] * 100:.2f} pp) → '
            f'{"supera" if c["significant"] else "no supera"} la regla de mejora.')


def _plot_rates(stage, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for i, (optimizer, rows) in enumerate(stage['runs'].items()):
        done = sorted((r for r in rows if r['status'] == 'completed'), key=lambda r: r['learning_rate'])
        ax.plot([r['learning_rate'] for r in done], [r['accuracy'] for r in done], marker='o',
                color=f'C{i}', label=optimizer)
        added = {x['added_learning_rate'] for x in stage['extensions'] if x['optimizer'] == optimizer}
        hits = [r for r in done if r['learning_rate'] in added]
        ax.scatter([r['learning_rate'] for r in hits], [r['accuracy'] for r in hits], s=140,
                   facecolors='none', edgecolors=f'C{i}')
    ax.set(xscale='log', xlabel='Learning rate', ylabel='Accuracy de validación',
           title='Etapa 1: tasa por optimizador (círculos: extensiones de borde)')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def generate_v2(results: Path, output: Path):
    """Per-stage tables and figures of the v2 protocol from stage files and run reports."""
    results, output = Path(results).resolve(), Path(output).resolve()
    stages = {n: json.loads((results / f'stage_{n}.json').read_text(encoding='utf-8'))
              for n in range(8) if (results / f'stage_{n}.json').exists()}
    if not stages:
        raise ValueError('No v2 stage files found.')
    output.mkdir(parents=True, exist_ok=True)

    def report(run_id):
        return json.loads((results / 'runs' / run_id / 'results.json').read_text(encoding='utf-8'))

    lines = ['# Búsqueda v2 — un factor a la vez (desarrollo, ejercicio 2)', '',
             'Partición fija de `digits.csv` (semilla 42, validación 20 %); `digits_test.csv` reservado.',
             'Generado sólo desde artefactos guardados. Accuracy del mejor checkpoint (mínima',
             'cross-entropy de validación). Con varias semillas: media ± desvío muestral (ddof=1).', '']
    for n, stage in stages.items():
        name = f'stage_{n}'
        entries = _display_groups(_stage_rows(stage))
        with (output / f'{name}.csv').open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(entries[0]))
            writer.writeheader()
            writer.writerows(entries)
        shown = [e for e in entries if e['accuracy_mean'] is not None]
        plot_accuracy_bars([e['label'] for e in shown], [e['accuracy_mean'] for e in shown],
                           [e['accuracy_std'] for e in shown], output / f'{name}_accuracy.png', STAGE_TITLES[n])
        lines += [f'## {STAGE_TITLES[n]}', '',
                  '| Candidato | config_id | Semillas | Accuracy | Cross-entropy | Mejor época | Épocas | Parada | Parámetros | s/corrida |',
                  '|---|---|---|---:|---:|---|---|---|---:|---:|']
        for e in entries:
            accuracy = _fmt(e['accuracy_mean'])
            if e['accuracy_std'] is not None:
                accuracy += f' ± {e["accuracy_std"]:.4f}'
            if e['completed'] < e['runs']:
                accuracy += f' ({e["runs"] - e["completed"]} fallida/s)'
            lines.append(f'| {e["label"]} | `{e["config_id"]}` | {e["seeds"]} | {accuracy} | {_fmt(e["loss_mean"])} '
                         f'| {e["chosen_epochs"]} | {e["epochs_run"]} | {", ".join(e["stop_reasons"])} '
                         f'| {e["parameter_count"]} | {e["duration_seconds_mean"]:.1f} |')
        lines += ['', f'![Accuracy]({name}_accuracy.png)', '']
        if n == 0:
            decision = stage['decision']
            difference = decision.get('difference')
            lines += [f'Decisión: {stage["winner"]["label"]} — {decision["reason"]}'
                      + (f' (diferencia {difference * 100:.2f} pp).' if difference is not None else '.'), '']
        if n == 1:
            _plot_rates(stage, output / 'stage_1_rates.png')
            lines += ['![Tasas](stage_1_rates.png)', '', '| Optimizador | Grilla probada | Mejor tasa |', '|---|---|---:|']
            for optimizer, grid in stage['grids'].items():
                best = stage['best'].get(optimizer)
                lines.append(f'| {optimizer} | {grid} | {best["learning_rate"] if best else "—"} |')
            lines += ['', 'Extensiones de borde:', '']
            lines += [f'- Ronda {x["round"]}: {x["optimizer"]}, mejor {x["best_learning_rate"]:g} en el borde '
                      f'{"superior" if x["direction"] == "up" else "inferior"} → se agrega {x["added_learning_rate"]:g}.'
                      for x in stage['extensions']] or ['- Ninguna.']
            lines += [f'- Sin resolver: {k}: {v}.' for k, v in stage['unresolved_borders'].items()]
            lines.append('')
        near = (stage.get('decision') or {}).get('near_tie')
        if near:
            lines += [f'Empate cercano (diferencia {near["difference"] * 100:.2f} pp): se agregaron las semillas 0 y 1. '
                      f'Resultado: {near["reason"]}.', '']
            if near.get('comparison'):
                lines += [_comparison_text('Regla de mejora', near['comparison']), '']
        comparisons = dict(stage.get('evidence') or {})
        if stage.get('top_vs_runner_up'):
            comparisons['primero vs. segundo'] = stage['top_vs_runner_up']
        lines += [_comparison_text(label, c) for label, c in comparisons.items()]
        by_id = {e['config_id']: e for e in entries}
        chosen = [by_id[i] for i in _relevant_ids(n, stage) if i in by_id]
        plot_run_curves([report(e['seed42_run_id']) for e in chosen], [e['label'] for e in chosen],
                        output / f'{name}_curves.png', STAGE_TITLES[n] + ' (semilla 42)')
        lines += ['', f'![Curvas train/validación]({name}_curves.png)', '']
    selection_path = results / 'selection.json'
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding='utf-8'))
        run_id = Path(selection['candidate_model']).parent.name
        winner = report(run_id)
        generate(results / 'runs' / run_id / 'results.json', output / 'winner')
        lines += ['## Ganador final (semilla 42, mejor checkpoint)', '',
                  f'`{selection["config_id"]}`: época {selection["candidate_epoch"]}; reentrenamiento eventual '
                  f'de {selection["retraining_epochs"]} épocas.', '',
                  '| Dígito | Soporte | Precision | Recall | F1 |', '|---|---:|---:|---:|---:|']
        for row in winner['metrics']['validation']['per_class']:
            mark = '**' if row['class'] in (5, 8) else ''
            recall = 'N/E (sin muestras)' if row['recall'] is None else f'{row["recall"]:.4f}'
            lines.append(f'| {mark}{row["class"]}{mark} | {row["support"]} | {_fmt(row["precision"])} '
                         f'| {recall} | {_fmt(row["f1"])} |')
        lines += ['', 'El 8 no tiene muestras en la validación de `digits.csv`: su recall no es evaluable.', '',
                  '![Curvas](winner/learning_curves.png)', '', '![Confusión](winner/confusion_normalized.png)', '']
    (output / 'comparison_v2.md').write_text('\n'.join(lines), encoding='utf-8')
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--protocol', choices=('v1', 'v2'), default='v2')
    args = parser.parse_args()
    (generate_v2 if args.protocol == 'v2' else generate_study)(args.results_dir, args.output_dir)


if __name__ == '__main__':
    main()
