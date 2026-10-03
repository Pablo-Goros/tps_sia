"""Tables and figures from saved results only; no dataset loading or training."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def load_reports(paths) -> list[dict]:
    return [json.loads(Path(path).read_text(encoding='utf-8')) for path in paths]


def comparison_rows(reports: list[dict]) -> list[dict]:
    """Failed and interrupted runs never enter performance comparisons."""
    rows = []
    for r in reports:
        if r.get('status') != 'completed':
            continue
        metrics = r['metrics']['validation']
        rows.append({'run_id': r['run_id'], 'config_id': r['config_id'],
                     'seed': r['config']['model_seed'], 'accuracy': metrics['accuracy'],
                     'loss': metrics['loss'], 'macro_f1': metrics['macro_f1'],
                     'balanced_accuracy': metrics['balanced_accuracy'],
                     'parameters': r['parameter_count'], 'duration_seconds': r['duration_seconds'],
                     'chosen_epoch': r['chosen_epoch'], 'epochs': r['history']['epocas_corridas'],
                     'stop_reason': r['stop_reason'], 'dataset_sha256': r['dataset']['sha256'],
                     'split_sha256': r['dataset']['split_sha256']})
    return sorted(rows, key=lambda row: (-row['accuracy'], row['loss'], row['parameters']))


def seed_summary(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault((row['config_id'], row['dataset_sha256'], row['split_sha256']), []).append(row)
    summary = []
    for (config_id, dataset_hash, split_hash), group in sorted(groups.items()):
        if len({r['seed'] for r in group}) != len(group):
            raise ValueError('Duplicate seed in the same configuration and partition.')
        entry = {'config_id': config_id, 'dataset_sha256': dataset_hash, 'split_sha256': split_hash,
                 'runs': len(group), 'seeds': [r['seed'] for r in group]}
        for key in ('accuracy', 'loss', 'macro_f1', 'balanced_accuracy', 'duration_seconds', 'parameters'):
            values = [r[key] for r in group]
            entry[key + '_mean'] = float(np.mean(values))
            entry[key + '_std'] = float(np.std(values))  # population spread; zero for one seed
        summary.append(entry)
    return summary


def _write_table(path: Path, rows: list[dict]):
    with path.open('w', encoding='utf-8', newline='') as file:
        if rows:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def generate_comparison(paths, output_dir: str | Path) -> list[dict]:
    reports = load_reports(paths)
    rows = comparison_rows(reports)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_table(output_dir / 'comparison.csv', rows)
    summary = seed_summary(rows)
    _write_table(output_dir / 'seed_summary.csv', summary)
    (output_dir / 'seed_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (output_dir / 'run_statuses.json').write_text(json.dumps(
        [{'run_id': r.get('run_id'), 'status': r.get('status'), 'stop_reason': r.get('stop_reason')}
         for r in reports], indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return rows


def generate(results_path: str | Path, output_dir: str | Path | None = None) -> None:
    """Render both legacy baseline and current runner artifacts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    results_path = Path(results_path)
    report = load_reports([results_path])[0]
    directory = Path(output_dir) if output_dir is not None else results_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    history = report['history']
    epochs = history.get('epochs') or list(range(1, history['epocas_corridas'] + 1))
    val_epochs = history.get('validation_epochs') or epochs[:len(history['costo_validacion'])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, key, val_key, label in (
            (axes[0], 'costo', 'costo_validacion', 'Costo'),
            (axes[1], 'accuracy', 'accuracy_validacion', 'Accuracy')):
        ax.plot(epochs, history[key], label='Train')
        ax.plot(val_epochs, history[val_key], label='Validación')
        if report.get('chosen_epoch') is not None:
            ax.axvline(report['chosen_epoch'], color='gray', linestyle=':', label='Época elegida')
        ax.set(xlabel='Época', ylabel=label)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[1].set_ylim(0, 1)
    fig.suptitle(f"{report['config']['architecture']} · {report['config']['optimizer']}")
    fig.savefig(directory / 'learning_curves.png', dpi=160)
    plt.close(fig)
    if 'validation_confusion_matrix' in report:
        matrix = np.asarray(report['validation_confusion_matrix'])
    elif report.get('metrics'):
        matrix = np.asarray(report['metrics']['validation']['confusion_matrix'])
    else:
        return  # Interrupted/failed histories can be inspected, never ranked.
    supports = matrix.sum(axis=1)
    normalized = np.divide(matrix, supports[:, None], out=np.full((10, 10), np.nan),
                           where=supports[:, None] != 0)
    for values, filename, label in ((matrix, 'confusion_matrix.png', 'Muestras'),
                                     (normalized, 'confusion_normalized.png', 'Proporción por clase real')):
        fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
        heatmap = ax.imshow(np.ma.masked_invalid(values), cmap='Blues', vmin=0,
                            vmax=1 if values is normalized else None)
        for row in range(10):
            for col in range(10):
                value = values[row, col]
                text = 'N/E' if not np.isfinite(value) else f'{value:.2f}' if values is normalized else str(int(value))
                ax.text(col, row, text, ha='center', va='center', fontsize=8,
                        color='white' if np.isfinite(value) and value > np.nanmax(values) / 2 else 'black')
        ax.set(xticks=range(10), yticks=range(10), xlabel='Dígito predicho', ylabel='Dígito real',
               yticklabels=[str(i) if supports[i] else f'{i} (sin muestras)' for i in range(10)], title=label)
        fig.colorbar(heatmap, ax=ax)
        fig.savefig(directory / filename, dpi=160)
        plt.close(fig)
    weight_path = results_path.parent / report.get('artifacts', {}).get('weights', 'weights.jsonl')
    if weight_path.exists():
        records = [json.loads(line) for line in weight_path.read_text(encoding='utf-8').splitlines() if line]
        if records:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
            for layer in range(len(records[0]['layers'])):
                for ax, key in zip(axes, ('weight_norm', 'gradient_norm', 'update_norm')):
                    ax.plot([r['update'] for r in records], [r['layers'][layer][key] for r in records], label=f'Capa {layer}')
                    ax.set(xlabel='Actualización', ylabel=key)
                    ax.legend()
            fig.savefig(directory / 'weight_norms.png', dpi=160)
            plt.close(fig)
            selected = records[0]['selected_weights']
            if selected:
                fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
                for i, item in enumerate(selected):
                    for ax, key in zip(axes, ('value', 'update')):
                        ax.plot([r['update'] for r in records], [r['selected_weights'][i][key] for r in records], label=str(item['index']))
                        ax.set(xlabel='Actualización', ylabel=key)
                        ax.legend()
                fig.savefig(directory / 'selected_weights.png', dpi=160)
                plt.close(fig)
