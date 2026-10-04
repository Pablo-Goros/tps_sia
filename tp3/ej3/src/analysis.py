"""Generate ej3 tables and plots exclusively from saved development results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tps_sia.tp3.shared.analysis import generate, plot_accuracy_bars, plot_run_curves

TP3 = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_table(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def accuracy_without_eight(metrics):
    matrix = np.asarray(metrics['confusion_matrix'])
    count = int(matrix.sum() - matrix[8].sum())
    return float((matrix.trace() - matrix[8, 8]) / count) if count else None


def per_class_rows(report):
    if report['status'] != 'completed':
        return []
    return [{'run_id': report['run_id'], 'seed': report['config']['model_seed'],
             'highlight': row['class'] in (5, 8), **row}
            for row in report['metrics']['validation']['per_class']]


def comparison_row(report, label):
    config = report['config']
    row = {'label': label, 'run_id': report['run_id'], 'config_id': report['config_id'],
           'seed': config['model_seed'], 'status': report['status'],
           'stop_reason': report['stop_reason'], 'chosen_epoch': report['chosen_epoch'],
           'epochs': report['history']['epocas_corridas'],
           'architecture': json.dumps(config['architecture']), 'activation': config['activation'],
           'optimizer': config['optimizer']['name'],
           'learning_rate': config['optimizer']['learning_rate'], 'batch_size': config['batch_size'],
           'parameters': report['parameter_count'], 'duration_seconds': report['duration_seconds'],
           'train_samples': report['dataset']['train_samples'],
           'validation_samples': report['dataset']['validation_samples']}
    if report['status'] == 'completed':
        metrics = report['metrics']['validation']
        row.update(accuracy=metrics['accuracy'], loss=metrics['loss'],
                   macro_f1=metrics['macro_f1'], balanced_accuracy=metrics['balanced_accuracy'],
                   accuracy_without_8=accuracy_without_eight(metrics),
                   recall_5=metrics['per_class'][5]['recall'], recall_8=metrics['per_class'][8]['recall'])
    return row


def summarize_labels(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row['label'], []).append(row)
    summaries = []
    for label, group in groups.items():
        if len({row['seed'] for row in group}) != len(group):
            raise ValueError(f'Duplicate seed in comparison label {label}.')
        entry = {'label': label, 'runs': len(group), 'seeds': json.dumps([r['seed'] for r in group]),
                 'status': 'completed' if all(r['status'] == 'completed' for r in group) else 'incomplete'}
        if entry['status'] == 'completed':
            for key in ('accuracy', 'loss', 'macro_f1', 'balanced_accuracy', 'accuracy_without_8',
                        'recall_5', 'recall_8', 'duration_seconds'):
                values = [r[key] for r in group if r[key] is not None]
                entry[key + '_mean'] = float(np.mean(values)) if values else None
                entry[key + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summaries.append(entry)
    return summaries


def paired_differences(rows):
    """Factor comparisons minus the first variant, paired by model seed."""
    labels = list(dict.fromkeys(row['label'] for row in rows))
    if not labels:
        return []
    groups = {label: {r['seed']: r for r in rows if r['label'] == label} for label in labels}
    control = groups[labels[0]]
    result = []
    for label in labels[1:]:
        group = groups[label]
        if set(control) != set(group):
            raise ValueError('Factor comparisons require identical seeds.')
        entry = {'control': labels[0], 'variant': label, 'seeds': json.dumps(sorted(control)),
                 'status': 'completed' if all(r['status'] == 'completed'
                                             for r in [*control.values(), *group.values()]) else 'incomplete'}
        if entry['status'] == 'completed':
            for key in ('accuracy', 'loss', 'accuracy_without_8', 'recall_5', 'recall_8'):
                differences = [group[s][key] - control[s][key] for s in sorted(control)
                               if group[s][key] is not None and control[s][key] is not None]
                entry[key + '_difference_mean'] = float(np.mean(differences)) if differences else None
                entry[key + '_difference_std'] = (float(np.std(differences, ddof=1))
                                                  if len(differences) > 1 else None)
        result.append(entry)
    return result


def render_group(value, source, destination, plots=True):
    destination.mkdir(parents=True, exist_ok=True)
    reports, rows, class_rows = [], [], []
    for row in value['runs']:
        path = source / 'runs' / row['run_id'] / 'results.json'
        report = read_json(path)
        if report['run_id'] != row['run_id'] or report['config_id'] != row['config_id']:
            raise ValueError('Stored summary and run report disagree.')
        expected_identity = value['identity']
        if report['metadata'] != expected_identity:
            raise ValueError('Run report belongs to another protocol.')
        reports.append(report)
        rows.append(comparison_row(report, row['label']))
        class_rows.extend({'label': row['label'], **r} for r in per_class_rows(report))
        if plots:
            generate(path, destination / 'runs' / row['run_id'])
    summary = summarize_labels(rows)
    write_table(destination / 'comparison.csv', rows)
    write_table(destination / 'seed_summary.csv', summary)
    write_table(destination / 'per_class.csv', class_rows)
    differences = paired_differences(rows) if 'part' in value else []
    write_table(destination / 'paired_differences.csv', differences)
    (destination / 'summary.json').write_text(json.dumps({
        'identity': value['identity'], 'groups': summary, 'paired_differences': differences,
        'data': value.get('data'), 'variants': value.get('variants'),
        'spread': 'sample standard deviation across seeds; null for one seed',
        'scope': 'development validation only'}, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        encoding='utf-8')
    if plots:
        completed = [r for r in summary if r['status'] == 'completed']
        if completed:
            plot_accuracy_bars([r['label'] for r in completed], [r['accuracy_mean'] for r in completed],
                               [r['accuracy_std'] for r in completed], destination / 'accuracy.png',
                               title='Comparación en validación')
        # Show a common seed across every variant, rather than different seeds per curve.
        seed = value['identity'].get('primary_seed')
        if seed is None and rows:
            seed = rows[0]['seed']
        curves = [(report, row['label']) for report, row in zip(reports, rows)
                  if row['seed'] == seed]
        if curves:
            plot_run_curves([r for r, label in curves], [label for r, label in curves],
                            destination / 'learning_curves.png', 'Train y validación — semilla común')
    lines = ['# Resultados de desarrollo', '',
             'Tablas generadas desde artefactos guardados. El desvío usa muestras de semillas; '
             'con una semilla figura N/E. No representa una evaluación final.', '',
             '| Variante | Estado | Accuracy media | Desvío |', '|---|---|---:|---:|']
    for row in summary:
        mean, std = row.get('accuracy_mean'), row.get('accuracy_std')
        lines.append(f"| {row['label']} | {row['status']} | "
                     f"{f'{mean:.4%}' if mean is not None else 'N/E'} | "
                     f"{f'{std:.4%}' if std is not None else 'N/E'} |")
    if 'data' in value:
        lines.extend(['', '## Condiciones de comparación', ''])
        lines.extend('- ' + limitation for limitation in value['data']['limitations'])
    (destination / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


def analyze(results_dir, factor_dir, output_dir, plots=True):
    results_dir, factor_dir, output_dir = map(Path, (results_dir, factor_dir, output_dir))
    documents = [('search', results_dir, path) for path in sorted(results_dir.glob('stage_*.json'))]
    documents += [('search', results_dir, path) for path in sorted(results_dir.glob('custom-*.json'))]
    documents += [('factors', factor_dir, factor_dir / f'{part}.json')
                  for part in ('dataset', 'coverage', 'size', 'techniques') if (factor_dir / f'{part}.json').exists()]
    if not documents:
        raise FileNotFoundError('No saved stages, custom runs or factor studies to analyze.')
    source_dirs = [source.resolve() for kind, source, path in documents]
    if output_dir.resolve() in source_dirs:
        raise ValueError('Analysis must use a separate output directory.')
    for kind, source, path in documents:
        render_group(read_json(path), source, output_dir / kind / path.stem, plots)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=TP3 / 'ej3/results/search')
    parser.add_argument('--factor-dir', type=Path, default=TP3 / 'ej3/results/factors')
    parser.add_argument('--output-dir', type=Path, default=TP3 / 'ej3/results/analysis')
    parser.add_argument('--tables-only', action='store_true', help='No Matplotlib required')
    args = parser.parse_args()
    analyze(args.results_dir, args.factor_dir, args.output_dir, not args.tables_only)
    print(f'Saved analysis in {args.output_dir}')


if __name__ == '__main__':
    main()
