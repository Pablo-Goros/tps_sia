"""Final evaluation of the ej2 selection on digits_test.csv: run once, at the end.

Loads the candidate checkpoint named by selection.json (no retraining), checks its
SHA-256, evaluates it on digits_test.csv with the shared loader and writes
final_evaluation.json/.md, confusion_matrix.png and per_class_recall.png next to
selection.json. The test set is used for nothing else: no threshold, no selection.
An existing final_evaluation.json is never replaced unless --overwrite is given.

    python -m tps_sia.tp3.ej2.src.final_evaluation [--overwrite]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tps_sia.tp3.shared.digit_dataset import cargar
from tps_sia.tp3.shared.experiments import portable_path, preprocess, sha256_file, write_json
from tps_sia.tp3.shared.metrics import classification_metrics
from tps_sia.tp3.shared.mlp import MLP

TP3 = Path(__file__).resolve().parents[2]
V2 = TP3 / 'ej2' / 'results' / 'v2'
DIGITS_TEST = TP3 / 'data' / 'digits_test.csv'
CACHE = TP3 / 'ej2' / 'cache'
UNSEEN = 8  # digits.csv has no 8: the model never saw this class in training.
COLORS = {'bar': '#2a78d6', 'ink': '#0b0b0b', 'ink_2': '#52514e', 'muted': '#898781',
          'surface': '#fcfcfb', 'ramp': ['#fcfcfb', '#cde2fb', '#86b6ef', '#2a78d6', '#1c5cab', '#0d366b']}


def evaluate_predictions(actual: np.ndarray, predicted: np.ndarray, unseen: int = UNSEEN) -> dict:
    """Global accuracy, accuracy without the unseen class and per-class metrics."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    metrics = classification_metrics(actual, predicted)
    matrix = np.asarray(metrics['confusion_matrix'])
    seen = actual != unseen
    return {
        'accuracy_all': metrics['accuracy'],
        'samples_all': int(actual.size),
        'accuracy_without_unseen': float(np.mean(predicted[seen] == actual[seen])) if seen.any() else None,
        'samples_without_unseen': int(seen.sum()),
        'unseen_class': unseen,
        'unseen_samples': int((~seen).sum()),
        'predictions_of_unseen': int(np.sum(predicted == unseen)),
        'unseen_assigned_to': {str(d): int(c) for d, c in enumerate(matrix[unseen]) if c},
        'per_class': metrics['per_class'],
        'macro_f1': metrics['macro_f1'],
        'balanced_accuracy': metrics['balanced_accuracy'],
        'undefined_policy': metrics['undefined_policy'],
        'confusion_matrix': matrix.tolist(),
        'confusion_matrix_row_normalized': [(row / row.sum()).tolist() if row.sum() else None for row in matrix],
        'confusion_axes': metrics['confusion_axes'],
    }


def validation_reference(results_dir: Path, selection: dict, candidate: Path) -> dict:
    """Validation accuracy of the same checkpoint and of the selected configuration."""
    run = json.loads((candidate.parent / 'results.json').read_text(encoding='utf-8'))
    stage_7 = json.loads((results_dir / 'stage_7.json').read_text(encoding='utf-8'))
    group = next(g for g in stage_7['ranking'] if g['config_id'] == selection['config_id'])
    return {'candidate_seed': selection['delivery_seed'], 'candidate_epoch': run['chosen_epoch'],
            'candidate_accuracy': run['metrics']['validation']['accuracy'],
            'candidate_cross_entropy': run['metrics']['validation']['loss'],
            'validation_samples': selection['dataset']['validation_samples'],
            'seeds': group['seeds'], 'accuracy_mean': group['accuracy_mean'],
            'accuracy_std': group['accuracy_std']}


def pct(value, digits=2):
    return 'indefinida' if value is None else f'{100 * value:.{digits}f}'


def markdown(report: dict) -> str:
    r, v, s = report['results'], report['validation'], report['selection']
    unseen = r['unseen_class']
    lines = [
        '# Evaluación final — ejercicio 2', '',
        f"Modelo: `{s['candidate_model']}` (configuración `{s['config_id']}`, semilla {v['candidate_seed']}, "
        f"época {v['candidate_epoch']}; SHA-256 `{s['model_sha256'][:16]}…` verificado contra `selection.json`). "
        f"Sin reentrenar.",
        f"Test: `{report['test']['path']}`, {r['samples_all']} muestras (SHA-256 `{report['test']['sha256'][:16]}…`). "
        f"Evaluado una sola vez; el test no se usó para ninguna selección ni umbral.", '',
        '## Accuracy', '',
        '| Conjunto | Muestras | Accuracy (%) | Cross-entropy |', '|---|---:|---:|---:|',
        f"| Test, todas las clases | {r['samples_all']} | **{pct(r['accuracy_all'])}** | {r['cross_entropy_all']:.4f} |",
        f"| Test, sin el {unseen} (clases vistas en entrenamiento) | {r['samples_without_unseen']} | "
        f"**{pct(r['accuracy_without_unseen'])}** | {r['cross_entropy_without_unseen']:.4f} |",
        f"| Validación, mismo modelo (semilla {v['candidate_seed']}) | {v['validation_samples']} | "
        f"{pct(v['candidate_accuracy'])} | {v['candidate_cross_entropy']:.4f} |",
        f"| Validación, misma configuración, media de {len(v['seeds'])} semillas | {v['validation_samples']} | "
        f"{pct(v['accuracy_mean'])} ± {pct(v['accuracy_std'])} | — |", '',
        f"`digits.csv` no contiene ningún {unseen}: el modelo nunca vio esa clase y, salvo por azar, no puede "
        f"predecirla. Cada {unseen} del test es entonces un error estructural, que la accuracy global incluye. "
        f"La accuracy sin el {unseen} mide el desempeño sobre las clases que el modelo pudo aprender y es la "
        f"comparable con validación, que tampoco tiene ningún {unseen}. Las dos se informan juntas: la global es "
        f"el resultado real sobre el test; la otra separa el efecto de la clase ausente.", '',
        f"## El dígito {unseen}", '',
        f"- Muestras con etiqueta {unseen} en test: {r['unseen_samples']}.",
        f"- Predicciones iguales a {unseen}: {r['predictions_of_unseen']} (esperado: 0).",
        f"- Los {unseen} del test se asignaron a: " + ', '.join(
            f"{d} ({c})" for d, c in sorted(r['unseen_assigned_to'].items(), key=lambda kv: -kv[1])) + '.', '',
        '## Métricas por clase', '',
        '| Dígito | Muestras | Predicciones | Precision (%) | Recall (%) | F1 (%) |', '|---|---:|---:|---:|---:|---:|',
    ]
    for c in r['per_class']:
        lines.append(f"| {c['class']} | {c['support']} | {c['predictions']} | {pct(c['precision'])} | "
                     f"{pct(c['recall'])} | {pct(c['f1'])} |")
    lines += ['', f"Para el {unseen}: recall 0 porque ninguna de sus muestras se clasificó como {unseen}; "
                  f"precision indefinida porque no hubo predicciones de {unseen} (0/0). "
                  f"Macro-F1 {pct(r['macro_f1'])} % y balanced accuracy {pct(r['balanced_accuracy'])} %, "
                  f"promediadas sobre las clases presentes en test.", '',
              '## Matriz de confusión', '', 'Filas: dígito real; columnas: dígito predicho (conteos).', '',
              '| Real \\ Pred. | ' + ' | '.join(map(str, range(10))) + ' |', '|---|' + '---:|' * 10]
    for d, row in enumerate(r['confusion_matrix']):
        lines.append(f'| **{d}** | ' + ' | '.join(str(x) for x in row) + ' |')
    lines += ['', 'Normalizada por fila (% de cada dígito real):', '',
              '| Real \\ Pred. | ' + ' | '.join(map(str, range(10))) + ' |', '|---|' + '---:|' * 10]
    for d, row in enumerate(r['confusion_matrix_row_normalized']):
        cells = ['—'] * 10 if row is None else [f'{100 * x:.1f}' if x else '0' for x in row]
        lines.append(f'| **{d}** | ' + ' | '.join(cells) + ' |')
    lines += ['', '![Matriz de confusión](confusion_matrix.png)', '', '![Recall por clase](per_class_recall.png)', '']
    return '\n'.join(lines)


def _style(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(COLORS['muted'])
    ax.tick_params(colors=COLORS['ink_2'])


def plot_confusion(results: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ramp = LinearSegmentedColormap.from_list('blue', COLORS['ramp'])
    raw = np.asarray(results['confusion_matrix'], dtype=float)
    normalized = np.array([np.zeros(10) if row is None else row for row in results['confusion_matrix_row_normalized']])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True, facecolor=COLORS['surface'])
    for ax, data, title, fmt in ((axes[0], raw, 'Conteos', lambda x: f'{int(x)}'),
                                 (axes[1], normalized, 'Normalizada por fila (%)', lambda x: f'{100 * x:.0f}' if x >= 0.01 else '<1')):
        image = ax.imshow(data, cmap=ramp, vmin=0)
        for i in range(10):
            for j in range(10):
                if data[i, j]:
                    dark = data[i, j] > 0.55 * data.max()
                    ax.text(j, i, fmt(data[i, j]), ha='center', va='center', fontsize=7,
                            color=COLORS['surface'] if dark else COLORS['ink'])
        ax.set(xticks=range(10), yticks=range(10), xlabel='Dígito predicho', ylabel='Dígito real')
        ax.set_title(title, color=COLORS['ink'], loc='left')
        ax.tick_params(colors=COLORS['ink_2'])
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle('Matriz de confusión sobre digits_test.csv', color=COLORS['ink'])
    fig.savefig(path, dpi=160, facecolor=COLORS['surface'])
    plt.close(fig)


def plot_recall(results: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    recall = [0 if c['recall'] is None else 100 * c['recall'] for c in results['per_class']]
    fig, ax = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True, facecolor=COLORS['surface'])
    ax.set_facecolor(COLORS['surface'])
    ax.bar(range(10), recall, width=0.6, color=COLORS['bar'])
    unseen = results['unseen_class']
    ax.annotate(f'{unseen}: no visto en\nentrenamiento (recall 0)', xy=(unseen, 0), xytext=(unseen, 25),
                ha='center', fontsize=8, color=COLORS['ink_2'],
                arrowprops={'arrowstyle': '-', 'color': COLORS['muted']})
    for d, value in enumerate(recall):
        if d != unseen:
            ax.text(d, value + 1, f'{value:.1f}', ha='center', fontsize=7, color=COLORS['ink_2'])
    ax.set(xticks=range(10), ylim=(0, 105), xlabel='Dígito', ylabel='Recall en test (%)')
    ax.set_title('Recall por clase sobre digits_test.csv', color=COLORS['ink'], loc='left')
    ax.grid(axis='y', color=COLORS['muted'], alpha=0.25)
    ax.set_axisbelow(True)
    _style(ax)
    fig.savefig(path, dpi=160, facecolor=COLORS['surface'])
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results-dir', type=Path, default=V2)
    parser.add_argument('--test-csv', type=Path, default=DIGITS_TEST)
    parser.add_argument('--cache-dir', type=Path, default=CACHE)
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace an existing final_evaluation.json (repeats the test evaluation).')
    args = parser.parse_args(argv)
    results_dir, test_csv = args.results_dir.resolve(), args.test_csv.resolve()
    output = results_dir / 'final_evaluation.json'
    if output.exists() and not args.overwrite:
        raise SystemExit(f'{output} already exists; the final evaluation is not repeated. '
                         'Use --overwrite only to deliberately evaluate again.')

    selection = json.loads((results_dir / 'selection.json').read_text(encoding='utf-8'))
    candidate = results_dir / selection['candidate_model']
    if sha256_file(candidate) != selection['model_sha256']:
        raise ValueError(f'{candidate} does not match the SHA-256 recorded in selection.json.')
    model = MLP.cargar(candidate)
    config = selection['config']
    if model.arquitectura != config['architecture'] or model.activacion.nombre != config['activation']:
        raise ValueError('The candidate model does not match the selected configuration.')
    validation = validation_reference(results_dir, selection, candidate)

    X, y = cargar(test_csv, args.cache_dir.resolve() / 'digits_test_final.npz')
    X = preprocess(X, model.preprocessing)
    actual = y.argmax(axis=1)
    probabilities = model.predecir(X)
    results = evaluate_predictions(actual, probabilities.argmax(axis=1))
    seen = actual != UNSEEN
    results['cross_entropy_all'] = float(model.costo(X, y))
    results['cross_entropy_without_unseen'] = float(model.costo(X[seen], y[seen]))

    try:
        test_path = portable_path(test_csv, TP3)
    except ValueError:
        test_path = test_csv.name
    report = {
        'schema_version': 1,
        'evaluated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'policy': 'Single final evaluation; the test set selects nothing and sets no threshold.',
        'selection': {k: selection[k] for k in ('config_id', 'candidate_model', 'model_sha256',
                                                'search_sha256', 'protocol')},
        'test': {'path': test_path, 'sha256': sha256_file(test_csv), 'rows': int(len(actual))},
        'validation': validation,
        'results': results,
        'artifacts': {'report': 'final_evaluation.md', 'confusion_matrix': 'confusion_matrix.png',
                      'per_class_recall': 'per_class_recall.png'},
    }
    plot_confusion(results, results_dir / 'confusion_matrix.png')
    plot_recall(results, results_dir / 'per_class_recall.png')
    (results_dir / 'final_evaluation.md').write_text(markdown(report), encoding='utf-8')
    write_json(output, report)  # Written last: its presence means a complete evaluation.
    print(f"Test accuracy: {100 * results['accuracy_all']:.2f}% (all), "
          f"{100 * results['accuracy_without_unseen']:.2f}% (without {UNSEEN}); report in {results_dir}")


if __name__ == '__main__':
    main()
