"""Audit and present the persisted architecture comparison without training."""
import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .autoencoder import Autoencoder
from .denoise_architecture_campaign import (DIMENSIONS, ENCODERS, TP5_ROOT,
                                             summarize, verify_completed)
from .denoise_campaign import read_json, sha256
from .denoise_evaluation import grouped_metrics
from .denoise_training import validation_score
from .train import write_json


LABELS = {'simple': 'Más simple', 'current': 'Actual', 'deeper': 'Más profunda'}


def audit(campaign):
    """Recalculate every selected checkpoint's validation, and verify frozen evidence."""
    campaign = Path(campaign)
    manifest, ledger, selection = verify_completed(campaign)
    with np.load(campaign/'collections/validation.npz', allow_pickle=False) as saved:
        validation = {key: saved[key][saved['levels'] != .3] for key in saved.files}
    checks = []
    for row in ledger['runs']:
        directory = campaign/'runs'/row['run_id']
        model = Autoencoder.load(campaign/row['best']['model']['path'])
        score = validation_score(grouped_metrics(validation, model.reconstruct(validation['inputs'])),
                                 manifest['noise']['selection_levels'])
        np.testing.assert_allclose(score, row['best']['score'], rtol=0, atol=1e-14)
        history = read_json(directory/'history.json')
        best = min(history['validation'], key=lambda point: (point['score'], point['epoch']))
        if (best['score'] != row['best']['score'] or best['epoch'] != row['best']['epoch']
                or model.history['epocas_corridas'] != best['epoch']
                or Autoencoder.load(campaign/row['last_model']['path']).history['epocas_corridas'] != 3000
                or history['training']['epocas_corridas'] != 3000):
            raise ValueError('History/checkpoint mismatch')
        checks.append({'run_id': row['run_id'], 'validation_recomputed': score,
                       'best_epoch': best['epoch'], 'reused': row['reused']})
    preserved = manifest['source']['files']
    source = TP5_ROOT/manifest['source']['path']
    for name, digest in preserved.items():
        if sha256(source/name) != digest:
            raise ValueError(f'Original campaign changed: {name}')
    metrics = read_json(campaign/'evaluation/metrics.json')
    if metrics['selection_sha256'] != sha256(campaign/'selection.json'):
        raise ValueError('Final evaluation and selection disagree')
    if metrics['collection_sha256'] != sha256(campaign/'evaluation/collection.npz'):
        raise ValueError('Final collection hash mismatch')
    for name, digest in metrics['prediction_artifacts'].items():
        if sha256(campaign/f'evaluation/{name}.npz') != digest:
            raise ValueError('Final predictions hash mismatch')
    if metrics['seed'] in [manifest['noise']['validation_seed'], manifest['noise']['evaluation_seed']]:
        raise ValueError('Final evaluation must use a fresh noise seed')
    result = {'status': 'passed', 'runs_verified': checks, 'new_runs': 33, 'reused_runs': 3,
              'new_epochs': ledger['new_epochs_completed'], 'represented_epochs': 108000,
              'original_files_preserved': len(preserved), 'selection_sha256': sha256(campaign/'selection.json'),
              'evaluation_sha256': sha256(campaign/'evaluation/metrics.json')}
    reload_path = campaign/'validation/reload-comparison.json'
    if reload_path.exists():
        repeated = read_json(reload_path)
        if (repeated['status'] != 'passed' or not repeated['metrics_equal']
                or repeated['first_metrics_sha256'] != result['evaluation_sha256']):
            raise ValueError('Reload comparison does not describe this evaluation')
        result['evaluation_reloaded'] = {key: repeated[key] for key in
                                         ['status', 'arrays_identical', 'csv_files_identical', 'metrics_equal']}
        result['reload_comparison_sha256'] = sha256(reload_path)
    test_log = campaign/'validation/tp5-tests.log'
    if test_log.exists():
        text = test_log.read_text(encoding='utf-8')
        match = re.search(r'Ran (\d+) tests? in', text)
        if match is None or not text.rstrip().endswith('OK'):
            raise ValueError('Test log does not record a passing suite')
        result['tests_passed'] = int(match.group(1))
        result['test_log_sha256'] = sha256(test_log)
    write_json(campaign/'verification.json', result)
    return result


def save(figure, path):
    figure.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(figure)


def figures(campaign, rows, manifest, selection, metrics):
    campaign = Path(campaign)
    output = campaign/'figures'
    output.mkdir(exist_ok=True)
    matrix = np.array([[next(r['mean_pixel_errors'] for r in rows if r['variant'] == f'{family}-d{d}')
                        for d in DIMENSIONS] for family in ENCODERS])
    figure, axis = plt.subplots(figsize=(8, 4))
    heat = axis.imshow(matrix, cmap='YlOrRd')
    axis.set_xticks(range(4), DIMENSIONS)
    axis.set_yticks(range(3), [LABELS[f] for f in ENCODERS])
    axis.set_xlabel('Ancho de la capa central (d)')
    axis.set_title('Validación: píxeles erróneos por imagen\nMedia de 3 semillas; niveles 0,05 / 0,10 / 0,20')
    for i in range(3):
        for j in range(4):
            axis.text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center',
                      color='white' if matrix[i,j] > (matrix.max()+matrix.min())/2 else 'black')
    figure.colorbar(heat, ax=axis, label='Menor es mejor')
    save(figure, output/'validation-heatmap.png')

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for axis, family in zip(axes, ENCODERS):
        group = sorted([r for r in rows if r['family'] == family], key=lambda r: r['dimension'])
        means = np.array([r['mean_pixel_errors'] for r in group])
        axis.errorbar(DIMENSIONS, means, yerr=[means-[r['min'] for r in group],
                                               [r['max'] for r in group]-means], capsize=5, marker='o')
        for seed, marker in zip(manifest['config']['seeds'], ['x', '+', '.']):
            axis.scatter(DIMENSIONS, [next(r['best']['score'][0] for r in selection['runs']
                                         if r['variant'] == f'{family}-d{d}' and r['seed'] == seed)
                                      for d in DIMENSIONS], marker=marker, label=f'Semilla {seed}')
        axis.set(title=LABELS[family], xlabel='Ancho central d', xticks=DIMENSIONS)
        axis.grid(alpha=.2)
    axes[0].set_ylabel('Error de validación (píxeles / imagen)')
    axes[-1].legend(fontsize=8)
    figure.suptitle('Ancho central y profundidad: media y rango de tres semillas')
    figure.tight_layout()
    save(figure, output/'validation-by-dimension.png')

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    levels = manifest['noise']['evaluation_levels']
    for variant, label, color in [('current-d10', 'Referencia actual, d=10', 'tab:blue'),
                                  (selection['variant'], f'Seleccionada: {selection["variant"]}', 'tab:green')]:
        if variant == 'current-d10' and selection['variant'] == 'current-d10':
            continue
        group = [next(r for r in metrics['seed_aggregates'] if r['variant'] == variant and r['level'] == level)
                 for level in levels]
        for axis, key in zip(axes, ['mean_pixel_errors', 'exact_fraction']):
            axis.plot(levels, [r[f'{key}_mean'] for r in group], marker='o', label=label, color=color)
            axis.fill_between(levels, [r[f'{key}_min'] for r in group],
                              [r[f'{key}_max'] for r in group], color=color, alpha=.15)
    for name, label, color in [('noisy-input', 'Entrada ruidosa', '0.5'),
                               ('ae-reference', 'AE de 1.a', 'tab:orange')]:
        group = [next(r for r in metrics['by_level'] if r['model'] == name and r['level'] == level)
                 for level in levels]
        for axis, key in zip(axes, ['mean_pixel_errors', 'exact_fraction']):
            axis.plot(levels, [r[key] for r in group], linestyle='--', label=label, color=color)
    for axis, ylabel in zip(axes, ['Píxeles erróneos por imagen', 'Fracción de recuperación exacta']):
        axis.set(xlabel='Probabilidad de invertir cada píxel', ylabel=ylabel, xticks=levels)
        axis.axvspan(.2, .3, color='0.9', alpha=.5)
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    figure.suptitle('Evaluación independiente (ruido nuevo); bandas: rango entre semillas')
    figure.tight_layout()
    save(figure, output/'recovery-selected-vs-current.png')

    reference = min((r for r in selection['runs'] if r['variant'] == 'current-d10'),
                    key=lambda r: (r['best']['score'], r['seed']))['run_id']
    with np.load(campaign/'evaluation/collection.npz', allow_pickle=False) as saved:
        collection = {key: saved[key] for key in saved.files}
    predictions = {}
    for name in [reference, selection['run_id']]:
        with np.load(campaign/f'evaluation/{name}.npz', allow_pickle=False) as saved:
            predictions[name] = saved['predictions']
    # The example policy was frozen before inspecting the final evaluation.
    for level in [0, .1, .2, .3]:
        indices = [i for i in metrics['example_indices'] if collection['levels'][i] == level]
        figure, axes = plt.subplots(6, 4, figsize=(7, 9))
        for row, index in enumerate(indices):
            images = [collection['targets'][index], collection['inputs'][index],
                      predictions[reference][index] >= .5, predictions[selection['run_id']][index] >= .5]
            for column, values in enumerate(images):
                axes[row, column].imshow(values.reshape(7, 5), cmap='gray_r', vmin=0, vmax=1,
                                         interpolation='nearest')
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
                if column:
                    errors = np.count_nonzero(values != collection['targets'][index])
                    axes[row, column].set_xlabel(f'{errors} errores', fontsize=8)
            axes[row, 0].set_ylabel(f'Código {int(collection["character_indices"][index])+96}')
        for axis, title in zip(axes[0], ['Limpio', 'Ruido', 'Actual d=10', selection['variant']]):
            axis.set_title(title, fontsize=9)
        figure.suptitle(f'Ejemplos fijados previamente, p={level:g}')
        figure.tight_layout()
        save(figure, output/f'samples-p{level:g}.png')
    write_json(output/'manifest.json', {
        'selection_sha256': sha256(campaign/'selection.json'),
        'evaluation_sha256': sha256(campaign/'evaluation/metrics.json'),
        'reference_run': reference, 'selected_run': selection['run_id'],
        'example_indices': metrics['example_indices'],
        'files': {p.name: sha256(p) for p in sorted(output.glob('*.png'))}})


def report(campaign):
    campaign = Path(campaign)
    manifest, ledger, selection = verify_completed(campaign)
    verification = audit(campaign)
    rows = summarize(campaign)
    metrics = read_json(campaign/'evaluation/metrics.json')
    figures(campaign, rows, manifest, selection, metrics)
    winner = next(r for r in rows if r['variant'] == selection['variant'])
    baseline = next(r for r in rows if r['variant'] == 'current-d10')
    relative_change = 100*(baseline['mean_pixel_errors']-winner['mean_pixel_errors'])/baseline['mean_pixel_errors']
    lines = ['# TP5 1.b — Comparación de arquitectura y ancho central', '',
             'Campaña autorizada el 2026-11-17. Diseño y criterio fijados en '
             '[el plan](../../denoising-architecture-plan.md) antes de entrenar.', '',
             '## Resultado de la selección', '',
             f'La variante seleccionada por validación es **`{selection["variant"]}`**, arquitectura '
             f'`{"→".join(map(str, manifest["variants"][selection["variant"]]["architecture"]))}`. '
             f'Su error medio entre semillas es **{winner["mean_pixel_errors"]:.4f}** píxeles por imagen, '
             f'frente a **{baseline["mean_pixel_errors"]:.4f}** de `current-d10` '
             f'({relative_change:.1f} % de reducción en esta validación). '
             f'Corrida representativa: `{selection["run_id"]}`.', '',
             'Estos puntajes promedian por igual los niveles 0,05/0,10/0,20. '
             'Cada semilla aporta su mejor checkpoint; no se escoge una arquitectura por su mejor semilla.', '',
             f'La seleccionada tiene menos capas, pero **{winner["parameters"]} parámetros**, '
             f'frente a {baseline["parameters"]} de la referencia: ampliar el código también '
             'modifica el tamaño del modelo.', '',
             '## Diseño y costo', '',
             '| Familia | Arquitectura |', '|---|---|']
    for family, encoder in ENCODERS.items():
        architecture = '→'.join(map(str, encoder))+'→d→'+'→'.join(map(str, encoder[::-1]))
        lines.append(f'| {LABELS[family]} (`{family}`) | `{architecture}` |')
    lines.extend(['', 'Dimensiones centrales: 5, 10, 15 y 20; semillas: 42, 43 y 44. '
                  'BCE, Adam 0,002, tanh interna, salida logística, lote 32 y 3000 épocas por corrida. '
                  'Ruido por época y validación conservan exactamente el protocolo anterior.', '',
                  f'Se completaron **36 combinaciones: 33 entrenamientos nuevos y 3 reutilizados** '
                  f'(`current-d10`). Consumo nuevo: **{ledger["new_epochs_completed"]:,} épocas**; '
                  f'108.000 representadas, incluidas las 9.000 históricas. '
                  f'Tiempo de la fase de entrenamiento: **{ledger["duration_seconds"]/60:.2f} minutos**, '
                  'incluyendo copia de corridas y guardado; no incluye preparación, evaluación ni gráficos. '
                  'No hubo búsqueda adicional fuera del diseño autorizado.', '',
                  '## Comparación de validación', '',
                  '| Variante | Parámetros | Error medio | Desvío entre semillas | Mín.–máx. | Épocas elegidas (42/43/44) |',
                  '|---|---:|---:|---:|---:|---|'])
    for row in rows:
        lines.append(f'| {row["variant"]} | {row["parameters"]} | {row["mean_pixel_errors"]:.4f} | '
                     f'{row["std_population"]:.4f} | {row["min"]:.4f}–{row["max"]:.4f} | {row["best_epochs"]} |')
    lines.extend(['', '![Error de validación por arquitectura y dimensión](figures/validation-heatmap.png)', '',
                  '![Variación entre semillas](figures/validation-by-dimension.png)', '',
                  '## Qué responde sobre las decisiones de arquitectura', ''])
    for family in ENCODERS:
        best = min((r for r in rows if r['family'] == family), key=lambda r: (r['mean_pixel_errors'], r['mean_mse']))
        lines.append(f'- Dentro de **{LABELS[family].lower()}**, el mejor ancho probado fue '
                     f'**{best["dimension"]}**, con {best["mean_pixel_errors"]:.4f} errores de validación.')
    fixed_ten = {family: next(r for r in rows if r['variant'] == f'{family}-d10')
                 for family in ENCODERS}
    at_budget = sum(r['best']['epoch'] == 3000 for r in ledger['runs'])
    lines.extend(['', 'Manteniendo el código en **10** para comparar con la referencia, '
                  f'la red simple obtiene {fixed_ten["simple"]["mean_pixel_errors"]:.4f}, '
                  f'la actual {fixed_ten["current"]["mean_pixel_errors"]:.4f} y '
                  f'la más profunda {fixed_ten["deeper"]["mean_pixel_errors"]:.4f} '
                  'errores por imagen. Esto compara profundidad a igual ancho central, '
                  'aunque no iguala el número de parámetros.', '',
                  f'En **{at_budget}/36 corridas** el mejor checkpoint es el de la época 3000, '
                  'el límite del presupuesto. Por eso el resultado describe rendimiento con '
                  'este número de actualizaciones; no demuestra que todas las redes hayan '
                  'agotado su capacidad de aprendizaje.'])
    lines.extend(['', 'La búsqueda permite justificar una elección entre estas 12 variantes bajo el '
                  'presupuesto de 3000 épocas. No prueba que la dimensión seleccionada sea óptima para '
                  'cualquier red, otro ruido o más entrenamiento. Profundidad y cantidad de parámetros '
                  'cambian conjuntamente. En `current` y `deeper` las capas vecinas siguen teniendo '
                  'ancho 10: ampliar el centro a 15/20 no elimina esa compresión previa. '
                  'Las tres semillas describen variación; no son una prueba de significancia estadística.', '',
                  '## Evaluación independiente después de seleccionar', '',
                  'Se generó una colección nueva con semilla **20261117**, después de congelar '
                  '`selection.json`. Contiene 12.832 ejemplos: los 32 limpios y 100 realizaciones por '
                  'carácter para cada nivel positivo. No participó en la selección. '
                  'Las tablas siguientes promedian las tres semillas; 0,30 está fuera del rango entrenado.', '',
                  '| Variante | Ruido p | Error medio | Recuperación exacta | Con ≤1 error |',
                  '|---|---:|---:|---:|---:|'])
    for variant in dict.fromkeys(['current-d10', selection['variant']]):
        for level in manifest['noise']['evaluation_levels']:
            row = next(r for r in metrics['seed_aggregates'] if r['variant'] == variant and r['level'] == level)
            lines.append(f'| {variant} | {level:g} | {row["mean_pixel_errors_mean"]:.4f} | '
                         f'{100*row["exact_fraction_mean"]:.2f} % | {100*row["within_tolerance_fraction_mean"]:.2f} % |')
    if selection['variant'] != 'current-d10':
        outside = {variant: next(r for r in metrics['seed_aggregates']
                                 if r['variant'] == variant and r['level'] == .3)
                   for variant in ['current-d10', selection['variant']]}
        old, new = outside['current-d10'], outside[selection['variant']]
        if new['exact_fraction_mean'] < old['exact_fraction_mean']:
            lines.extend(['', 'La mejora no es uniforme en todas las métricas: con **p=0,30**, '
                          f'la recuperación exacta de la seleccionada es '
                          f'**{100*new["exact_fraction_mean"]:.2f} %**, frente a '
                          f'**{100*old["exact_fraction_mean"]:.2f} %** de la referencia. '
                          'El criterio acordado prioriza error medio en los niveles entrenados, '
                          'no recuperación exacta fuera de rango.'])
    lines.extend(['', '![Recuperación con ruido nuevo](figures/recovery-selected-vs-current.png)', '',
                  'Las tablas de evaluación de las 12 variantes están en '
                  '[evaluation/seed-aggregates.csv](evaluation/seed-aggregates.csv). '
                  'No se cambia la selección si otra variante obtiene un mejor valor final. '
                  'Esto mide recuperación de caracteres conocidos ante ruido nuevo, no generalización '
                  'a caracteres desconocidos.', '',
                  'Los ejemplos usan siempre la primera realización de los caracteres 96, 97, 104, '
                  '112, 120 y 127, fijados antes de inspeccionar las imágenes.', '',
                  '![Ejemplos con 10 % de ruido](figures/samples-p0.1.png)', '',
                  '![Ejemplos con 20 % de ruido](figures/samples-p0.2.png)', '',
                  '## Verificaciones y reproducción', '',
                  f'Se recalculó la validación de los **{len(verification["runs_verified"])}** '
                  'mejores checkpoints desde sus modelos persistidos, se comprobaron configuraciones, '
                  'épocas, historias y hashes, y se verificó la selección por el criterio congelado. '
                  f'Los **{verification["original_files_preserved"]} archivos** de la campaña original '
                  'permanecen idénticos por SHA256. Los tres modelos reutilizados conservan '
                  'su procedencia y verificaron sus puntajes antes de entrenar.', '',
                  'Archivos principales: [protocolo](protocol.json), [manifiesto](manifest.json), '
                  '[selección](selection.json), [tabla por corrida](validation-runs.csv), '
                  '[tabla agregada](validation-aggregates.csv), [verificación](verification.json). '
                  'Los comandos para entrenar desde cero o recalcular evaluación sin entrenamiento '
                  'están en [el plan](../../denoising-architecture-plan.md). '
                  'La campaña conserva copias del código principal ejecutado en `implementation/`.', '',
                  'Fundamento: un DAE entrena pares de entrada contaminada y objetivo limpio '
                  '(OFF-022, p. 22; wiki de autoencoders del curso). La consigna PA-004 1.b pide '
                  'justificar una arquitectura y estudiar niveles de ruido. Las variantes y el '
                  'presupuesto de esta comparación son decisiones del grupo.'])
    if 'evaluation_reloaded' in verification:
        repeated = verification['evaluation_reloaded']
        lines.extend(['', 'La evaluación completa se repitió desde los checkpoints sin entrenar: '
                      f'**{repeated["arrays_identical"]} arrays**, las **tres tablas CSV** y las '
                      'métricas coinciden exactamente. Evidencia en '
                      '[validation/reload-comparison.json](validation/reload-comparison.json).'])
    if 'tests_passed' in verification:
        lines.extend(['', f'La suite del TP5 pasó **{verification["tests_passed"]} tests**, '
                      'incluidos los nuevos de diseño de grilla y rechazo de reutilización '
                      'incompatible o corrupta. [Log](validation/tp5-tests.log).'])
    (campaign/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    args = parser.parse_args()
    report(args.campaign)


if __name__ == '__main__':
    main()
