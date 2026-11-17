"""Continue the selected DAE to 6000 total epochs, preserving Adam and absolute noise epochs."""
import argparse
import copy
from pathlib import Path
import platform
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .autoencoder import Autoencoder
from .dataset import load_dataset
from .denoise_architecture_campaign import verify_completed
from .denoise_campaign import read_json, sha256
from .denoise_evaluation import grouped_metrics, write_csv
from .denoise_training import validation_score
from .noise import training_inputs, make_collection
from .train import write_json


FINAL_SEED = 20261118


def check_adam_state(path):
    with np.load(path, allow_pickle=False) as archive:
        config = read_archive_config(archive)
        state = config['optimizer_state']
        if (config['version'] < 3 or config['optimizador']['name'] != 'adam'
                or state['updates'] != config['historia']['updates'][-1]
                or not config.get('rng')):
            raise ValueError('Missing Adam/RNG continuation state')
        for key in ['first_moment', 'second_moment']:
            if not state[key] or any(name not in archive.files for name in state[key]):
                raise ValueError('Missing persisted Adam moments')
        return {'updates': state['updates'], 'epoch': config['historia']['epocas_corridas'],
                'first_moment_tensors': len(state['first_moment']),
                'second_moment_tensors': len(state['second_moment']), 'rng': config['rng']}


def read_archive_config(archive):
    import json
    return json.loads(str(archive['configuracion']))


def continue_run(source, output, clean, validation, target_epoch):
    """Continue the LAST checkpoint, retain the previous BEST, and index noise absolutely."""
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('Run output must be new; no automatic retries')
    config = read_json(source/'config.json')
    original = read_json(source/'manifest.json')
    if original['status'] != 'completed':
        raise ValueError('Source run must be complete')
    for entry, name in [(original['best']['model'], 'best_model.npz'),
                         (original['last_model'], 'last_model.npz')]:
        if sha256(source/name) != entry['sha256']:
            raise ValueError('Source checkpoint hash mismatch')
    model = Autoencoder.load(source/'last_model.npz')
    initial = model.history['epocas_corridas']
    if (type(target_epoch) is not int or target_epoch <= initial
            or initial != original['epochs_completed']):
        raise ValueError('Target must exceed the completed source epoch')
    if model.configuration != Autoencoder(**config['model'], seed=config['seed']).configuration:
        raise ValueError('Source model configuration mismatch')
    adam = check_adam_state(source/'last_model.npz')
    best_model = Autoencoder.load(source/'best_model.npz')
    score = validation_score(grouped_metrics(validation, best_model.reconstruct(validation['inputs'])),
                             config['selection_levels'])
    np.testing.assert_allclose(score, original['best']['score'], rtol=0, atol=1e-14)
    history = copy.deepcopy(read_json(source/'history.json'))
    prior_best = min(history['validation'], key=lambda point: (point['score'], point['epoch']))
    if (prior_best['score'] != score or prior_best['epoch'] != original['best']['epoch']
            or best_model.history['epocas_corridas'] != prior_best['epoch']):
        raise ValueError('Source validation history mismatch')
    output.mkdir(parents=True)
    write_json(output/'config.json', {**config, 'mode': 'continuation',
                                     'initial_epoch': initial, 'target_epoch': target_epoch})
    shutil.copyfile(source/'best_model.npz', output/'best_model.npz')
    result = {'run_id': output.name, 'variant': config['variant'], 'seed': config['seed'],
              'status': 'running', 'initial_epoch': initial, 'target_epoch': target_epoch,
              'epochs_completed': initial, 'additional_epochs_completed': 0,
              'best': copy.deepcopy(original['best']), 'initial_best': copy.deepcopy(original['best']),
              'initial_adam_state': adam, 'effective_model': model.configuration,
              'source_files': {p.name: sha256(p) for p in source.iterdir() if p.is_file()}}
    start = time.perf_counter()
    try:
        for epoch in range(initial+1, target_epoch+1):
            model.train(training_inputs(clean, config['training_levels'], config['seed'], epoch),
                        1, targets=clean)
            result.update(epochs_completed=model.history['epocas_corridas'],
                          additional_epochs_completed=model.history['epocas_corridas']-initial)
            if model.history['stop_reason'] in ['interrupted', 'numerical_failure']:
                raise RuntimeError(f'Training stopped: {model.history["stop_reason"]}')
            if epoch % config['evaluate_every'] == 0 or epoch == target_epoch:
                rows = grouped_metrics(validation, model.reconstruct(validation['inputs']))
                score = validation_score(rows, config['selection_levels'])
                history['validation'].append({'epoch': epoch, 'score': score, 'levels': rows})
                if (score, epoch) < (result['best']['score'], result['best']['epoch']):
                    model.save(output/'best_model.npz')
                    result['best'] = {'epoch': epoch, 'score': score}
                model.save(output/'last_model.npz')
                write_json(output/'manifest.json', result)
                print(f'{output.name} epoch={epoch} validation_errors={score[0]:.5f}', flush=True)
        result['status'] = 'completed'
    except BaseException as error:
        result['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        result['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        model.save(output/'last_model.npz')
        result['duration_seconds'] = time.perf_counter()-start
        result['best']['model'] = {'path': 'best_model.npz', 'sha256': sha256(output/'best_model.npz')}
        result['last_model'] = {'path': 'last_model.npz', 'sha256': sha256(output/'last_model.npz')}
        history['training'] = model.history
        write_json(output/'history.json', history)
        write_json(output/'manifest.json', result)
    return result


def run_campaign(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('Campaign output must be new')
    origin, ledger, selected = verify_completed(source)
    if selected['variant'] != 'simple-d20':
        raise ValueError('Expected the previously selected simple-d20 architecture')
    versions = {'python': platform.python_version(), 'numpy': np.__version__}
    if versions != origin['versions']:
        raise ValueError('Continuation requires the original Python/NumPy versions')
    output.mkdir(parents=True)
    (output/'origins').mkdir()
    (output/'collections').mkdir()
    shutil.copyfile(source/'reference/font.h', output/'font.h')
    shutil.copyfile(source/'collections/validation.npz', output/'collections/validation.npz')
    with np.load(output/'collections/validation.npz', allow_pickle=False) as saved:
        validation = {key: saved[key][saved['levels'] != .3] for key in saved.files}
    clean = load_dataset(output/'font.h').vectors
    seeds = origin['config']['seeds']
    source_rows = [r for r in ledger['runs'] if r['variant'] == selected['variant']]
    for row in source_rows:
        shutil.copytree(source/'runs'/row['run_id'], output/'origins'/row['run_id'])
        check_adam_state(output/'origins'/row['run_id']/'last_model.npz')
    manifest = {'schema_version': 1, 'source': str(source.resolve()),
                'source_selection_sha256': sha256(source/'selection.json'),
                'variant': selected['variant'], 'seeds': seeds, 'initial_epoch': 3000,
                'target_epoch': 6000, 'max_additional_epochs': 9000,
                'architecture': origin['variants'][selected['variant']]['architecture'],
                'versions': versions, 'noise': origin['noise'], 'final_seed': FINAL_SEED,
                'selection_policy': 'Keep best validation checkpoint over epochs 0..6000; same score and ties as before',
                'source_files': {p.relative_to(source).as_posix(): sha256(p)
                                 for p in sorted(source.rglob('*')) if p.is_file()}}
    (output/'implementation').mkdir()
    shutil.copyfile(Path(__file__), output/'implementation/denoise_continuation.py')
    manifest['frozen_files'] = {p.relative_to(output).as_posix(): sha256(p)
                               for p in output.rglob('*') if p.is_file()}
    write_json(output/'manifest.json', manifest)
    (output/'runs').mkdir()
    result = {'status': 'running', 'runs': [], 'additional_epochs_completed': 0,
              'manifest_sha256': sha256(output/'manifest.json')}
    write_json(output/'training.json', result)
    start = time.perf_counter()
    try:
        for seed in seeds:
            name = f'simple-d20-seed-{seed}'
            row = continue_run(output/'origins'/name, output/'runs'/name, clean, validation, 6000)
            result['runs'].append(row)
            result['additional_epochs_completed'] += row['additional_epochs_completed']
            write_json(output/'training.json', result)
        result['status'] = 'completed'
    except BaseException as error:
        result['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        result['error'] = f'{type(error).__name__}: {error}'
        result['runs'] = [read_json(p) for p in (output/'runs').glob('*/manifest.json')]
        result['additional_epochs_completed'] = sum(r['additional_epochs_completed'] for r in result['runs'])
        raise
    finally:
        result['duration_seconds'] = time.perf_counter()-start
        write_json(output/'training.json', result)
    if result['additional_epochs_completed'] != 9000 or len(result['runs']) != 3:
        raise ValueError('Incomplete continuation campaign')
    representative = min(result['runs'], key=lambda r: (r['best']['score'], r['seed']))
    write_json(output/'selection.json', {'variant': selected['variant'],
               'run_id': representative['run_id'], 'runs': result['runs'],
               'old_mean_score': np.mean([r['initial_best']['score'] for r in result['runs']], axis=0).tolist(),
               'new_mean_score': np.mean([r['best']['score'] for r in result['runs']], axis=0).tolist(),
               'manifest_sha256': sha256(output/'manifest.json')})
    evaluate(output)
    report(output)


def verified(campaign):
    campaign = Path(campaign)
    manifest, training, selection = [read_json(campaign/name) for name in
                                     ['manifest.json', 'training.json', 'selection.json']]
    if (training['status'] != 'completed' or training['additional_epochs_completed'] != 9000
            or len(training['runs']) != 3 or training['runs'] != selection['runs']
            or training['manifest_sha256'] != sha256(campaign/'manifest.json')
            or selection['manifest_sha256'] != training['manifest_sha256']):
        raise ValueError('Incomplete campaign or changed selection/manifest')
    for name, digest in manifest['frozen_files'].items():
        if sha256(campaign/name) != digest:
            raise ValueError(f'Frozen source changed: {name}')
    expected = min(training['runs'], key=lambda r: (r['best']['score'], r['seed']))['run_id']
    if expected != selection['run_id']:
        raise ValueError('Representative no longer matches validation')
    for row in training['runs']:
        if row['status'] != 'completed' or row['epochs_completed'] != 6000 or row['additional_epochs_completed'] != 3000:
            raise ValueError('Incomplete run')
        for key in ['best', 'last_model']:
            entry = row['best']['model'] if key == 'best' else row[key]
            if sha256(campaign/'runs'/row['run_id']/entry['path']) != entry['sha256']:
                raise ValueError('Continuation model hash mismatch')
    return manifest, training, selection


def evaluate(campaign, output=None):
    campaign = Path(campaign)
    manifest, training, selection = verified(campaign)
    output = Path(output) if output else campaign/'evaluation'
    if output.exists():
        raise ValueError('Evaluation output must be new')
    output.mkdir(parents=True)
    collection = make_collection(load_dataset(campaign/'font.h').vectors,
                                 manifest['noise']['evaluation_levels'], 100, manifest['final_seed'])
    np.savez_compressed(output/'collection.npz', **collection)
    rows = [{'model': 'noisy-input', 'budget': 0, 'seed': None, **r}
            for r in grouped_metrics(collection, collection['inputs'])]
    for row in training['runs']:
        for budget, directory in [(3000, 'origins'), (6000, 'runs')]:
            path = campaign/directory/row['run_id']/'best_model.npz'
            prediction = Autoencoder.load(path).reconstruct(collection['inputs'])
            name = f'budget-{budget}-seed-{row["seed"]}'
            np.savez_compressed(output/f'{name}.npz', predictions=prediction)
            rows.extend({'model': name, 'budget': budget, 'seed': row['seed'], **r}
                        for r in grouped_metrics(collection, prediction))
    write_csv(output/'by-level.csv', rows)
    write_json(output/'metrics.json', {'selection_sha256': sha256(campaign/'selection.json'),
               'collection_sha256': sha256(output/'collection.npz'), 'seed': manifest['final_seed'],
               'rows': rows, 'out_of_training_range': [.3],
               'versions': {'python': platform.python_version(), 'numpy': np.__version__}})
    return rows


def report(campaign):
    campaign = Path(campaign)
    manifest, training, selection = verified(campaign)
    with np.load(campaign/'collections/validation.npz', allow_pickle=False) as saved:
        validation = {key: saved[key][saved['levels'] != .3] for key in saved.files}
    for row in training['runs']:
        directory = campaign/'runs'/row['run_id']
        score = validation_score(grouped_metrics(validation, Autoencoder.load(directory/'best_model.npz').reconstruct(
            validation['inputs'])), manifest['noise']['selection_levels'])
        np.testing.assert_allclose(score, row['best']['score'], rtol=0, atol=1e-14)
        history = read_json(directory/'history.json')
        point = min(history['validation'], key=lambda p: (p['score'], p['epoch']))
        assert point['epoch'] == row['best']['epoch'] and point['score'] == row['best']['score']
        initial_history = read_json(campaign/'origins'/row['run_id']/'history.json')['training']
        for key in ['epochs', 'costo', 'mse', 'updates', 'learning_rate']:
            assert history['training'][key][:3000] == initial_history[key]
        state = check_adam_state(directory/'last_model.npz')
        assert state['epoch'] == state['updates'] == 6000
    for name, digest in manifest['source_files'].items():
        if sha256(Path(manifest['source'])/name) != digest:
            raise ValueError(f'Previous campaign changed: {name}')
    metrics = read_json(campaign/'evaluation/metrics.json')
    assert metrics['selection_sha256'] == sha256(campaign/'selection.json')
    rows = metrics['rows']
    aggregates = []
    for budget in [3000, 6000]:
        for level in manifest['noise']['evaluation_levels']:
            group = [r for r in rows if r['budget'] == budget and r['level'] == level]
            aggregates.append({'budget': budget, 'level': level,
                **{f'{key}_{stat}': float(fn([r[key] for r in group]))
                   for key in ['mean_pixel_errors', 'exact_fraction', 'within_tolerance_fraction']
                   for stat, fn in [('mean', np.mean), ('min', np.min), ('max', np.max)]}})
    write_csv(campaign/'evaluation/seed-aggregates.csv', aggregates)
    figure, axis = plt.subplots(figsize=(9, 4))
    for row in training['runs']:
        history = read_json(campaign/'runs'/row['run_id']/'history.json')['validation']
        axis.plot([p['epoch'] for p in history], [p['score'][0] for p in history], label=f'Semilla {row["seed"]}')
        axis.scatter([row['best']['epoch']], [row['best']['score'][0]], marker='*', s=100)
    axis.axvline(3000, linestyle='--', color='0.5', label='Presupuesto anterior')
    axis.set(xlabel='Época total', ylabel='Error medio de validación (píxeles)',
             title='Continuación de simple-d20: tres semillas hasta 6000 épocas', ylim=(.5, 2))
    axis.grid(alpha=.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(campaign/'validation-curves.png', dpi=160)
    plt.close(figure)
    old, new = selection['old_mean_score'][0], selection['new_mean_score'][0]
    lines = ['# TP5 1.b — Continuación de la ganadora hasta 6000 épocas', '',
             f'Arquitectura: `{"→".join(map(str, manifest["architecture"]))}` (`simple-d20`). '
             'Se continuaron las semillas 42, 43 y 44 desde su último checkpoint de época 3000, '
             'con pesos, momentos y contador de Adam, RNG e historia conservados.', '',
             f'**Validación media: {old:.5f} → {new:.5f} píxeles por imagen '
             f'({100*(old-new)/old:.2f} % de reducción).** '
             f'Representante: `{selection["run_id"]}`.', '',
             f'Se consumieron **9000 épocas adicionales**: 3000 por semilla, hasta 6000 totales. '
             f'Tiempo de continuación: **{training["duration_seconds"]/60:.2f} minutos**. '
             'No se entrenaron otras arquitecturas ni se cambiaron tasa, pérdida, ruido o lotes.', '',
             '## Checkpoints por semilla', '',
             '| Semilla | Mejor época hasta 3000 | Error anterior | Mejor época hasta 6000 | Error nuevo |',
             '|---|---:|---:|---:|---:|']
    for row in training['runs']:
        lines.append(f'| {row["seed"]} | {row["initial_best"]["epoch"]} | {row["initial_best"]["score"][0]:.5f} | '
                     f'{row["best"]["epoch"]} | {row["best"]["score"][0]:.5f} |')
    lines.extend(['', '![Curvas de validación](validation-curves.png)', '',
                  'Se compara el mejor checkpoint disponible bajo cada presupuesto, no solo las '
                  'últimas épocas. Se conservaron los mejores anteriores como candidatos. '
                  'El ruido continúa con la época absoluta (3001–6000), sin reiniciar su secuencia. '
                  'La selección mantiene el promedio por igual de errores de 0,05/0,10/0,20; '
                  'desempata MSE y luego época más temprana. El representante desempata por semilla.', '',
                  '## Evaluación independiente con ruido nuevo', '',
                  f'Colección nueva: semilla **{FINAL_SEED}**, 12.832 imágenes sobre los mismos '
                  '32 caracteres. Se generó después de congelar la selección. Ambos presupuestos '
                  'se evalúan sobre exactamente las mismas entradas; los valores son medias de tres semillas. '
                  'p=0,30 está fuera del rango entrenado. No se estima generalización a caracteres nuevos.', '',
                  '| Presupuesto | Ruido p | Error medio | Recuperación exacta | Con ≤1 error |',
                  '|---|---:|---:|---:|---:|'])
    for row in aggregates:
        lines.append(f'| {row["budget"]} | {row["level"]:g} | {row["mean_pixel_errors_mean"]:.5f} | '
                     f'{100*row["exact_fraction_mean"]:.2f} % | {100*row["within_tolerance_fraction_mean"]:.2f} % |')
    lines.extend(['', '## Verificaciones y reproducción', '',
                  'Se recalculó la validación de los tres modelos elegidos, se comprobó que las '
                  'primeras 3000 épocas de las historias sigan idénticas y que Adam alcance '
                  '6000 actualizaciones. Todos los archivos de la campaña anterior conservan sus SHA256. '
                  'La prueba automatizada contrasta una continuación corta con entrenamiento '
                  'ininterrumpido, incluidas las predicciones y los momentos de Adam.', '',
                  'Desde `tp5/`, para repetir el experimento en una salida nueva (consume otras 9000 épocas):', '',
                  '```powershell', 'python -m src.denoise_continuation run --source results/denoising-architectures-2026-11-17 --output results/denoising-6000-reproduction', '```', '',
                  'Para recalcular únicamente la evaluación, sin entrenar:', '',
                  '```powershell', 'python -m src.denoise_continuation evaluate --campaign results/denoising-6000-2026-11-17 --output results/denoising-6000-evaluation-reloaded', '```', '',
                  'El fundamento del DAE sigue siendo el par entrada contaminada/objetivo limpio '
                  '(OFF-022, p. 22). El presupuesto de 6000 épocas es una decisión experimental '
                  'autorizada por el usuario. Esta comparación no demuestra convergencia definitiva.'])
    (campaign/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    verification = {'status': 'passed', 'additional_epochs': 9000,
               'total_epochs_per_seed': 6000, 'source_files_preserved': len(manifest['source_files']),
               'history_prefix_preserved': True, 'adam_updates_per_seed': 6000,
               'validation_recomputed': True, 'selection_sha256': sha256(campaign/'selection.json')}
    reload_path = campaign/'validation/reload-comparison.json'
    if reload_path.exists():
        repeated = read_json(reload_path)
        if repeated['status'] != 'passed' or repeated['metrics_sha256'] != sha256(campaign/'evaluation/metrics.json'):
            raise ValueError('Reload verification does not describe this evaluation')
        verification['evaluation_reload'] = repeated
        with (campaign/'report.md').open('a', encoding='utf-8') as stream:
            stream.write(f'\nLa evaluación se repitió sin entrenar: **{repeated["arrays_identical"]} arrays**, '
                         'la tabla CSV y las métricas resultaron idénticos. '
                         '[Verificación](validation/reload-comparison.json).\n')
    test_log = campaign/'validation/tp5-tests.log'
    if test_log.exists():
        import re
        text = test_log.read_text(encoding='utf-8')
        count = re.search(r'Ran (\d+) tests? in', text)
        if count is None or not text.rstrip().endswith('OK'):
            raise ValueError('Test log does not record a passing suite')
        verification['tests_passed'] = int(count.group(1))
        verification['test_log_sha256'] = sha256(test_log)
        with (campaign/'report.md').open('a', encoding='utf-8') as stream:
            stream.write(f'\nPasaron **{count.group(1)} tests** del TP5. [Log](validation/tp5-tests.log).\n')
    write_json(campaign/'verification.json', verification)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--source', required=True)
    run.add_argument('--output', required=True)
    evaluation = sub.add_parser('evaluate')
    evaluation.add_argument('--campaign', required=True)
    evaluation.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'run':
        run_campaign(args.source, args.output)
    else:
        evaluate(args.campaign, args.output)


if __name__ == '__main__':
    main()
