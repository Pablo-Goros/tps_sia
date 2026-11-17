"""Epoch-wise noisy-input training, validation checkpoints and fixed selection."""
import time
from pathlib import Path
import numpy as np
from .autoencoder import Autoencoder
from .noise import training_inputs
from .denoise_evaluation import grouped_metrics
from .train import write_json, model_identity


def validation_score(rows, levels):
    chosen = [row for row in rows if row['level'] in levels]
    if len(chosen) != len(levels):
        raise ValueError('Validation must contain every selection level exactly once')
    return [float(np.mean([row[key] for row in chosen])) for key in ('mean_pixel_errors', 'mse_per_pixel')]


def select_runs(rows, variants, seeds):
    expected = {(v, s) for v in variants for s in seeds}
    if (len(rows) != len(expected) or {(r['variant'], r['seed']) for r in rows} != expected
            or any(r['status'] != 'completed' for r in rows)):
        raise ValueError('Cannot select an incomplete campaign')
    aggregates = []
    for variant in variants:
        scores = [r['best']['score'] for r in rows if r['variant'] == variant]
        aggregates.append({'variant': variant, 'mean_score': np.mean(scores, axis=0).tolist()})
    winner = min(aggregates, key=lambda item: (item['mean_score'], variants.index(item['variant'])))
    representative = min((r for r in rows if r['variant'] == winner['variant']),
                         key=lambda row: (row['best']['score'], row['seed']))
    return {'variant': winner['variant'], 'run_id': representative['run_id'],
            'aggregates': aggregates, 'criterion': 'mean binary errors, mean MSE; representative tie: lower seed',
            'selection_levels': [.05, .1, .2]}


def train_run(config, clean, validation, output):
    """A fresh run only. Small budgets support smoke checks; campaigns supply the approved budget."""
    output = Path(output)
    if output.exists():
        raise ValueError('Run output already exists; automatic retries are disabled')
    for key in ('max_epochs', 'evaluate_every'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError('Training budgets must be positive integers')
    model = Autoencoder(**config['model'], seed=config['seed'])
    output.mkdir(parents=True)
    write_json(output/'config.json', config)
    result = {'run_id': output.name, 'variant': config['variant'], 'seed': config['seed'],
              'status': 'running', 'epochs_completed': 0, 'best': None,
              'effective_model': model.configuration}
    history = {'validation': [], 'training': {}}
    start = time.perf_counter()
    failure = None
    try:
        for epoch in range(config['max_epochs']+1):
            if epoch:
                inputs = training_inputs(clean, config['training_levels'], config['seed'], epoch)
                model.train(inputs, 1, targets=clean)
                result['epochs_completed'] = model.history['epocas_corridas']
                if model.history['stop_reason'] in ('interrupted', 'numerical_failure'):
                    result['status'] = model.history['stop_reason']
                    break
            if epoch % config['evaluate_every'] == 0 or epoch == config['max_epochs']:
                predictions = model.reconstruct(validation['inputs'])
                rows = grouped_metrics(validation, predictions)
                score = validation_score(rows, config['selection_levels'])
                point = {'epoch': epoch, 'score': score, 'levels': rows}
                history['validation'].append(point)
                if result['best'] is None or (score, epoch) < (result['best']['score'], result['best']['epoch']):
                    model.save(output/'best_model.npz')
                    result['best'] = {'score': score, 'epoch': epoch,
                                      'model': model_identity(output/'best_model.npz')}
                model.save(output/'last_model.npz')
                write_json(output/'manifest.json', result)
                print(f'{output.name} epoch={epoch} validation_errors={score[0]:.5f}', flush=True)
        else:
            result['status'] = 'completed'
    except KeyboardInterrupt:
        result['status'] = 'interrupted'
    except Exception as error:
        result['status'] = 'failed'
        result['error'] = f'{type(error).__name__}: {error}'
        failure = error
    finally:
        model.save(output/'last_model.npz')
        result.update(epochs_completed=model.history['epocas_corridas'],
                      duration_seconds=time.perf_counter()-start,
                      last_model=model_identity(output/'last_model.npz'))
        history['training'] = model.history
        write_json(output/'history.json', history)
        write_json(output/'manifest.json', result)
    if failure is not None:
        raise failure
    return result
