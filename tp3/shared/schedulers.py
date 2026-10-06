"""Serializable validation-loss plateau scheduler; updates the next epoch rate."""
import copy
import math


def validate_scheduler(config):
    if config is None:
        return None
    defaults = {'name': 'reduce_on_plateau', 'factor': 0.5, 'patience': 3,
                'min_lr': 0.001875, 'min_delta': 0.0001}
    if not isinstance(config, dict) or set(config) - set(defaults):
        raise ValueError('Unknown scheduler configuration.')
    result = {**defaults, **copy.deepcopy(config)}
    if result['name'] != 'reduce_on_plateau':
        raise ValueError('Only reduce_on_plateau is supported.')
    for key in ('factor', 'min_lr', 'min_delta'):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Scheduler numeric settings must be finite.')
    if (not 0 < result['factor'] < 1 or result['min_lr'] <= 0 or result['min_delta'] < 0
            or isinstance(result['patience'], bool) or not isinstance(result['patience'], int)
            or result['patience'] < 1):
        raise ValueError('Invalid scheduler settings.')
    return result


def plateau_step(config, state, loss, rate):
    """Mutate scheduler state, preserving optimizer moments; return next rate."""
    if not math.isfinite(loss):
        raise ValueError('Scheduler requires finite validation loss.')
    if state['best'] is None or state['best'] - loss > config['min_delta']:
        state['best'], state['bad_epochs'] = float(loss), 0
    else:
        state['bad_epochs'] += 1
    if state['bad_epochs'] >= config['patience']:
        next_rate = max(config['min_lr'], rate * config['factor'])
        if next_rate < rate:
            state['reductions'] += 1
        rate = next_rate
        state['bad_epochs'] = 0
    state['current_lr'] = float(rate)
    return float(rate)
