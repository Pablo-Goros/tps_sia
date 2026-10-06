"""Hidden-layer BatchNorm configuration and analytic mini-batch derivative."""
import numpy as np


def validate_batch_norm(config):
    if config is None:
        return None
    defaults = {'epsilon': 1e-5, 'momentum': 0.1}
    if not isinstance(config, dict) or set(config) - set(defaults):
        raise ValueError('Unknown BatchNorm settings.')
    result = {**defaults, **config}
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError('BatchNorm settings must be finite real numbers.')
    if result['epsilon'] <= 0 or not 0 < result['momentum'] <= 1:
        raise ValueError('BatchNorm requires epsilon>0 and0<momentum<=1.')
    return result


def backward(delta, normalized, inverse_std, gamma):
    """delta already contains mean-loss reduction; do not divide gamma/beta again."""
    beta_gradient = delta.sum(axis=0)
    gamma_gradient = (delta * normalized).sum(axis=0)
    n = len(delta)
    input_gradient = gamma * inverse_std / n * (
        n * delta - beta_gradient - normalized * gamma_gradient)
    return input_gradient, gamma_gradient, beta_gradient
