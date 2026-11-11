"""Independent bit flips and fixed collections for known binary characters."""
import numpy as np


def _binary(values):
    values = np.asarray(values)
    if values.ndim != 2 or not len(values) or values.shape[1] != 35 or not np.isin(values, [0, 1]).all():
        raise ValueError('Expected nonempty binary (N,35) data')
    return values


def _seed(seed):
    if type(seed) is not int or seed < 0:
        raise ValueError('Seed and absolute epoch must be nonnegative integers')


def _levels(levels):
    values = np.asarray(levels, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Noise levels must be finite probabilities in [0,1]')
    return values


def corrupt(clean, probability, rng):
    """Flip each bit independently; scalar or one probability per image."""
    clean = _binary(clean)
    p = np.asarray(probability, dtype=float)
    if p.ndim > 1 or (p.ndim == 1 and p.shape != (len(clean),)):
        raise ValueError('Probability must be scalar or one value per image')
    _levels(np.atleast_1d(p))
    return np.logical_xor(clean, rng.random(clean.shape) < (p[:, None] if p.ndim else p)).astype(float)


def training_inputs(clean, levels, seed, epoch):
    """Epoch-indexed noise is independent of model RNG and validation calls."""
    _seed(seed)
    _seed(epoch)
    clean = _binary(clean)
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 1]))
    return corrupt(clean, rng.choice(_levels(levels), len(clean)), rng)


def make_collection(clean, levels, realizations, seed):
    """Order: level, character index, realization; p=0 has one copy."""
    clean = _binary(clean)
    levels = _levels(levels)
    _seed(seed)
    if type(realizations) is not int or realizations < 1 or len(set(levels)) != len(levels):
        raise ValueError('Require positive realization count and distinct noise levels')
    rng = np.random.default_rng(seed)
    indices, probabilities, repetitions = [], [], []
    for p in levels:
        count = 1 if p == 0 else realizations
        indices.extend(np.repeat(np.arange(len(clean)), count))
        probabilities.extend([p]*(len(clean)*count))
        repetitions.extend(np.tile(np.arange(count), len(clean)))
    indices = np.asarray(indices, dtype=np.int64)
    probabilities = np.asarray(probabilities)
    targets = clean[indices].copy()
    return {'inputs': corrupt(targets, probabilities, rng), 'targets': targets,
            'character_indices': indices, 'levels': probabilities,
            'realization_indices': np.asarray(repetitions, dtype=np.int64)}
