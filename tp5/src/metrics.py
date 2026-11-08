"""Reconstruction metrics; no classification accuracy is used."""
import numpy as np


def reconstruction_metrics(target, prediction):
    target, prediction = np.asarray(target), np.asarray(prediction)
    if (target.ndim != 2 or target.shape[0] == 0 or target.shape[1] != 35
            or prediction.shape != target.shape or not np.isfinite(prediction).all()
            or not np.isin(target, [0, 1]).all()):
        raise ValueError('Expected binary (N,35) targets and finite matching predictions')
    errors = np.count_nonzero((prediction >= 0.5) != target, axis=1)
    squared = (prediction-target)**2
    return {'pixel_errors': errors.tolist(), 'max_pixel_errors': int(errors.max()),
            'mean_pixel_errors': float(errors.mean()),
            'exact_fraction': float(np.mean(errors == 0)),
            'within_tolerance_fraction': float(np.mean(errors <= 1)),
            'all_within_tolerance': bool(np.all(errors <= 1)),
            'mse_per_pixel': float(squared.mean()),
            'training_loss': float(0.5 * squared.sum(axis=1).mean())}
