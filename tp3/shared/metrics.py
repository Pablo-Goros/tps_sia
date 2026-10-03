"""Ten-class metrics with explicit undefined denominators and JSON-safe values."""
from __future__ import annotations

import numpy as np


def confusion_matrix(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Rows are actual classes; columns are predictions, including absent digits."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if (actual.ndim != 1 or actual.shape != predicted.shape or not actual.size
            or actual.dtype.kind not in 'iu' or predicted.dtype.kind not in 'iu'
            or np.any((actual < 0) | (actual >= 10))
            or np.any((predicted < 0) | (predicted >= 10))):
        raise ValueError('Expected equal nonempty integer labels between 0 and 9.')
    matrix = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matrix, (actual, predicted), 1)
    return matrix


def classification_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    matrix = confusion_matrix(actual, predicted)
    support, predictions = matrix.sum(axis=1), matrix.sum(axis=0)
    classes = []
    for digit in range(10):
        tp, real, guesses = int(matrix[digit, digit]), int(support[digit]), int(predictions[digit])
        classes.append({'class': digit, 'support': real, 'predictions': guesses,
                        'precision': tp / guesses if guesses else None,
                        'recall': tp / real if real else None,
                        # F1 is directly 2TP/(2TP+FP+FN), including zero when
                        # an observed class has no predictions. Absent classes
                        # are excluded from the supported-class macro average.
                        'f1': 2 * tp / (real + guesses) if real + guesses else None})
    evaluated = np.flatnonzero(support).tolist()
    return {'accuracy': float(matrix.trace() / matrix.sum()),
            'per_class': classes,
            'macro_f1': float(np.mean([classes[i]['f1'] for i in evaluated])),
            'balanced_accuracy': float(np.mean([classes[i]['recall'] for i in evaluated])),
            'average_classes': {'macro_f1': evaluated, 'balanced_accuracy': evaluated},
            'undefined_policy': {'precision': 'null when no predictions',
                                 'recall': 'null when no actual samples',
                                 'f1': '2TP/(support+predictions); null when both are zero',
                                 'averages': 'only classes with actual support'},
            'confusion_matrix': matrix.tolist(),
            'confusion_axes': {'rows': 'actual', 'columns': 'predicted'}}


def cross_entropy(actual: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean negative log probability, clipped at machine epsilon for zeros."""
    actual, probabilities = np.asarray(actual), np.asarray(probabilities, dtype=float)
    if (actual.ndim != 1 or not actual.size or actual.dtype.kind not in 'iu'
            or probabilities.shape != (actual.size, 10)
            or np.any((actual < 0) | (actual >= 10))
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or np.any(probabilities > 1) or not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError('Expected labels and finite ten-class probability distributions.')
    return float(-np.log(np.clip(probabilities[np.arange(actual.size), actual],
                                 np.finfo(float).eps, 1)).mean())


def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> dict:
    probabilities = model.predecir(X)
    report = classification_metrics(y.argmax(axis=1), probabilities.argmax(axis=1))
    # Stable logit-based loss supplied by MLP avoids clipping underflowed softmax.
    report['loss'] = float(model.costo(X, y))
    report['cross_entropy'] = report['loss'] if model.salida == 'softmax' else None
    return report
