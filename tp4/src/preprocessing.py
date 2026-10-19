"""Escalado de variables (teórica Kohonen 02.2 - Estandarización).

Tres opciones vistas en clase:
  - min-max feature scaling       X' = (X - Xmin) / (Xmax - Xmin) * (b - a) + a
  - estandarización (z-score)     X~ = (X - mean) / std
  - unit length scaling           X' = X / ||X||
"""
from __future__ import annotations

import numpy as np


class Standardizer:
    """Z-score por columna. Guarda media y desvío para poder transformar datos nuevos
    (p. ej. creadores que llegan el mes que viene) con los mismos parámetros."""

    def __init__(self, ddof: int = 1):
        # ddof=1 -> varianza muestral (1/(n-1)) como en la teórica de PCA.
        self.ddof = ddof
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0, ddof=self.ddof)
        # una columna constante tendría std 0 -> la dejamos en 0 en vez de dividir por 0
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(Z, dtype=float) * self.std_ + self.mean_


def min_max_scale(X: np.ndarray, a: float = 0.0, b: float = 1.0) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mn, mx = X.min(axis=0), X.max(axis=0)
    rng = np.where(mx - mn == 0, 1.0, mx - mn)
    return (X - mn) / rng * (b - a) + a


def unit_length_scale(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms
