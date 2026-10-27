"""Regla de Oja (teórica 06 - Regla de Oja y Sanger).

Perceptrón lineal simple con aprendizaje hebbiano normalizado:
    O = sum_i x_i w_i                       (salida lineal)
    dw = eta * (O x - O^2 w)                (regla de Oja, aprox. de Taylor de
                                            la normalización w/||w||)
Implementación (slide "Algoritmo"):
    input: X datos de dimensión N con media 0; eta; w inicial uniforme en [0,1]
    for epoch in epochs:
        for i in 1..N:
            y = inner(x_i, w)
            w += eta * y * (x_i - y * w)
    return w
Convergencia: ||w|| -> 1 y w -> autovector del mayor autovalor de la matriz de
correlaciones (= primera componente principal).
Tasa de aprendizaje: eta0 = 1/(1.2 lambda_1); como no se conoce lambda_1 se
estandariza y se usa eta0 <= 0.5 decreciente, o eta = 1e-3 constante.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OjaConfig:
    eta0: float = 0.01
    eta_schedule: str = "constant"   # 'constant' | 'inverse' (eta0/(1+epoch)) | 'linear'
    eta_final: float = 1e-4
    epochs: int = 200
    seed: int | None = 42
    shuffle: bool = True


@dataclass
class OjaHistory:
    w: list = field(default_factory=list)      # w al final de cada época
    norm: list = field(default_factory=list)
    eta: list = field(default_factory=list)
    cos_to_ref: list = field(default_factory=list)   # |cos(w, PC1)| por época


class Oja:
    def __init__(self, n_features: int, config: OjaConfig | None = None, **kwargs):
        self.cfg = config or OjaConfig(**kwargs)
        self.n_features = n_features
        self.rng = np.random.default_rng(self.cfg.seed)
        self.w = self.rng.uniform(0, 1, size=n_features)   # inicialización de la teórica
        self.history = OjaHistory()

    def _eta(self, epoch: int) -> float:
        e0, ef, s = self.cfg.eta0, self.cfg.eta_final, self.cfg.eta_schedule
        if s == "constant":
            return e0
        if s == "inverse":
            return max(ef, e0 / (1.0 + epoch))
        if s == "linear":
            return max(ef, e0 + (ef - e0) * epoch / max(self.cfg.epochs - 1, 1))
        raise ValueError(f"eta_schedule desconocido: {s}")

    def train(self, X: np.ndarray, reference: np.ndarray | None = None) -> "Oja":
        """X debe estar estandarizado (media 0). `reference` (opcional) es el
        autovector PC1 para registrar la convergencia época a época."""
        X = np.asarray(X, dtype=float)
        P = len(X)
        for epoch in range(self.cfg.epochs):
            eta = self._eta(epoch)
            order = self.rng.permutation(P) if self.cfg.shuffle else np.arange(P)
            for p in order:
                x = X[p]
                y = float(x @ self.w)
                self.w += eta * y * (x - y * self.w)
            self.history.w.append(self.w.copy())
            self.history.norm.append(float(np.linalg.norm(self.w)))
            self.history.eta.append(eta)
            if reference is not None:
                self.history.cos_to_ref.append(abs(cosine(self.w, reference)))
        return self

    def project(self, X: np.ndarray) -> np.ndarray:
        """Índice = salida de la neurona lineal: O^p = X^p . w"""
        return np.asarray(X, dtype=float) @ self.w


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
