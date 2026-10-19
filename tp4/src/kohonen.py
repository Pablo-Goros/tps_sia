"""Red de Kohonen / Mapa Auto-Organizado (teórica 02 - Modelo de Kohonen).

Algoritmo de la teórica (slides "Algoritmo"):
  Inicialización
    1. X^p, p = 1..P registros de entrada de dimensión n (ESTANDARIZADOS)
    2. Definir la grilla de salida k x k
    3. Inicializar los pesos W_j (j = 1..k^2):
         - con valores aleatorios uniformes, o
         - con ejemplos al azar del conjunto de entrenamiento (recomendado para
           evitar "unidades muertas")
    4. Radio inicial R(0)
    5. Tasa de aprendizaje inicial eta(0) < 1
  Iteración i
    1. Seleccionar un registro X^p
    2. Neurona ganadora: W_k = argmin_j d(X^p, W_j)   (distancia euclídea)
    3. Regla de Kohonen: para j en N_k(i) = { n : ||n - n_k|| <= R(i) }
         W_j(i+1) = W_j(i) + eta(i) (X^p - W_j(i))
  R(i) -> 1 cuando i -> inf (o constante); eta(i) -> 0 (p. ej. eta(i) = 1/i).
  Cantidad total de iteraciones: en función de la cantidad de datos (p. ej. 500*N).

Vecindario (slide "Vecindario"): grilla rectangular, R = 1 -> 4 vecinos,
R = sqrt(2) -> 8 vecinos. Se usa `<=` para que con R = 1 se actualicen
efectivamente los vecinos pegados (como describe la teórica).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class KohonenConfig:
    k: int = 5                     # grilla k x k
    radius0: float | None = None   # R(0); None -> k (toda la grilla)
    radius_final: float = 1.0      # R(i) -> radius_final
    radius_schedule: str = "linear"  # 'linear' | 'exp' | 'constant'
    eta0: float = 0.5              # eta(0) < 1
    eta_schedule: str = "inverse"  # 'inverse' (eta0/(1+t)), 'linear', 'exp', 'constant'
    eta_final: float = 0.01
    epochs: int = 500              # iteraciones totales = epochs * P
    init: str = "samples"          # 'samples' | 'uniform'
    seed: int | None = 42


@dataclass
class KohonenHistory:
    quantization_error: list = field(default_factory=list)
    radius: list = field(default_factory=list)
    eta: list = field(default_factory=list)


class Kohonen:
    def __init__(self, n_features: int, config: KohonenConfig | None = None, **kwargs):
        self.cfg = config or KohonenConfig(**kwargs)
        self.k = self.cfg.k
        self.n_features = n_features
        self.rng = np.random.default_rng(self.cfg.seed)
        # coordenadas (fila, col) de cada una de las k^2 neuronas
        self.coords = np.array([(r, c) for r in range(self.k) for c in range(self.k)], dtype=float)
        # distancias en la grilla entre neuronas, precalculadas (k^2 x k^2)
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        self.grid_dist = np.sqrt((diff ** 2).sum(axis=2))
        self.W: np.ndarray | None = None     # (k^2, n_features)
        self.history = KohonenHistory()

    # ----------------------------------------------------------- schedules
    def _radius(self, t: float) -> float:
        """t en [0, 1] = fracción del entrenamiento transcurrida."""
        R0 = self.cfg.radius0 if self.cfg.radius0 is not None else float(self.k)
        Rf = self.cfg.radius_final
        s = self.cfg.radius_schedule
        if s == "constant":
            return R0
        if s == "linear":
            return max(Rf, R0 + (Rf - R0) * t)
        if s == "exp":
            return max(Rf, R0 * (Rf / R0) ** t)
        raise ValueError(f"radius_schedule desconocido: {s}")

    def _eta(self, t: float, epoch: int) -> float:
        e0, ef, s = self.cfg.eta0, self.cfg.eta_final, self.cfg.eta_schedule
        if s == "constant":
            return e0
        if s == "inverse":          # eta(i) ~ 1/i  (teórica), aplicado por época
            return max(ef, e0 / (1.0 + epoch))
        if s == "linear":
            return max(ef, e0 + (ef - e0) * t)
        if s == "exp":
            return max(ef, e0 * (ef / e0) ** t)
        raise ValueError(f"eta_schedule desconocido: {s}")

    # ---------------------------------------------------------------- init
    def _init_weights(self, X: np.ndarray) -> None:
        n_neurons = self.k * self.k
        if self.cfg.init == "samples":
            idx = self.rng.choice(len(X), size=n_neurons, replace=len(X) < n_neurons)
            self.W = X[idx].copy()
        elif self.cfg.init == "uniform":
            lo, hi = X.min(axis=0), X.max(axis=0)
            self.W = self.rng.uniform(lo, hi, size=(n_neurons, X.shape[1]))
        else:
            raise ValueError(f"init desconocido: {self.cfg.init}")

    # ------------------------------------------------------------- winner
    def winner(self, x: np.ndarray) -> int:
        """Índice (0..k^2-1) de la neurona con W_j más cercano a x (euclídea)."""
        d = np.linalg.norm(self.W - x, axis=1)
        return int(np.argmin(d))

    def winners(self, X: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(X[:, None, :] - self.W[None, :, :], axis=2)
        return np.argmin(d, axis=1)

    def idx_to_rc(self, j: int | np.ndarray):
        return np.divmod(j, self.k)

    # --------------------------------------------------------------- train
    def train(self, X: np.ndarray, verbose: bool = False) -> "Kohonen":
        X = np.asarray(X, dtype=float)
        P = len(X)
        self._init_weights(X)
        total = self.cfg.epochs * P
        it = 0
        for epoch in range(self.cfg.epochs):
            order = self.rng.permutation(P)
            for p in order:
                t = it / max(total - 1, 1)
                R, eta = self._radius(t), self._eta(t, epoch)
                x = X[p]
                kwin = self.winner(x)
                neigh = self.grid_dist[kwin] <= R
                self.W[neigh] += eta * (x - self.W[neigh])
                it += 1
            self.history.quantization_error.append(self.quantization_error(X))
            self.history.radius.append(R)
            self.history.eta.append(eta)
            if verbose and (epoch % max(1, self.cfg.epochs // 10) == 0 or epoch == self.cfg.epochs - 1):
                print(f"epoch {epoch:4d}  R={R:.2f}  eta={eta:.4f}  QE={self.history.quantization_error[-1]:.4f}")
        return self

    # ------------------------------------------------------------- metrics
    def quantization_error(self, X: np.ndarray) -> float:
        """Distancia media de cada registro a su neurona ganadora.
        Mide cuánto "se parecen los pesos a los datos de entrada" (slide Convergencia)."""
        w = self.winners(X)
        return float(np.linalg.norm(X - self.W[w], axis=1).mean())

    def topographic_error(self, X: np.ndarray) -> float:
        """Fracción de registros cuya 1ra y 2da neurona más cercana NO son vecinas
        pegadas (distancia en grilla > sqrt(2)). Mide si se preservó la topología:
        'neuronas vecinas contienen datos con algún grado de similitud entre sí'."""
        d = np.linalg.norm(X[:, None, :] - self.W[None, :, :], axis=2)
        two = np.argsort(d, axis=1)[:, :2]
        gd = self.grid_dist[two[:, 0], two[:, 1]]
        return float((gd > np.sqrt(2) + 1e-9).mean())

    def hits(self, X: np.ndarray) -> np.ndarray:
        """Cantidad de registros que caen en cada neurona (k x k)."""
        counts = np.bincount(self.winners(X), minlength=self.k * self.k)
        return counts.reshape(self.k, self.k)

    def dead_units(self, X: np.ndarray) -> int:
        return int((self.hits(X) == 0).sum())

    def u_matrix(self, neighbors: int = 8) -> np.ndarray:
        """Matriz U: para cada neurona, promedio de la distancia euclídea entre su
        vector de pesos y los de sus neuronas vecinas (4 u 8 vecinos)."""
        R = 1.0 if neighbors == 4 else np.sqrt(2)
        U = np.zeros(self.k * self.k)
        for j in range(self.k * self.k):
            mask = (self.grid_dist[j] <= R + 1e-9) & (self.grid_dist[j] > 0)
            U[j] = np.linalg.norm(self.W[mask] - self.W[j], axis=1).mean()
        return U.reshape(self.k, self.k)

    def component_plane(self, feature_idx: int) -> np.ndarray:
        """Valor del peso de una variable en cada neurona (slide 'Observar una sola variable')."""
        return self.W[:, feature_idx].reshape(self.k, self.k)

    def mean_feature_map(self, X: np.ndarray, feature_idx: int) -> np.ndarray:
        """Promedio de una variable entre los REGISTROS que cayeron en cada neurona."""
        w = self.winners(X)
        out = np.full(self.k * self.k, np.nan)
        for j in range(self.k * self.k):
            m = w == j
            if m.any():
                out[j] = X[m, feature_idx].mean()
        return out.reshape(self.k, self.k)
