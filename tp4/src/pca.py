"""Análisis de Componentes Principales (teórica 05 - PCA).

Procedimiento de la teórica (slide "Procedimiento"):
  1. Construir X con las variables en columnas
  2. Estandarizar X
  3. Calcular la matriz de correlaciones Sx
  4. Calcular autovalores y autovectores
  5. Ordenar los autovalores de mayor a menor
  6. Construir V con los autovectores de mayor autovalor
  7. Calcular Y = X~ V  (combinación lineal de las originales)

Como las variables tienen escalas muy distintas (seguidores vs porcentajes), se
usa la matriz de CORRELACIONES, que es la covarianza de las variables
estandarizadas (slides "Covarianza vs Correlación").
"""
from __future__ import annotations

import numpy as np

from .preprocessing import Standardizer


class PCA:
    def __init__(self, n_components: int | None = None):
        self.n_components = n_components
        self.scaler = Standardizer(ddof=1)
        self.corr_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None       # ordenados desc
        self.components_: np.ndarray | None = None        # (p, p) columnas = autovectores
        self.explained_variance_ratio_: np.ndarray | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray) -> "PCA":
        Z = self.scaler.fit_transform(X)
        n = Z.shape[0]
        # matriz de correlaciones = covarianza muestral de las variables estandarizadas
        self.corr_ = (Z.T @ Z) / (n - 1)
        # Sx es simétrica -> eigh (autovalores reales, autovectores ortonormales)
        vals, vecs = np.linalg.eigh(self.corr_)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        vals = np.clip(vals, 0, None)
        # Convención de signo: un autovector y su opuesto son ambos válidos
        # (slide "Índice ... Autovector opuesto"). Para que la interpretación sea
        # reproducible, se fija el signo de cada componente de modo que la carga de
        # mayor valor absoluto sea positiva.
        for j in range(vecs.shape[1]):
            i_max = np.argmax(np.abs(vecs[:, j]))
            if vecs[i_max, j] < 0:
                vecs[:, j] *= -1
        self.eigenvalues_ = vals
        self.components_ = vecs
        self.explained_variance_ratio_ = vals / vals.sum()
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X: np.ndarray, n_components: int | None = None) -> np.ndarray:
        k = n_components or self.n_components or self.components_.shape[1]
        Z = self.scaler.transform(X)
        return Z @ self.components_[:, :k]

    def fit_transform(self, X: np.ndarray, n_components: int | None = None) -> np.ndarray:
        return self.fit(X).transform(X, n_components)

    # -------------------------------------------------------------- helpers
    def orient(self, k: int, feature_idx: int) -> "PCA":
        """Elige el signo de la componente k para que la carga de `feature_idx`
        sea positiva (ambos signos son autovectores válidos; esto sólo fija la
        lectura: 'más PCk' = 'más de esa variable')."""
        if self.components_[feature_idx, k] < 0:
            self.components_[:, k] *= -1
        return self

    def loadings(self, k: int) -> np.ndarray:
        """Cargas (coeficientes a_ij) de la componente k (0-indexed)."""
        return self.components_[:, k]

    def cumulative_variance(self) -> np.ndarray:
        return np.cumsum(self.explained_variance_ratio_)

    def n_components_for(self, threshold: float) -> int:
        """Cantidad mínima de componentes que acumulan `threshold` de la varianza."""
        return int(np.searchsorted(self.cumulative_variance(), threshold) + 1)
