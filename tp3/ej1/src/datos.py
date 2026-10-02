"""Carga, exploración y preprocesamiento del dataset de fraude (TP3 - SIA)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Columna objetivo: la salida de BigModel que TinyModel debe destilar.
OBJETIVO = "big_model_fraud_probability"
# Ground truth: el enunciado prohíbe explícitamente usarla para entrenar.
PROHIBIDA = "flagged_fraud"


@dataclass
class Normalizador:
    """Estandarización z-score: (x - media) / desvio, ajustada sobre entrenamiento."""
    media: np.ndarray
    desvio: np.ndarray

    @classmethod
    def ajustar(cls, X: np.ndarray) -> "Normalizador":
        desvio = X.std(axis=0)
        desvio[desvio == 0.0] = 1.0  # columnas constantes: no escalar
        return cls(media=X.mean(axis=0), desvio=desvio)

    def aplicar(self, X: np.ndarray) -> np.ndarray:
        return (X - self.media) / self.desvio


def cargar(ruta: str, columnas_excluidas: Optional[List[str]] = None
           ) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """Devuelve (X, y, nombres_de_features, dataframe_crudo)."""
    df = pd.read_csv(ruta)
    excluidas = set(columnas_excluidas or []) | {OBJETIVO, PROHIBIDA}
    features = [c for c in df.columns if c not in excluidas]
    X = df[features].to_numpy(dtype=float)
    y = df[OBJETIVO].to_numpy(dtype=float)
    return X, y, features, df


def explorar(df: pd.DataFrame) -> Dict:
    """Chequeos de calidad de datos que pide el enunciado antes de modelar."""
    num = df.select_dtypes(include=[np.number])
    y = df[OBJETIVO]
    return {
        "filas": int(df.shape[0]),
        "columnas": int(df.shape[1]),
        "nulos_por_columna": {c: int(v) for c, v in df.isna().sum().items()},
        "filas_duplicadas": int(df.duplicated().sum()),
        "valores_negativos_por_columna": {c: int(v) for c, v in (num < 0).sum().items()},
        "objetivo_fuera_de_0_1": int(((y < 0) | (y > 1)).sum()),
        "objetivo_en_extremos": {
            "iguales_a_1": int((y == 1.0).sum()),
            "iguales_a_0": int((y == 0.0).sum()),
            "mayores_a_0.99": int((y > 0.99).sum()),
            "menores_a_0.01": int((y < 0.01).sum()),
        },
        "balance_flagged_fraud": {str(k): int(v) for k, v in df[PROHIBIDA].value_counts().items()},
        "estadisticos": {
            c: {
                "min": float(num[c].min()), "max": float(num[c].max()),
                "media": float(num[c].mean()), "desvio": float(num[c].std()),
                "mediana": float(num[c].median()),
                "rango_ordenes_de_magnitud": float(np.log10(max(abs(num[c].max()), 1e-12) /
                                                            max(abs(num[c].min()), 1e-12))),
            }
            for c in num.columns
        },
        "correlacion_pearson_con_objetivo": {
            c: float(v) for c, v in num.corr()[OBJETIVO].sort_values().items() if c != OBJETIVO
        },
        "correlacion_spearman_con_objetivo": {
            c: float(v) for c, v in num.corr(method="spearman")[OBJETIVO].sort_values().items()
            if c != OBJETIVO
        },
    }


def solucion_analitica_lineal(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mínimos cuadrados (ecuaciones normales) sobre X con bias.

    Es el ÓPTIMO GLOBAL exacto del perceptrón simple lineal con costo cuadrático:
    sirve de cota inferior del error alcanzable y permite distinguir un problema
    de optimización (mal eta, pocas épocas) de una limitación de capacidad.
    """
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    w, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return w, Xb @ w


def referencia_logistica_analitica(X: np.ndarray, y: np.ndarray,
                                   recorte: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Mínimos cuadrados en el espacio logit: sigmoide(w·x) ajustada en forma cerrada.

    No es el óptimo exacto del perceptrón logístico con costo cuadrático (ese no
    tiene solución cerrada), pero sí una referencia fuerte del nivel de error que
    alcanza un modelo de la forma sigmoide(w·x). Sirve para verificar que el
    perceptrón no lineal también llegó a su propio techo de capacidad.
    """
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    yc = np.clip(y, recorte, 1.0 - recorte)
    z = np.log(yc / (1.0 - yc))
    w, *_ = np.linalg.lstsq(Xb, z, rcond=None)
    return w, 1.0 / (1.0 + np.exp(-(Xb @ w)))


# --------------------------------------------------------------------------- #
# Particiones para el estudio de generalización.

def estratos(y: np.ndarray, clase: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Estrato = decil del objetivo x clase de fraude.

    Estratificar por deciles de y mantiene la distribución de la probabilidad de
    BigModel (incluida la masa en los extremos) en cada partición; cruzarlo con
    flagged_fraud asegura la misma proporción de fraudes (11.6 %), necesaria para
    que las métricas de clasificación y el umbral sean comparables entre particiones.
    flagged_fraud se usa SÓLO para partir, nunca como entrada ni como objetivo.
    """
    cortes = np.quantile(y, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(y, cortes) * 2 + clase.astype(int)


def k_fold(n: int, k: int, rng: np.random.Generator,
           estrato: Optional[np.ndarray] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Lista de k pares (idx_entrenamiento, idx_validacion), estratificada si se pasa `estrato`."""
    fold = np.empty(n, dtype=int)
    if estrato is None:
        fold[rng.permutation(n)] = np.arange(n) % k
    else:
        for e in np.unique(estrato):
            idx = rng.permutation(np.where(estrato == e)[0])
            # Se rota el fold de arranque por estrato para no cargar siempre el fold 0.
            fold[idx] = (np.arange(len(idx)) + rng.integers(k)) % k
    return [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(k)]
