"""Métricas de regresión usadas en el estudio de aprendizaje (TP3 - SIA)."""
from __future__ import annotations

from typing import Dict

import numpy as np


def error_cuadratico_medio(y: np.ndarray, o: np.ndarray) -> float:
    return float(np.mean((y - o) ** 2))


def costo_catedra(y: np.ndarray, o: np.ndarray) -> float:
    """E(w) = 1/2 * sum (zeta - O)^2, tal cual la define la Clase 10.2."""
    return float(0.5 * np.sum((y - o) ** 2))


def raiz_error_cuadratico_medio(y: np.ndarray, o: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - o) ** 2)))


def error_absoluto_medio(y: np.ndarray, o: np.ndarray) -> float:
    return float(np.mean(np.abs(y - o)))


def error_maximo(y: np.ndarray, o: np.ndarray) -> float:
    return float(np.max(np.abs(y - o)))


def r2(y: np.ndarray, o: np.ndarray) -> float:
    ss_res = float(np.sum((y - o) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot


def fraccion_fuera_de_rango(o: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> float:
    """Proporción de salidas que caen fuera del rango válido de una probabilidad."""
    return float(np.mean((o < lo) | (o > hi)))


def resumen(y: np.ndarray, o: np.ndarray) -> Dict[str, float]:
    return {
        "mse": error_cuadratico_medio(y, o),
        "rmse": raiz_error_cuadratico_medio(y, o),
        "mae": error_absoluto_medio(y, o),
        "error_max": error_maximo(y, o),
        "r2": r2(y, o),
        "costo_catedra": costo_catedra(y, o),
        "fuera_de_rango": fraccion_fuera_de_rango(o),
        "salida_min": float(np.min(o)),
        "salida_max": float(np.max(o)),
    }
