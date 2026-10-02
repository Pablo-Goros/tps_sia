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


# --------------------------------------------------------------------------- #
# Métricas de CLASIFICACIÓN (estudio de generalización / umbral de detección).
# `clase` es el ground truth binario (flagged_fraud) y `puntaje` la probabilidad
# estimada; con un umbral u, se predice fraude cuando puntaje >= u.

def matriz_confusion(clase: np.ndarray, puntaje: np.ndarray, umbral: float) -> Dict[str, int]:
    pred = puntaje >= umbral
    real = clase == 1
    return {"vp": int(np.sum(pred & real)), "fp": int(np.sum(pred & ~real)),
            "fn": int(np.sum(~pred & real)), "vn": int(np.sum(~pred & ~real))}


def clasificacion(clase: np.ndarray, puntaje: np.ndarray, umbral: float) -> Dict[str, float]:
    """Precision, recall, F1, especificidad y accuracy para un umbral dado."""
    m = matriz_confusion(clase, puntaje, umbral)
    vp, fp, fn, vn = m["vp"], m["fp"], m["fn"], m["vn"]
    precision = vp / (vp + fp) if vp + fp else 0.0
    recall = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**m, "umbral": float(umbral), "precision": precision, "recall": recall, "f1": f1,
            "especificidad": vn / (vn + fp) if vn + fp else 0.0,
            "accuracy": (vp + vn) / len(clase),
            "tasa_alertas": float(np.mean(puntaje >= umbral))}
