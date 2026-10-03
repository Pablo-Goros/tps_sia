"""Carga cacheada y partición de dígitos con rutas explícitas del consumidor."""
from __future__ import annotations

import ast
import csv
from pathlib import Path

import numpy as np


SEMILLA = 42
N_ENTRADAS = 784
N_CLASES = 10
VERSION_CACHE = 1


def cargar(
    ruta_csv: str | Path,
    ruta_cache: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve X (N×784) e y one-hot (N×10), ambos float32.

    Replica la deserialización del loader de la cátedra: literal_eval de
    image y conversión a float32, sin transformar los valores de los píxeles.
    El CSV se parsea sólo si falta el caché o cambió el archivo de origen.
    El consumidor proporciona ambas rutas; este módulo no elige un ejercicio
    ni un directorio de caché.
    """
    ruta_csv = Path(ruta_csv).resolve()
    ruta_cache = Path(ruta_cache)
    if ruta_cache.resolve() == ruta_csv:
        raise ValueError("El caché debe tener una ruta distinta del CSV.")
    estado = ruta_csv.stat()
    firma = np.array([VERSION_CACHE, estado.st_size, estado.st_mtime_ns], dtype=np.int64)
    if ruta_cache.exists():
        with np.load(ruta_cache, allow_pickle=False) as datos:
            if (
                np.array_equal(datos["firma"], firma)
                and datos["origen"].item() == str(ruta_csv)
            ):
                return datos["X"], datos["y"]

    imagenes = []
    etiquetas = []
    with ruta_csv.open(encoding="utf-8-sig", newline="") as archivo:
        lector = csv.DictReader(archivo)
        if not {"image", "label"}.issubset(lector.fieldnames or []):
            raise ValueError("El CSV debe contener las columnas image y label.")
        for fila_numero, fila in enumerate(lector, start=2):
            imagen = np.array(ast.literal_eval(fila["image"]), dtype=np.float32)
            etiqueta = int(fila["label"])
            if imagen.shape != (N_ENTRADAS,) or not np.isfinite(imagen).all():
                raise ValueError(f"Fila {fila_numero}: se esperaban 784 píxeles finitos.")
            if not 0 <= etiqueta < N_CLASES:
                raise ValueError(f"Fila {fila_numero}: el dígito debe estar entre 0 y 9.")
            imagenes.append(imagen)
            etiquetas.append(etiqueta)
    if not imagenes:
        raise ValueError("El CSV no contiene muestras.")

    X = np.stack(imagenes)
    # Diez salidas incluso cuando alguna clase no aparece en entrenamiento.
    y = np.eye(N_CLASES, dtype=np.float32)[etiquetas]
    ruta_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ruta_cache, X=X, y=y, firma=firma, origen=str(ruta_csv))
    return X, y


def particionar(
    X: np.ndarray,
    y: np.ndarray,
    semilla: int = SEMILLA,
    *, return_indices: bool = False, validation_fraction: float = 0.2,
) -> tuple:
    """Devuelve (X_train, y_train, X_val, y_val), estratificados por dígito.

    Por defecto se reserva el 20 % de cada clase para validación, redondeando al entero
    más próximo y dejando al menos una muestra de cada clase en cada parte.
    La proporción puede diferir levemente por el redondeo. return_indices=True
    agrega los índices de train/validación referidos al orden original del CSV;
    validation_fraction permite otra proporción sin cambiar los defaults.
    """
    if X.ndim != 2 or X.shape[1] != N_ENTRADAS or y.shape != (len(X), N_CLASES):
        raise ValueError("Se esperan X con forma (N, 784) e y con forma (N, 10).")
    if len(X) == 0 or not np.all((y == 0) | (y == 1)) or not np.all(y.sum(axis=1) == 1):
        raise ValueError("y debe contener muestras con etiquetas one-hot válidas.")
    if not np.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction debe estar entre 0 y 1.")
    rng = np.random.default_rng(semilla)
    etiquetas = y.argmax(axis=1)
    entrenamiento = []
    validacion = []
    for digito in np.unique(etiquetas):
        indices = rng.permutation(np.flatnonzero(etiquetas == digito))
        if len(indices) < 2:
            raise ValueError(f"El dígito {digito} necesita al menos dos muestras para estratificar.")
        n_val = min(len(indices) - 1, max(1, int(len(indices) * validation_fraction + 0.5)))
        validacion.append(indices[:n_val])
        entrenamiento.append(indices[n_val:])
    idx_train = rng.permutation(np.concatenate(entrenamiento))
    idx_val = rng.permutation(np.concatenate(validacion))
    parts = (X[idx_train], y[idx_train], X[idx_val], y[idx_val])
    return (*parts, idx_train, idx_val) if return_indices else parts
