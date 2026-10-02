"""Carga cacheada y partición de digits.csv; el test se reserva para el final."""
from __future__ import annotations

import ast
import csv
import hashlib
from pathlib import Path

import numpy as np


EJERCICIO = Path(__file__).resolve().parents[1]
DIGITS = EJERCICIO.parent / "data" / "digits.csv"
CACHE = EJERCICIO / "cache" / "digits.npz"
SEMILLA = 42
N_ENTRADAS = 784
N_CLASES = 10
VERSION_CACHE = 1


def cargar(
    ruta_csv: str | Path = DIGITS,
    ruta_cache: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve X (N×784) e y one-hot (N×10), ambos float32.

    Replica la deserialización del loader de la cátedra: literal_eval de
    image y conversión a float32, sin transformar los valores de los píxeles.
    El CSV se parsea sólo si falta el caché o cambió el archivo de origen.
    """
    ruta_csv = Path(ruta_csv).resolve()
    if ruta_cache is None:
        if ruta_csv == DIGITS.resolve():
            ruta_cache = CACHE
        else:
            identificador = hashlib.sha256(str(ruta_csv).encode()).hexdigest()[:12]
            ruta_cache = CACHE.parent / f"{ruta_csv.stem}-{identificador}.npz"
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (X_train, y_train, X_val, y_val), estratificados por dígito.

    Se reserva el 20 % de cada clase para validación, redondeando al entero
    más próximo y dejando al menos una muestra de cada clase en cada parte.
    La proporción puede diferir levemente de 80/20 por el redondeo.
    """
    if X.ndim != 2 or X.shape[1] != N_ENTRADAS or y.shape != (len(X), N_CLASES):
        raise ValueError("Se esperan X con forma (N, 784) e y con forma (N, 10).")
    if len(X) == 0 or not np.all((y == 0) | (y == 1)) or not np.all(y.sum(axis=1) == 1):
        raise ValueError("y debe contener muestras con etiquetas one-hot válidas.")
    rng = np.random.default_rng(semilla)
    etiquetas = y.argmax(axis=1)
    entrenamiento = []
    validacion = []
    for digito in np.unique(etiquetas):
        indices = rng.permutation(np.flatnonzero(etiquetas == digito))
        if len(indices) < 2:
            raise ValueError(f"El dígito {digito} necesita al menos dos muestras para estratificar.")
        n_val = min(len(indices) - 1, max(1, int(len(indices) * 0.2 + 0.5)))
        validacion.append(indices[:n_val])
        entrenamiento.append(indices[n_val:])
    idx_train = rng.permutation(np.concatenate(entrenamiento))
    idx_val = rng.permutation(np.concatenate(validacion))
    return X[idx_train], y[idx_train], X[idx_val], y[idx_val]


def explorar(y: np.ndarray, y_train: np.ndarray, y_val: np.ndarray) -> None:
    """Imprime la distribución de las diez clases y sus particiones."""
    total, train, val = (etiquetas.sum(axis=0).astype(int) for etiquetas in (y, y_train, y_val))
    print(f"digits.csv: {len(y)} muestras")
    print("Dígito    Total    Train    Val")
    for digito in range(N_CLASES):
        print(f"{digito:6d} {total[digito]:8d} {train[digito]:8d} {val[digito]:6d}")
    print(f"Totales {len(y):8d} {len(y_train):8d} {len(y_val):6d}")
    print(f"Dígito 8: {total[8]} muestras. Dígito 5: {total[5]} muestras.")
    if total[8] == 0:
        print("El 8 está ausente: y conserva sus 10 columnas, con la columna 8 en cero.")


def main() -> None:
    X, y = cargar()
    X_train, y_train, X_val, y_val = particionar(X, y)
    print(f"X: {X.shape}, {X.dtype}; y: {y.shape}, {y.dtype}")
    print(f"Partición estratificada 80/20; semilla {SEMILLA}")
    explorar(y, y_train, y_val)


if __name__ == "__main__":
    main()
