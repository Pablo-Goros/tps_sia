"""Defaults y comando de exploración de ej2; el loader reside en shared."""
from __future__ import annotations

import hashlib
from pathlib import Path

# Conservar la invocación histórica por archivo sin modificar sys.path.
# La implementación siempre se ejecuta como paquete desde la raíz.
if __name__ == "__main__" and not __package__:
    import subprocess
    import sys

    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "tps_sia.tp3.ej2.src.datos_digitos", *sys.argv[1:]],
        cwd=Path(__file__).resolve().parents[4],
    ))

import numpy as np

from tps_sia.tp3.shared.digit_dataset import (
    N_CLASES, N_ENTRADAS, SEMILLA, VERSION_CACHE, cargar as _cargar, particionar,
)

EJERCICIO = Path(__file__).resolve().parents[1]
DIGITS = EJERCICIO.parent / "data" / "digits.csv"
CACHE = EJERCICIO / "cache" / "digits.npz"


def cargar(
    ruta_csv: str | Path = DIGITS,
    ruta_cache: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Delega al loader común conservando los defaults y cachés de ej2."""
    ruta_csv = Path(ruta_csv).resolve()
    if ruta_cache is None:
        if ruta_csv == DIGITS.resolve():
            ruta_cache = CACHE
        else:
            identificador = hashlib.sha256(str(ruta_csv).encode()).hexdigest()[:12]
            ruta_cache = CACHE.parent / f"{ruta_csv.stem}-{identificador}.npz"
    return _cargar(ruta_csv, ruta_cache)


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
