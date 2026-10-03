"""Chequeos del paso 1, desde la raíz: python -m tps_sia.tp3.ej2.tests.test_datos_digitos.

Estos chequeos leen test únicamente para verificar los datos, sin entrenar
modelos ni seleccionar hiperparámetros con él.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np

from ..src import datos_digitos as datos
from tps_sia.tp3.shared import digit_dataset
from tps_sia.tp3.shared.tests.test_digit_dataset import (
    indices_particion, test_cache_desde_csv,
)

TRAIN = datos.DIGITS
TEST = TRAIN.with_name("digits_test.csv")


def test_forma_y_rango():
    for ruta, n in ((TRAIN, 12449), (TEST, 2497)):
        X, y = datos.cargar(ruta)
        assert X.shape == (n, 784) and X.dtype == np.float32
        assert np.isfinite(X).all() and X.min() >= 0 and X.max() <= 1
        assert y.shape == (n, 10) and y.dtype == np.float32
        assert np.all((y == 0) | (y == 1)) and np.all(y.sum(axis=1) == 1)
        print(f"  ok  {ruta.name}: X{X.shape} float32 en [0,1], y{y.shape} one-hot")


def test_cache_consistente():
    X1, y1 = datos.cargar(TRAIN)
    # Alternar archivos no debe reemplazar el caché de entrenamiento.
    datos.cargar(TEST)
    with patch.object(digit_dataset.csv, "DictReader", side_effect=AssertionError("Se reparseó el CSV")):
        X2, y2 = datos.cargar(TRAIN)
        datos.cargar(TEST)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)
    print("  ok  cargas idénticas, sin reparsear y con cachés separados")


def test_one_hot():
    _, y = datos.cargar(TRAIN)
    etiquetas = y.argmax(axis=1)
    assert np.array_equal(y, np.eye(10, dtype=np.float32)[etiquetas])
    assert y[:, 8].sum() == 0
    assert y[:, 5].sum() == 271
    print("  ok  one-hot de diez salidas; 8 ausente y 271 muestras del 5")


def test_particion():
    X, y = datos.cargar(TRAIN)
    etiquetas = y.argmax(axis=1)
    itr, iva = indices_particion(y, datos.SEMILLA)
    assert len(np.intersect1d(itr, iva)) == 0
    assert np.array_equal(np.sort(np.concatenate([itr, iva])), np.arange(len(y)))
    assert abs(len(iva) / len(y) - 0.2) < 0.01
    for d in np.unique(etiquetas):
        assert np.sum(etiquetas[itr] == d) > 0 and np.sum(etiquetas[iva] == d) > 0
        assert abs(np.mean(etiquetas[itr] == d) - np.mean(etiquetas == d)) < 0.005
    Xtr, ytr, Xva, yva = datos.particionar(X, y)
    assert np.array_equal(Xtr, X[itr]) and np.array_equal(Xva, X[iva])
    assert np.array_equal(ytr, y[itr]) and np.array_equal(yva, y[iva])
    print(f"  ok  partición {len(itr)}/{len(iva)} completa, disjunta y estratificada")


def test_particion_reproducible():
    _, y = datos.cargar(TRAIN)
    a, b, c = (indices_particion(y, semilla) for semilla in (3, 3, 4))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1])
    print("  ok  misma semilla = misma partición; otra semilla = otra")


def test_test_set_aparte():
    Xa, _ = datos.cargar(TRAIN)
    Xb, _ = datos.cargar(TEST)
    claves = {fila.tobytes() for fila in Xa}
    repetidas = sum(fila.tobytes() in claves for fila in Xb)
    assert repetidas == 0, f"{repetidas} imágenes de test están en train"
    print("  ok  sin imágenes idénticas entre train y test")


if __name__ == "__main__":
    for fn in (test_forma_y_rango, test_cache_consistente, test_cache_desde_csv,
               test_one_hot, test_particion, test_particion_reproducible, test_test_set_aparte):
        print(f"\n== {fn.__name__}", flush=True)
        fn()
    print("\nTodos los chequeos del paso 1 pasaron.")
