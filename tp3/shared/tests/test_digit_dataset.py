"""Synthetic loader checks: python -m tps_sia.tp3.shared.tests.test_digit_dataset."""
from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from tps_sia.tp3.shared import digit_dataset as datos


def test_cache_desde_csv():
    # Dataset pequeño para verificar un caché nuevo y su invalidación.
    with TemporaryDirectory() as directorio:
        ruta = Path(directorio) / "digits.csv"
        cache = Path(directorio) / "digits.npz"

        def escribir(etiquetas):
            with ruta.open("w", newline="", encoding="utf-8") as archivo:
                escritor = csv.writer(archivo)
                escritor.writerow(["image", "label"])
                for digito in etiquetas:
                    escritor.writerow([str([0.25] * 784), digito])

        escribir([0, 5])
        X1, y1 = datos.cargar(ruta, cache)
        assert np.all(X1 == np.float32(0.25))
        assert np.array_equal(y1.argmax(axis=1), [0, 5])
        with np.load(cache, allow_pickle=False) as guardado:
            assert np.array_equal(guardado["X"], X1)
            assert np.array_equal(guardado["y"], y1)
        with patch.object(datos.csv, "DictReader", side_effect=AssertionError("Se reparseó el CSV")):
            X2, y2 = datos.cargar(ruta, cache)
        assert np.array_equal(X1, X2) and np.array_equal(y1, y2)
        escribir([0, 5, 8])
        X3, y3 = datos.cargar(ruta, cache)
        assert X3.shape == (3, 784) and np.array_equal(y3.argmax(axis=1), [0, 5, 8])
    print("  ok  caché nuevo guarda X/y y se invalida al cambiar el CSV")


def indices_particion(y, semilla):
    # Identificadores únicos para comprobar qué filas selecciona particionar.
    identificadores = np.zeros((len(y), 784), dtype=np.float32)
    identificadores[:, 0] = np.arange(len(y))
    Xtr, ytr, Xva, yva = datos.particionar(identificadores, y, semilla)
    itr, iva = Xtr[:, 0].astype(int), Xva[:, 0].astype(int)
    assert np.array_equal(ytr, y[itr]) and np.array_equal(yva, y[iva])
    return itr, iva


def test_particion_sintetica():
    # Uneven class counts, including class 8: no dependency on course datasets.
    labels = np.repeat([0, 5, 8], [11, 7, 2])
    y = np.eye(datos.N_CLASES, dtype=np.float32)[labels]
    train, val = indices_particion(y, datos.SEMILLA)
    assert not np.intersect1d(train, val).size
    np.testing.assert_array_equal(np.sort(np.r_[train, val]), np.arange(len(y)))
    np.testing.assert_array_equal(np.bincount(labels[val], minlength=10)[[0, 5, 8]], [2, 1, 1])
    for digit in np.unique(labels):
        assert np.any(labels[train] == digit) and np.any(labels[val] == digit)
    other_train, other_val = indices_particion(y, datos.SEMILLA)
    np.testing.assert_array_equal(train, other_train)
    np.testing.assert_array_equal(val, other_val)
    assert not np.array_equal(train, indices_particion(y, 0)[0])


def test_cache_source_and_paths():
    with TemporaryDirectory() as directory:
        base = Path(directory)
        cache = base / "exercise-cache" / "digits.npz"
        first, second = (base / name / "digits.csv" for name in ("first", "second"))
        for path, label in ((first, 0), (second, 8)):
            path.parent.mkdir()
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["image", "label"])
                writer.writerow([str([0.25] * 784), label])
        # Same name and size must not reuse a different dataset's cache.
        datos.cargar(first, cache)
        _, y = datos.cargar(second, cache)
        np.testing.assert_array_equal(y.argmax(axis=1), [8])
        with patch.object(datos.csv, "DictReader", side_effect=AssertionError("CSV reparsed")):
            datos.cargar(second, cache)
        assert sorted(base.rglob("*.npz")) == [cache]
        try:
            datos.cargar(first, first)
        except ValueError:
            pass
        else:
            raise AssertionError("Cache must not overwrite the source")


def main() -> None:
    for check in (test_cache_desde_csv, test_particion_sintetica, test_cache_source_and_paths):
        print(f"\n== {check.__name__}")
        check()
    print("Todos los chequeos del loader común pasaron.")


if __name__ == "__main__":
    main()


