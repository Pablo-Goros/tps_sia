"""Checks for the exercise wrappers: python -m tps_sia.tp3.ej2.tests.test_shared_compatibility."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from tps_sia.tp3.ej1.src import activaciones as old_activations
from tps_sia.tp3.ej1.src.perceptron import PerceptronSimple
from tps_sia.tp3.ej2.src import baseline, datos_digitos, mlp, optimizadores
from tps_sia.tp3.shared import activations, digit_dataset, optimizers
from tps_sia.tp3.shared.mlp import Historia, MLP


def test_shared_identity():
    assert mlp.MLP is baseline.MLP is MLP
    assert mlp.Historia is Historia
    assert optimizadores.SGD is baseline.SGD is optimizers.SGD
    assert optimizadores.Momentum is optimizers.Momentum
    assert optimizadores.Adam is optimizers.Adam
    assert optimizadores.Optimizador is optimizers.Optimizador
    assert optimizadores.construir_optimizador is optimizers.construir_optimizador
    assert old_activations.Activacion is activations.Activacion
    assert old_activations.construir_activacion is activations.construir_activacion
    assert datos_digitos.particionar is baseline.particionar is digit_dataset.particionar
    assert datos_digitos._cargar is digit_dataset.cargar
    assert baseline.cargar is digit_dataset.cargar
    assert isinstance(PerceptronSimple(2).activacion, activations.Activacion)


def test_exercise_cache_defaults():
    expected = (np.zeros((1, 784), dtype=np.float32), np.eye(10, dtype=np.float32)[:1])
    # Patch the delegate, leaving the wrapper's path selection real.
    with patch.object(datos_digitos, "_cargar", return_value=expected) as loader:
        assert datos_digitos.cargar() is expected
        loader.assert_called_with(datos_digitos.DIGITS.resolve(), datos_digitos.CACHE)
        with TemporaryDirectory() as directory:
            source = Path(directory) / "other_digits.csv"
            explicit = Path(directory) / "ej3-cache" / "digits.npz"
            datos_digitos.cargar(source, explicit)
            loader.assert_called_with(source.resolve(), explicit)
            datos_digitos.cargar(source)
            default_cache = loader.call_args.args[1]
            assert default_cache.parent == datos_digitos.CACHE.parent
            assert default_cache != datos_digitos.CACHE
            datos_digitos.cargar(source)
            assert loader.call_args.args[1] == default_cache
    assert datos_digitos.CACHE.parent == datos_digitos.EJERCICIO / "cache"
    assert baseline.RESULTS.parent == datos_digitos.EJERCICIO / "results"


def main() -> None:
    test_shared_identity()
    test_exercise_cache_defaults()
    print("Todos los chequeos de compatibilidad pasaron.")


if __name__ == "__main__":
    main()
