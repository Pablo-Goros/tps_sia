"""Chequeos del MLP: python -m tps_sia.tp3.shared.tests.test_mlp."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tps_sia.tp3.shared.mlp import MLP
from tps_sia.tp3.shared.optimizers import SGD


X_XOR = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]], dtype=float)
Y_XOR = np.array([1, 1, 0, 0], dtype=float)  # (-1,+1) -> (0,1) para logística


def comprobar_gradientes(modelo, X, y, paso=1e-6):
    """Diferencias centradas sobre TODOS los pesos y biases del costo público."""
    originales = [p.copy() for p in modelo.parametros]
    analiticos = modelo.backprop(X, y)
    errores = []
    for parametro, analitico in zip(modelo.parametros, analiticos):
        numerico = np.empty_like(parametro)
        for indice in np.ndindex(parametro.shape):
            valor = parametro[indice]
            try:
                parametro[indice] = valor + paso
                mas = modelo.costo(X, y)
                parametro[indice] = valor - paso
                menos = modelo.costo(X, y)
            finally:
                parametro[indice] = valor
            numerico[indice] = (mas - menos) / (2 * paso)
        np.testing.assert_allclose(analitico, numerico, atol=2e-8, rtol=2e-5)
        errores.append(float(np.max(np.abs(analitico - numerico))))
    for antes, despues in zip(originales, modelo.parametros):
        np.testing.assert_array_equal(antes, despues)
    return max(errores)


def test_gradient_check():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(5, 2))
    for activacion in ("tanh", "relu"):
        for salida in ("softmax", "logistica"):
            for arquitectura in ([2, 3, 2], [2, 3, 2, 3]):
                modelo = MLP(arquitectura, activacion=activacion, salida=salida,
                             beta=0.7, semilla=8)
                y = np.eye(arquitectura[-1])[np.arange(len(X)) % arquitectura[-1]]
                # ReLU no es diferenciable en cero; comprobar que las diferencias
                # no crucen ese punto (incluye neuronas activas e inactivas).
                if activacion == "relu":
                    for b in modelo.biases[:-1]:
                        b[:] = np.linspace(-0.37, 0.61, len(b))
                    _, nets = modelo._forward(X)
                    assert all(np.min(np.abs(h)) > 1e-4 for h in nets[:-1])
                error = comprobar_gradientes(modelo, X, y)
                comprobar_gradientes(modelo, X[:1], y[:1])
                print(f"  ok  gradientes {arquitectura}, {activacion}/{salida}: error={error:.2e}")
    for arquitectura in ([2, 2, 1], [2, 3, 2, 1]):
        modelo = MLP(arquitectura, salida="logistica", beta=1.3, semilla=4)
        comprobar_gradientes(modelo, X, rng.uniform(size=(len(X), 1)))


def test_xor():
    for arquitectura in ([2, 2, 1], [2, 3, 2, 1]):
        modelo = MLP(arquitectura, salida="logistica", eta=0.3,
                     tamano_lote=2, semilla=0)
        inicial = modelo.costo(X_XOR, Y_XOR)
        historia = modelo.entrenar(X_XOR, Y_XOR, epocas=4000, epsilon=0.001)
        np.testing.assert_array_equal(modelo.predecir_clases(X_XOR), Y_XOR)
        # Comparación con los objetivos originales del enunciado.
        np.testing.assert_array_equal(2 * modelo.predecir_clases(X_XOR) - 1, [1, 1, -1, -1])
        assert historia.costo[-1] < 0.001 and historia.costo[-1] < inicial / 10
        assert historia.accuracy[-1] == 1
        print(f"  ok  XOR {arquitectura}: {historia.epocas_corridas} épocas, MSE={historia.mse[-1]:.6f}")
    # Comprobar también aprendizaje con la combinación por defecto.
    modelo = MLP([2, 3, 2], eta=0.1, tamano_lote=2, semilla=0)
    modelo.entrenar(X_XOR, np.eye(2)[Y_XOR.astype(int)], epocas=1000, epsilon=0.01)
    np.testing.assert_array_equal(modelo.predecir_clases(X_XOR), Y_XOR)


def test_softmax_estable():
    modelo = MLP([2, 3], semilla=0)
    modelo.pesos[0][:] = [[1000, 0, -1000], [0, 0, 0]]
    modelo.biases[0][:] = 0
    X = np.array([[1., 0], [-1., 0]])
    y = np.eye(3)[[2, 0]]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        o = modelo.forward(X)
        assert np.all(np.isfinite(o))
        np.testing.assert_allclose(o.sum(axis=1), 1)
        np.testing.assert_allclose(modelo.costo(X, y), 2000)
        assert all(np.all(np.isfinite(g)) for g in modelo.backprop(X, y))
        # El gradiente sigue siendo correcto aunque p de la clase real sea cero.
        comprobar_gradientes(modelo, X, y, paso=1e-3)
    modelo.biases[0] += 5000
    np.testing.assert_array_equal(modelo.forward(X), o)


def test_sgd_y_mini_batches():
    parametro = np.array([1., 2.])
    SGD(0.2).paso([parametro], [np.array([3., -4.])])
    np.testing.assert_allclose(parametro, [0.4, 2.8])
    X = np.vstack([X_XOR, [[0.2, -0.4]]])
    y = np.eye(2)[[1, 1, 0, 0, 1]]
    for lote in (1, 2, None, 20):
        modelo = MLP([2, 3, 2], eta=0.1, tamano_lote=lote, semilla=5)
        referencia = MLP([2, 3, 2], eta=0.1, tamano_lote=lote, semilla=5)
        k = len(X) if lote is None else lote
        for inicio in range(0, len(X), k):
            g = referencia.backprop(X[inicio:inicio + k], y[inicio:inicio + k])
            for p, grad in zip(referencia.parametros, g):
                p -= 0.1 * grad
        hist = modelo.entrenar(X, y, epocas=1, mezclar=False)
        for actual, esperado in zip(modelo.parametros, referencia.parametros):
            np.testing.assert_allclose(actual, esperado, atol=1e-15)
        np.testing.assert_allclose(hist.costo[-1], modelo.costo(X, y))
    # Repetir un lote no cambia el gradiente promedio (normalización por N).
    g = modelo.backprop(X, y)
    g_repetido = modelo.backprop(np.tile(X, (3, 1)), np.tile(y, (3, 1)))
    for original, repetido in zip(g, g_repetido):
        np.testing.assert_allclose(original, repetido, atol=1e-15)


def test_historia_y_validacion():
    y = np.eye(2)[Y_XOR.astype(int)]
    modelo = MLP([2, 3, 2], semilla=2)
    control = MLP([2, 3, 2], semilla=2)
    hist = modelo.entrenar(X_XOR, y, epocas=3, X_val=X_XOR, y_val=1 - y)
    control.entrenar(X_XOR, y, epocas=3)
    for actual, esperado in zip(modelo.parametros, control.parametros):
        np.testing.assert_array_equal(actual, esperado)  # validación no actualiza ni altera RNG
    for clave, valor in hist.to_dict().items():
        if isinstance(valor, list):
            assert len(valor) == 3 and np.all(np.isfinite(valor))
    assert hist.epocas_corridas == 3 and hist.tiempo_segundos > 0
    np.testing.assert_allclose(hist.costo_validacion[-1], modelo.costo(X_XOR, 1 - y))


def test_guardar_cargar_y_continuar():
    for salida, arquitectura in (("softmax", [2, 3, 2]), ("logistica", [2, 3, 2, 1])):
        y = np.eye(2)[Y_XOR.astype(int)] if salida == "softmax" else Y_XOR
        modelo = MLP(arquitectura, activacion="relu", salida=salida, beta=0.7,
                     eta=0.15, tamano_lote=3, inicializacion="he", semilla=17)
        modelo.entrenar(X_XOR, y, epocas=3, X_val=X_XOR, y_val=y)
        with TemporaryDirectory() as directorio:
            ruta = Path(directorio) / "modelo.npz"
            modelo.guardar(ruta)
            copia = MLP.cargar(ruta)
        np.testing.assert_array_equal(modelo.predecir(X_XOR), copia.predecir(X_XOR))
        assert copia.historia.to_dict() == modelo.historia.to_dict()
        assert copia.optimizador.configuracion() == modelo.optimizador.configuracion()
        assert copia.beta == modelo.beta and copia.tamano_lote == modelo.tamano_lote
        modelo.entrenar(X_XOR, y, epocas=4)
        copia.entrenar(X_XOR, y, epocas=4)
        for actual, esperado in zip(copia.parametros, modelo.parametros):
            np.testing.assert_array_equal(actual, esperado)  # también conserva estado RNG
        assert copia.historia.costo == modelo.historia.costo


def test_inicializacion():
    for activacion in ("tanh", "relu"):
        for metodo in ("auto", "xavier", "he"):
            modelo = MLP([200, 300, 10], activacion=activacion,
                         inicializacion=metodo, semilla=4)
            replica = MLP([200, 300, 10], activacion=activacion,
                          inicializacion=metodo, semilla=4)
            for i, (n_in, n_out) in enumerate(zip(modelo.arquitectura, modelo.arquitectura[1:])):
                he = metodo == "he" or (metodo == "auto" and activacion == "relu" and i == 0)
                escala = np.sqrt(2 / n_in) if he else np.sqrt(2 / (n_in + n_out))
                assert abs(modelo.pesos[i].std() / escala - 1) < 0.06
                assert abs(modelo.pesos[i].mean()) < 0.06 * escala
                np.testing.assert_array_equal(modelo.pesos[i], replica.pesos[i])
                assert np.all(modelo.biases[i] == 0)
    modelo = MLP([784, 128, 10], activacion="relu")
    X = np.random.default_rng(42).uniform(size=(3, 784))
    o = modelo.predecir(X)
    assert o.shape == (3, 10)
    np.testing.assert_allclose(o.sum(axis=1), 1)
    assert all(g.shape == p.shape for g, p in
               zip(modelo.backprop(X, np.eye(10)[[0, 1, 2]]), modelo.parametros))


def test_entradas_invalidas():
    casos = [{"arquitectura": [2, 1]}, {"arquitectura": [2, 0, 2]},
             {"arquitectura": [2.5, 2]}, {"arquitectura": [2, 2], "tamano_lote": 0},
             {"arquitectura": [2, 2], "activacion": "escalon"},
             {"arquitectura": [2, 2], "eta": float("nan")}]
    for kwargs in casos:
        try:
            MLP(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Se aceptó configuración inválida: {kwargs}")
    modelo = MLP([2, 3, 2])
    originales = [p.copy() for p in modelo.parametros]
    for X, y, kwargs in ((X_XOR, np.zeros((4, 2)), {}),
                          (X_XOR, np.eye(2)[[1, 1, 0, 0]], {"X_val": X_XOR}),
                          (np.zeros((0, 2)), np.zeros((0, 2)), {}),
                          (X_XOR, np.eye(2)[[1, 1, 0, 0]], {"epocas": -1})):
        try:
            modelo.entrenar(X, y, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("Se aceptaron datos de entrenamiento inválidos")
    for antes, despues in zip(originales, modelo.parametros):
        np.testing.assert_array_equal(antes, despues)


def main() -> None:
    for fn in (test_gradient_check, test_xor, test_softmax_estable,
               test_sgd_y_mini_batches, test_historia_y_validacion,
               test_guardar_cargar_y_continuar, test_inicializacion, test_entradas_invalidas):
        print(f"\n== {fn.__name__}")
        fn()
    print("\nTodas las validaciones del MLP pasaron.")


if __name__ == "__main__":
    main()
