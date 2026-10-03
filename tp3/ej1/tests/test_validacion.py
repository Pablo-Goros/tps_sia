"""Ejercicios de validación del enunciado + chequeos de las derivadas.

    python -m tps_sia.tp3.ej1.tests.test_validacion
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tps_sia.tp3.shared.activations import construir_activacion
from tps_sia.tp3.shared.tests.test_activations import test_derivadas_numericas
from tps_sia.tp3.ej1.src.perceptron import PerceptronSimple


# Entradas de las funciones lógicas del enunciado (ejercicio de validación).
X_LOGICA = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]], dtype=float)


def test_escalon_aprende_and():
    """Enunciado: el perceptrón simple escalón debe resolver el AND (linealmente separable)."""
    y = np.array([-1, -1, -1, 1], dtype=float)
    m = PerceptronSimple(2, "escalon", eta=0.1, tamano_lote=1, semilla=0)
    h = m.entrenar(X_LOGICA, y, epocas=50, epsilon=1e-12, mezclar=True)  # corta con error 0
    assert np.array_equal(m.predecir(X_LOGICA), y)
    print(f"  ok  escalón resuelve AND en {h.epocas_corridas} épocas, w={np.round(m.w, 3)}")


def test_escalon_no_puede_con_xor():
    """Clase 11: el XOR no es linealmente separable -> un perceptrón simple no lo resuelve."""
    y = np.array([1, 1, -1, -1], dtype=float)
    for s in range(10):
        m = PerceptronSimple(2, "escalon", eta=0.1, tamano_lote=1, semilla=s)
        m.entrenar(X_LOGICA, y, epocas=200, mezclar=True)
        aciertos = int(np.sum(m.predecir(X_LOGICA) == y))
        assert aciertos < 4, "un perceptrón simple no debería resolver el XOR"
    print("  ok  escalón no resuelve XOR (ninguna de 10 semillas acierta las 4 salidas)")


def test_perceptron_lineal_ajusta_una_recta():
    """Clase 10.2: 50 muestras de una recta deben ajustarse casi exactamente."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(50, 1))
    y = (2.0 * X[:, 0] + 1.0)
    m = PerceptronSimple(1, "lineal", eta=0.05, tamano_lote=1, semilla=0)
    h = m.entrenar(X, y, epocas=300, mezclar=True)
    print(f"  ok  lineal sobre y=2x+1: MSE={h.mse[-1]:.2e}, "
          f"w=[{m.w[0]:.4f}, {m.w[1]:.4f}] (esperado [1, 2])")
    assert h.mse[-1] < 1e-6
    assert abs(m.w[0] - 1.0) < 1e-2 and abs(m.w[1] - 2.0) < 1e-2


def test_perceptron_logistico_recupera_su_propia_sigmoide():
    """Si los datos vienen de theta(w·x), el perceptrón debe recuperar w."""
    rng = np.random.default_rng(1)
    X = rng.uniform(-2, 2, size=(200, 2))
    w_real = np.array([0.5, 1.5, -1.0])  # [bias, w1, w2]
    act = construir_activacion("logistica", 0.5)
    y = act.theta(w_real[0] + X @ w_real[1:])
    m = PerceptronSimple(2, "logistica", eta=0.5, beta=0.5, tamano_lote=1, semilla=0)
    h = m.entrenar(X, y, epocas=800, mezclar=True)
    print(f"  ok  logística sobre sigmoid(w*x): MSE={h.mse[-1]:.2e}, "
          f"w={np.round(m.w, 3)} (esperado {w_real})")
    assert h.mse[-1] < 1e-5
    assert np.max(np.abs(m.w - w_real)) < 0.1


def test_perceptron_tanh_ajusta_tanh():
    """Enunciado: 50 muestras de y = tanh(x) deben ajustarse con activación tanh."""
    rng = np.random.default_rng(3)
    X = rng.uniform(-3, 3, size=(50, 1))
    y = np.tanh(X[:, 0])
    m = PerceptronSimple(1, "tanh", eta=0.1, beta=1.0, tamano_lote=1, semilla=0)
    h = m.entrenar(X, y, epocas=500, mezclar=True)
    print(f"  ok  tanh sobre y=tanh(x): MSE={h.mse[-1]:.2e}, "
          f"w=[{m.w[0]:.4f}, {m.w[1]:.4f}] (esperado [0, 1])")
    assert h.mse[-1] < 1e-6
    assert abs(m.w[0]) < 1e-2 and abs(m.w[1] - 1.0) < 1e-2


def test_salida_logistica_dentro_de_0_1():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 50, size=(500, 4))  # entradas grandes a propósito
    m = PerceptronSimple(4, "logistica", beta=0.5, semilla=0)
    o = m.predecir(X)
    assert np.all((o >= 0.0) & (o <= 1.0)) and np.all(np.isfinite(o))
    print("  ok  la logística nunca sale de [0,1] ni desborda")


def test_guardar_y_cargar():
    m = PerceptronSimple(3, "logistica", eta=0.1, beta=0.5, semilla=7)
    X = np.random.default_rng(0).normal(size=(10, 3))
    antes = m.predecir(X)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "model.npz"
        m.guardar(path)
        otro = PerceptronSimple.cargar(path)
    assert np.allclose(antes, otro.predecir(X))
    print("  ok  guardar/cargar preserva el modelo")


if __name__ == "__main__":
    for fn in (test_derivadas_numericas, test_escalon_aprende_and, test_escalon_no_puede_con_xor,
               test_perceptron_lineal_ajusta_una_recta,
               test_perceptron_logistico_recupera_su_propia_sigmoide, test_perceptron_tanh_ajusta_tanh,
               test_salida_logistica_dentro_de_0_1, test_guardar_y_cargar):
        print(f"\n== {fn.__name__}")
        fn()
    print("\nTodas las validaciones pasaron.")
