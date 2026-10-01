"""Ejercicios de validación del enunciado + chequeos de las derivadas.

    python3 -m tp3.tests.test_validacion
"""
from __future__ import annotations

import numpy as np

from tp3.src.activaciones import construir_activacion
from tp3.src.perceptron import PerceptronSimple


def test_derivadas_numericas():
    """theta'(h) analítica vs. diferencia centrada, para cada activación."""
    h = np.linspace(-3, 3, 61)
    eps = 1e-6
    for nombre in ("lineal", "logistica", "tanh"):
        for beta in (0.25, 0.5, 2.0):
            act = construir_activacion(nombre, beta)
            analitica = act.dtheta(h, act.theta(h))
            numerica = (act.theta(h + eps) - act.theta(h - eps)) / (2 * eps)
            err = np.max(np.abs(analitica - numerica))
            assert err < 1e-6, f"{nombre} beta={beta}: error {err}"
            print(f"  ok  θ'({nombre}, β={beta})  error máx = {err:.2e}")


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
    print(f"  ok  logística sobre σ(w·x): MSE={h.mse[-1]:.2e}, "
          f"w={np.round(m.w, 3)} (esperado {w_real})")
    assert h.mse[-1] < 1e-5
    assert np.max(np.abs(m.w - w_real)) < 0.1


def test_salida_logistica_dentro_de_0_1():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 50, size=(500, 4))  # entradas grandes a propósito
    m = PerceptronSimple(4, "logistica", beta=0.5, semilla=0)
    o = m.predecir(X)
    assert np.all((o >= 0.0) & (o <= 1.0)) and np.all(np.isfinite(o))
    print("  ok  la logística nunca sale de [0,1] ni desborda")


def test_guardar_y_cargar(tmp="/tmp/_tp3_modelo.npz"):
    m = PerceptronSimple(3, "logistica", eta=0.1, beta=0.5, semilla=7)
    X = np.random.default_rng(0).normal(size=(10, 3))
    antes = m.predecir(X)
    m.guardar(tmp)
    otro = PerceptronSimple.cargar(tmp)
    assert np.allclose(antes, otro.predecir(X))
    print("  ok  guardar/cargar preserva el modelo")


if __name__ == "__main__":
    for fn in (test_derivadas_numericas, test_perceptron_lineal_ajusta_una_recta,
               test_perceptron_logistico_recupera_su_propia_sigmoide,
               test_salida_logistica_dentro_de_0_1, test_guardar_y_cargar):
        print(f"\n== {fn.__name__}")
        fn()
    print("\nTodas las validaciones pasaron.")
