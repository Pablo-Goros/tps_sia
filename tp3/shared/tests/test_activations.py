"""Activation checks: python -m tps_sia.tp3.shared.tests.test_activations."""
import numpy as np

from tps_sia.tp3.shared.activations import construir_activacion


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
            print(f"  ok  theta'({nombre}, beta={beta})  error max = {err:.2e}")


def test_derivada_relu():
    """ReLU: theta'(h) vs. diferencia centrada fuera de h = 0, donde no es derivable.

    En h = 0 la implementación usa la convención theta'(0) = 0 (la muestra no corrige pesos).
    """
    act = construir_activacion("relu")
    h = np.linspace(-3, 3, 61)
    h = h[np.abs(h) > 1e-3]
    eps = 1e-6
    analitica = act.dtheta(h, act.theta(h))
    numerica = (act.theta(h + eps) - act.theta(h - eps)) / (2 * eps)
    err = np.max(np.abs(analitica - numerica))
    assert err < 1e-6, f"relu: error {err}"
    cero = np.array([0.0])
    assert act.theta(cero)[0] == 0.0 and act.dtheta(cero, act.theta(cero))[0] == 0.0
    print(f"  ok  theta'(relu)  error max = {err:.2e}  (theta'(0) = 0 por convención)")


def main() -> None:
    test_derivadas_numericas()
    test_derivada_relu()
    print("Todas las validaciones de activaciones pasaron.")


if __name__ == "__main__":
    main()
