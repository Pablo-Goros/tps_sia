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


def main() -> None:
    test_derivadas_numericas()
    print("Todas las validaciones de activaciones pasaron.")


if __name__ == "__main__":
    main()
