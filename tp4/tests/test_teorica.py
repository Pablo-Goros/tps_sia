"""Verificación de las implementaciones contra los ejemplos numéricos de las teóricas.
Correr con:  python3 -m pytest tests/   (o  python3 tests/test_teorica.py)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.hopfield import Hopfield                      # noqa: E402
from src.kohonen import Kohonen, KohonenConfig         # noqa: E402
from src.oja import Oja, OjaConfig, cosine             # noqa: E402
from src.letras import letter                          # noqa: E402


def test_hopfield_ejemplo_teorica():
    """Teórica Hopfield, slides 26-32: xi1=(1,1,-1,-1), xi2=(-1,-1,1,1)."""
    xi1, xi2 = np.array([1, 1, -1, -1.0]), np.array([-1, -1, 1, 1.0])
    net = Hopfield().train(np.array([xi1, xi2]))
    W_esperada = np.array([[0, .5, -.5, -.5], [.5, 0, -.5, -.5], [-.5, -.5, 0, .5], [-.5, -.5, .5, 0]])
    assert np.allclose(net.W, W_esperada), net.W            # w12 = 0.5, w13 = -0.5 (slide 28)
    r = net.recall(np.array([1, -1, -1, -1.0]))               # consulta de la slide 31
    assert np.array_equal(r.states[1], xi1)                   # S1 = sign(W zeta) = xi1
    assert np.array_equal(r.final, xi1) and r.converged       # se mantiene estable (slide 32)
    assert r.energies[-1] <= r.energies[0]                    # la energía no crece


def test_hopfield_letra_J_enunciado():
    J = np.array([[1, 1, 1, 1, 1], [-1, -1, -1, 1, -1], [-1, -1, -1, 1, -1], [1, -1, -1, 1, -1], [1, 1, 1, -1, -1]])
    assert np.array_equal(letter("J"), J)


def test_pca_ejercicio_teorica():
    """Teórica PCA, slides 28-31: S = [[3,1,1],[1,3,1],[1,1,5]] -> autovalores 6, 3, 2."""
    S = np.array([[3, 1, 1], [1, 3, 1], [1, 1, 5.0]])
    vals, vecs = np.linalg.eigh(S)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    assert np.allclose(vals, [6, 3, 2])
    assert np.allclose(np.abs(vecs[:, 0]), [0.408, 0.408, 0.816], atol=1e-2)   # (-0.40, -0.40, -0.81)
    assert np.allclose(np.abs(vecs[:, 1]), [0.577, 0.577, 0.577], atol=1e-2)   # (-0.57, -0.57, 0.57)
    assert np.allclose(vals / vals.sum(), [6 / 11, 3 / 11, 2 / 11])


def test_oja_converge_a_pc1():
    rng = np.random.default_rng(0)
    # datos gaussianos correlacionados, centrados
    C = np.array([[3, 1.5, 0.2], [1.5, 2, 0.1], [0.2, 0.1, 0.5]])
    X = rng.multivariate_normal(np.zeros(3), C, size=500)
    X -= X.mean(axis=0)
    pc1 = np.linalg.eigh(np.cov(X.T))[1][:, -1]
    w = Oja(3, OjaConfig(eta0=1e-3, epochs=100, seed=0)).train(X).w
    assert abs(cosine(w, pc1)) > 0.995
    assert abs(np.linalg.norm(w) - 1) < 0.05


def test_kohonen_agrupa_clusters_separados():
    rng = np.random.default_rng(0)
    A = rng.normal([0, 0], 0.1, size=(30, 2))
    B = rng.normal([5, 5], 0.1, size=(30, 2))
    X = np.vstack([A, B])
    som = Kohonen(2, KohonenConfig(k=2, radius0=2, eta0=0.5, epochs=50, seed=0)).train(X)
    wa, wb = som.winners(A), som.winners(B)
    assert len(set(wa)) == 1 and len(set(wb)) == 1 and wa[0] != wb[0]
    assert som.quantization_error(X) < 0.2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("OK", name)
