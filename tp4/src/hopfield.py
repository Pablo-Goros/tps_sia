"""Red de Hopfield discreta (teórica 03 - Modelo de Hopfield).

Algoritmo (slides "Algoritmo"):
  1. Almacenamiento: xi^1..xi^p patrones binarios {-1, 1} de dimensión N.
  2. Pesos sinápticos:
        w_ij = (1/N) sum_mu xi_i^mu xi_j^mu   si i != j
        w_ii = 0
     En forma matricial, con K = (xi^1 ... xi^p) en columnas:  W = (1/N) K K^T
     y se anula la diagonal. W es simétrica.
  3. Inicialización: S(0) = zeta (patrón de consulta).
  4. Iteración hasta la convergencia:  S_i(t+1) = sign( sum_{j != i} w_ij S_j(t) )
     Si h_i = 0 la neurona conserva su estado previo.
  5. Output: el estado estable S_fijo es el patrón asociado a zeta.

Energía (slide "Convergencia"):  H = -1/2 sum_{i,j} w_ij S_i S_j
  Siempre decrece (o permanece constante) cuando el sistema evoluciona;
  los mínimos locales son los patrones almacenados... y también pueden
  aparecer otros mínimos: ESTADOS ESPURIOS.

Limitaciones: p <= 0.15 N y patrones "más o menos" ortogonales (crosstalk).

Extensión (no está en la teórica): regla de la pseudoinversa, train(..., rule="pinv"),
  W = K^T (K K^T)^-1 K, que descuenta la correlación entre patrones (ver DECISIONES.md, 2.3).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RecallResult:
    states: list                      # S(0), S(1), ..., S(T)
    energies: list                    # H(S(t)) para cada t
    converged: bool                   # se alcanzó un estado estable
    cycle: bool                       # se detectó un ciclo (típico del modo sincrónico)
    iterations: int                   # cantidad de actualizaciones de S hasta frenar
    final: np.ndarray = field(init=False)

    def __post_init__(self):
        self.final = self.states[-1]


class Hopfield:
    def __init__(self, update: str = "sync", seed: int | None = 0):
        """update: 'sync'  -> todas las neuronas a la vez, S(t+1) = sign(W S(t))
                           (es la forma usada en el ejemplo de la teórica)
                   'async' -> de a una neurona, en orden aleatorio. Con este modo
                           la demostración de que la energía nunca crece aplica
                           exactamente (cambia un S_i por vez)."""
        assert update in ("sync", "async")
        self.update = update
        self.rng = np.random.default_rng(seed)
        self.W: np.ndarray | None = None
        self.patterns: np.ndarray | None = None   # (p, N)
        self.N: int = 0
        self.rule: str = "hebb"

    # ---------------------------------------------------------------- train
    def train(self, patterns: np.ndarray, rule: str = "hebb", zero_diag: bool | None = None) -> "Hopfield":
        """rule: 'hebb' -> W = (1/N) K^T K              (por defecto, diagonal en 0)
                 'pinv' -> W = K^T (K K^T)^-1 K         (pseudoinversa; Personnaz / Kanter-Sompolinsky)
        K tiene un patrón por fila. Con 'pinv' W es la proyección ortogonal sobre el espacio que
        generan los patrones: W xi^mu = xi^mu, así que cada patrón almacenado es punto fijo si los
        patrones son linealmente independientes. Si son ortogonales (K K^T = N I) coincide con Hebb.
        zero_diag: anular w_ii. Por defecto True para 'hebb' y False para 'pinv' (con la diagonal
        la energía de un patrón es exactamente -N/2, que es el piso -1/2 ||W S||^2 >= -N/2)."""
        assert rule in ("hebb", "pinv")
        K = np.asarray(patterns, dtype=float)
        if K.ndim != 2:
            K = K.reshape(len(patterns), -1)
        assert set(np.unique(K)).issubset({-1.0, 1.0}), "los patrones deben ser {-1, 1}"
        self.patterns = K
        p, N = K.shape
        self.N = N
        self.rule = rule
        if rule == "hebb":
            W = (K.T @ K) / N            # (1/N) sum_mu xi_i xi_j
        else:
            if np.linalg.matrix_rank(K) < p:
                raise ValueError("pinv requiere patrones linealmente independientes (K K^T singular)")
            W = K.T @ np.linalg.inv(K @ K.T) @ K
            W = (W + W.T) / 2            # simetría exacta (el producto la tiene salvo redondeo)
        if zero_diag if zero_diag is not None else rule == "hebb":
            np.fill_diagonal(W, 0.0)     # w_ii = 0
        self.W = W
        return self

    # --------------------------------------------------------------- energy
    def energy(self, S: np.ndarray) -> float:
        return float(-0.5 * S @ self.W @ S)

    # --------------------------------------------------------------- recall
    @staticmethod
    def _sign_keep(h: np.ndarray, prev: np.ndarray) -> np.ndarray:
        """sign(h); si h == 0 se mantiene el estado previo (slide 'Modelo de Hopfield')."""
        out = np.sign(h)
        zero = out == 0
        out[zero] = prev[zero]
        return out

    def recall(self, zeta: np.ndarray, max_iter: int = 100) -> RecallResult:
        S = np.asarray(zeta, dtype=float).reshape(-1).copy()
        states, energies = [S.copy()], [self.energy(S)]
        seen = {S.tobytes(): 0}
        converged = cycle = False
        it = 0
        for it in range(1, max_iter + 1):
            if self.update == "sync":
                S_new = self._sign_keep(self.W @ S, S)
            else:
                S_new = S.copy()
                for i in self.rng.permutation(self.N):
                    h = self.W[i] @ S_new
                    if h != 0:
                        S_new[i] = np.sign(h)
            states.append(S_new.copy())
            energies.append(self.energy(S_new))
            if np.array_equal(S_new, S):
                converged = True
                break
            key = S_new.tobytes()
            if key in seen:            # volvió a un estado anterior -> ciclo
                cycle = True
                S = S_new
                break
            seen[key] = it
            S = S_new
        return RecallResult(states, energies, converged, cycle, it)

    # ------------------------------------------------------------- analysis
    def match(self, S: np.ndarray) -> tuple[int | None, bool]:
        """Devuelve (índice del patrón almacenado igual a S, es_negativo).
        (None, False) si S no coincide con ningún patrón ni con su opuesto
        -> estado espurio (u otro punto fijo)."""
        for mu, xi in enumerate(self.patterns):
            if np.array_equal(S, xi):
                return mu, False
            if np.array_equal(S, -xi):
                return mu, True
        return None, False

    def is_spurious(self, S: np.ndarray) -> bool:
        return self.match(S)[0] is None

    def overlaps(self, S: np.ndarray) -> np.ndarray:
        """m_mu = (1/N) xi^mu . S  en [-1, 1]; 1 = idéntico al patrón mu."""
        return self.patterns @ S / self.N

    def capacity_limit(self) -> float:
        return 0.15 * self.N


# ---------------------------------------------------------------- utilities
def add_noise(pattern: np.ndarray, p_flip: float, rng: np.random.Generator) -> np.ndarray:
    """Invierte cada componente con probabilidad p_flip (ruido 'sal y pimienta')."""
    x = np.asarray(pattern, dtype=float).copy()
    flip = rng.random(x.shape) < p_flip
    x[flip] *= -1
    return x


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int((np.asarray(a) != np.asarray(b)).sum())


def orthogonality_matrix(patterns: np.ndarray) -> np.ndarray:
    """Producto interno entre pares de patrones: (K K^T)_{mu,nu} = xi^mu . xi^nu.
    0 = ortogonales; +-N = iguales/opuestos. Es el término de crosstalk."""
    K = np.asarray(patterns, dtype=float).reshape(len(patterns), -1)
    return K @ K.T


def orthogonality_score(patterns: np.ndarray) -> dict:
    """Resumen: media y máximo de |xi^mu . xi^nu| / N fuera de la diagonal."""
    G = orthogonality_matrix(patterns)
    N = np.asarray(patterns).reshape(len(patterns), -1).shape[1]
    off = np.abs(G[~np.eye(len(G), dtype=bool)]) / N
    return {"mean_abs": float(off.mean()), "max_abs": float(off.max()), "gram": G}
