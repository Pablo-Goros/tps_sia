"""Patrones de letras 5x5 en {1, -1} (enunciado 2.2, ejemplo de la letra J).

'*' = 1 (pixel activo), '.' = -1 (inactivo).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .hopfield import orthogonality_matrix

_FONT = {
    "A": ["..*..", ".*.*.", "*...*", "*****", "*...*"],
    "B": ["****.", "*...*", "****.", "*...*", "****."],
    "C": [".****", "*....", "*....", "*....", ".****"],
    "D": ["****.", "*...*", "*...*", "*...*", "****."],
    "E": ["*****", "*....", "****.", "*....", "*****"],
    "F": ["*****", "*....", "****.", "*....", "*...."],
    "G": [".****", "*....", "*..**", "*...*", ".****"],
    "H": ["*...*", "*...*", "*****", "*...*", "*...*"],
    "I": ["*****", "..*..", "..*..", "..*..", "*****"],
    "J": ["*****", "...*.", "...*.", "*..*.", "***.."],   # exactamente la del enunciado
    "K": ["*...*", "*..*.", "***..", "*..*.", "*...*"],
    "L": ["*....", "*....", "*....", "*....", "*****"],
    "M": ["*...*", "**.**", "*.*.*", "*...*", "*...*"],
    "N": ["*...*", "**..*", "*.*.*", "*..**", "*...*"],
    "O": [".***.", "*...*", "*...*", "*...*", ".***."],
    "P": ["****.", "*...*", "****.", "*....", "*...."],
    "Q": [".***.", "*...*", "*.*.*", "*..*.", ".**.*"],
    "R": ["****.", "*...*", "****.", "*..*.", "*...*"],
    "S": [".****", "*....", ".***.", "....*", "****."],
    "T": ["*****", "..*..", "..*..", "..*..", "..*.."],
    "U": ["*...*", "*...*", "*...*", "*...*", ".***."],
    "V": ["*...*", "*...*", "*...*", ".*.*.", "..*.."],
    "W": ["*...*", "*...*", "*.*.*", "**.**", "*...*"],
    "X": ["*...*", ".*.*.", "..*..", ".*.*.", "*...*"],
    "Y": ["*...*", ".*.*.", "..*..", "..*..", "..*.."],
    "Z": ["*****", "...*.", "..*..", ".*...", "*****"],
}


def letter(ch: str) -> np.ndarray:
    """Matriz 5x5 con 1 / -1."""
    rows = _FONT[ch.upper()]
    return np.array([[1 if c == "*" else -1 for c in row] for row in rows], dtype=float)


def letter_vector(ch: str) -> np.ndarray:
    return letter(ch).reshape(-1)


def all_letters() -> dict[str, np.ndarray]:
    return {ch: letter_vector(ch) for ch in _FONT}


def to_ascii(vec: np.ndarray, size: int = 5) -> str:
    m = np.asarray(vec).reshape(size, size)
    return "\n".join("".join("*" if v > 0 else "." for v in row) for row in m)


def best_orthogonal_subset(n: int = 4, letters: str | None = None) -> list[tuple[float, float, tuple[str, ...]]]:
    """Busca los grupos de `n` letras con menor crosstalk.
    Ordena por (max |xi.xi'|, media |xi.xi'|) de a pares. Devuelve lista ordenada."""
    chars = list(letters) if letters else list(_FONT)
    vecs = {c: letter_vector(c) for c in chars}
    results = []
    for combo in combinations(chars, n):
        G = orthogonality_matrix(np.array([vecs[c] for c in combo]))
        off = np.abs(G[~np.eye(n, dtype=bool)])
        results.append((float(off.max()), float(off.mean()), combo))
    results.sort(key=lambda r: (r[0], r[1]))
    return results
