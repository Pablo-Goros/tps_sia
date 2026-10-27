"""Desgastes "realistas" de un QR, parametrizados por la fracción f de módulos que cubren.

Cada desgaste pinta una región de un color fijo: blanco (-1, se perdió la tinta) o negro (+1, mancha).
Solo cambian los módulos que no eran de ese color, así que el número de módulos realmente cambiados
(`np.sum(z != xi)`) es menor que f*N; se mide aparte. Posición, ángulo y forma salen del generador `rng`.

  esquina rota  triángulo blanco en una esquina al azar (papel rasgado)
  mancha        disco negro en un lugar al azar
  rayón         franja blanca recta de ancho f*S, con ángulo y posición al azar
  borrado       rectángulo blanco de relación de aspecto al azar
  desvanecido   los módulos oscuros pasan a claros con probabilidad f / (fracción de oscuros)
"""
from __future__ import annotations

import numpy as np

S = 37


def _grid(size):
    return np.mgrid[0:size, 0:size]


def esquina_rota(f, rng, size=S):
    ii, jj = _grid(size)
    L = np.sqrt(2 * f * size * size)
    c = rng.integers(4)
    di = ii if c in (0, 1) else size - 1 - ii
    dj = jj if c in (0, 2) else size - 1 - jj
    return (di + 0.5) + (dj + 0.5) < L, -1


def mancha(f, rng, size=S):
    ii, jj = _grid(size)
    r = np.sqrt(f * size * size / np.pi)
    lo, hi = min(r, size / 2), max(size - r, size / 2)
    ci, cj = rng.uniform(lo, hi, 2)
    return (ii + 0.5 - ci) ** 2 + (jj + 0.5 - cj) ** 2 <= r ** 2, +1


def rayon(f, rng, size=S):
    ii, jj = _grid(size)
    w = f * size
    th = rng.uniform(0, np.pi)
    ci, cj = rng.uniform(0.25 * size, 0.75 * size, 2)
    return np.abs((ii + 0.5 - ci) * np.cos(th) - (jj + 0.5 - cj) * np.sin(th)) < w / 2, -1


def borrado(f, rng, size=S):
    ii, jj = _grid(size)
    n = size * size
    a = rng.uniform(0.5, 2.0)
    h = min(size, np.sqrt(f * n * a))
    w = min(size, f * n / h)
    i0 = rng.uniform(0, size - h)
    j0 = rng.uniform(0, size - w)
    return (ii + 0.5 >= i0) & (ii + 0.5 < i0 + h) & (jj + 0.5 >= j0) & (jj + 0.5 < j0 + w), -1


# desgastes con región (máscara, color); el desvanecido se aplica aparte
TIPOS = {"esquina rota": esquina_rota, "mancha": mancha, "rayón": rayon, "borrado": borrado}
TODOS = list(TIPOS) + ["desvanecido"]


def desvanecido(xi, f, rng):
    oscuros = xi > 0
    q = min(1.0, f / oscuros.mean())
    z = xi.copy()
    z[oscuros & (rng.random(xi.size) < q)] = -1
    return z


def aplicar(tipo, xi, f, rng, size=S):
    """Devuelve el QR `xi` (vector {-1,1} de size*size) deteriorado con el desgaste `tipo` y severidad f."""
    if tipo == "desvanecido":
        return desvanecido(xi, f, rng)
    mascara, color = TIPOS[tipo](f, rng, size)
    z = xi.copy()
    z[mascara.reshape(-1)] = color
    return z
