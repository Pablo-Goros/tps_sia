"""Funciones de activación para el perceptrón simple (TP3 - SIA).

Cada activación expone:
  - ``theta(h)``   : la función de activación propiamente dicha.
  - ``dtheta(h,o)``: su derivada, calculada a partir de la pre-activación ``h``
                     y/o de la salida ``o = theta(h)`` (lo que resulte más barato).
  - ``imagen``     : el rango de salida, necesario para saber si hay que
                     escalar los valores deseados antes de entrenar.

Las fórmulas siguen las Clases 10.1 y 10.2 de la cátedra:
    escalón     theta(h) = signo(h) en {-1, 1}    (Rosenblatt: Δw = η(ζ-O)x, sin derivada)
    lineal      theta(h) = h                      theta'(h) = 1
    logística   theta(h) = 1 / (1 + e^(-2*beta*h))  theta'(h) = 2*beta*theta(h)*(1 - theta(h))
    tanh        theta(h) = tanh(beta*h)             theta'(h) = beta*(1 - theta(h)^2)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Activacion:
    nombre: str
    theta: Callable[[np.ndarray], np.ndarray]
    dtheta: Callable[[np.ndarray, np.ndarray], np.ndarray]
    imagen: Optional[Tuple[float, float]]  # None => no acotada (lineal, relu)

    @property
    def acotada(self) -> bool:
        return self.imagen is not None


def _sigmoide_estable(z: np.ndarray) -> np.ndarray:
    """1/(1+e^-z) sin overflow para z muy negativo."""
    salida = np.empty_like(z, dtype=float)
    pos = z >= 0
    salida[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    salida[~pos] = ez / (1.0 + ez)
    return salida


def construir_activacion(nombre: str, beta: float = 0.5) -> Activacion:
    nombre = nombre.lower()

    if nombre in ("escalon", "escalón", "signo"):
        # No es derivable; dtheta = 1 hace que la regla general Δw = η(ζ-O)θ'(h)x
        # se reduzca a la regla de Rosenblatt de la Clase 10.1.
        return Activacion(
            nombre="escalon",
            theta=lambda h: np.where(h >= 0.0, 1.0, -1.0),
            dtheta=lambda h, o: np.ones_like(h),
            imagen=(-1.0, 1.0),
        )

    if nombre == "lineal":
        return Activacion(
            nombre="lineal",
            theta=lambda h: h,
            dtheta=lambda h, o: np.ones_like(h),
            imagen=None,
        )

    if nombre in ("logistica", "logística", "sigmoide"):
        return Activacion(
            nombre="logistica",
            theta=lambda h: _sigmoide_estable(2.0 * beta * h),
            dtheta=lambda h, o: 2.0 * beta * o * (1.0 - o),
            imagen=(0.0, 1.0),
        )

    if nombre == "tanh":
        return Activacion(
            nombre="tanh",
            theta=lambda h: np.tanh(beta * h),
            dtheta=lambda h, o: beta * (1.0 - o ** 2),
            imagen=(-1.0, 1.0),
        )

    if nombre == "relu":  # opcional del enunciado
        return Activacion(
            nombre="relu",
            theta=lambda h: np.maximum(0.0, h),
            dtheta=lambda h, o: (h > 0).astype(float),
            imagen=None,
        )

    raise ValueError(f"Activación desconocida: {nombre!r}")
