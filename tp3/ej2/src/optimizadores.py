"""Optimizadores independientes de la arquitectura y del cálculo del gradiente."""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class Optimizador(Protocol):
    """Interfaz compartida por SGD y los futuros momentum/Adam.

    Los parámetros se actualizan in-place, en el orden recibido. Cada llamada
    representa un paso sobre un lote; los gradientes ya están promediados.
    """

    def paso(self, parametros: Sequence[np.ndarray],
             gradientes: Sequence[np.ndarray]) -> None: ...

    def configuracion(self) -> dict: ...


class SGD:
    def __init__(self, eta: float = 0.01) -> None:
        if not np.isfinite(eta) or eta <= 0:
            raise ValueError("eta debe ser positiva y finita")
        self.eta = float(eta)

    def paso(self, parametros: Sequence[np.ndarray],
             gradientes: Sequence[np.ndarray]) -> None:
        if len(parametros) != len(gradientes):
            raise ValueError("Debe haber un gradiente por parámetro")
        # Validar todo antes de modificar cualquier parámetro.
        for parametro, gradiente in zip(parametros, gradientes):
            if parametro.shape != gradiente.shape:
                raise ValueError("La forma del gradiente no coincide con el parámetro")
            if not np.all(np.isfinite(gradiente)):
                raise ValueError("El gradiente contiene valores no finitos")
        for parametro, gradiente in zip(parametros, gradientes):
            parametro -= self.eta * gradiente

    def configuracion(self) -> dict:
        return {"nombre": "sgd", "eta": self.eta}


def construir_optimizador(configuracion: dict) -> Optimizador:
    if configuracion["nombre"] == "sgd":
        return SGD(eta=configuracion["eta"])
    raise ValueError(f"Optimizador desconocido: {configuracion['nombre']!r}")
