"""Definicion publica del problema de Sokoban."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .estado import Direccion, Estado, Mapa, ORDEN_ACCIONES


@dataclass(frozen=True)
class ProblemaSokoban:
    """Mapa, estado inicial y orden determinista de acciones de un problema."""

    mapa: Mapa
    estado_inicial: Estado
    orden_acciones: Tuple[Direccion, ...] = ORDEN_ACCIONES

    def __post_init__(self) -> None:
        if self.estado_inicial.jugador not in self.mapa.pisos:
            raise ValueError("el jugador debe estar sobre un piso valido")
        if not self.estado_inicial.cajas <= self.mapa.pisos:
            raise ValueError("todas las cajas deben estar sobre pisos validos")
        if not self.estado_inicial.cajas:
            raise ValueError("el problema debe tener al menos una caja")
        if len(self.estado_inicial.cajas) != len(self.mapa.objetivos):
            raise ValueError("la cantidad de cajas y objetivos debe coincidir")
        if set(self.orden_acciones) != set(ORDEN_ACCIONES):
            raise ValueError(
                "orden_acciones debe contener arriba, abajo, izquierda y derecha"
            )
        if len(self.orden_acciones) != len(ORDEN_ACCIONES):
            raise ValueError("orden_acciones no puede contener acciones repetidas")


__all__ = ["ProblemaSokoban"]
