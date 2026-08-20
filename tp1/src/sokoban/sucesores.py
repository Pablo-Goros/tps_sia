"""Adaptadores temporales para la generacion de sucesores."""

from __future__ import annotations

from typing import List

from .estado import Estado, Mapa
from .problema import COSTO_MOVIMIENTO, ProblemaSokoban, Sucesor


def generar_sucesores(
    estado: Estado, mapa: Mapa, podar_deadlocks: bool = True
) -> List[Sucesor]:
    problema = ProblemaSokoban(mapa=mapa, estado_inicial=estado)
    return problema.generar_sucesores(estado, podar_deadlocks)


def es_deadlock(nuevo: Estado, anterior: Estado, mapa: Mapa, muertas=None) -> bool:
    problema = ProblemaSokoban(mapa=mapa, estado_inicial=anterior)
    return problema.es_deadlock(nuevo, anterior)


__all__ = [
    "COSTO_MOVIMIENTO",
    "Sucesor",
    "es_deadlock",
    "generar_sucesores",
]
