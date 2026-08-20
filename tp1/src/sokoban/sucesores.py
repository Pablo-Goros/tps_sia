"""Generacion de sucesores y poda de deadlocks."""

from __future__ import annotations

from typing import List, Optional, Tuple

from .distancias import celdas_muertas
from .estado import ORDEN_ACCIONES, Estado, Mapa, aplicar_accion, empuja

Sucesor = Tuple[Estado, str, int]

# Todo movimiento cuesta 1: el enunciado pide optimizar la cantidad de movimientos.
COSTO_MOVIMIENTO = 1


def generar_sucesores(
    estado: Estado, mapa: Mapa, podar_deadlocks: bool = True
) -> List[Sucesor]:
    muertas = celdas_muertas(mapa) if podar_deadlocks else frozenset()
    sucesores: List[Sucesor] = []

    for accion in ORDEN_ACCIONES:
        hubo_empuje = empuja(estado, accion)
        nuevo = aplicar_accion(estado, mapa, accion)
        if nuevo is None:
            continue
        if podar_deadlocks and hubo_empuje and es_deadlock(nuevo, estado, mapa, muertas):
            continue
        sucesores.append((nuevo, accion, COSTO_MOVIMIENTO))

    return sucesores


def es_deadlock(nuevo: Estado, anterior: Estado, mapa: Mapa, muertas) -> bool:
    """True si el ultimo empuje dejo el nivel sin solucion posible.

    Solo se mira la caja que se acaba de mover: el resto ya fue validado al
    generar el estado anterior. La poda es conservadora (descarta unicamente
    estados que con certeza no llevan a solucion), asi que no afecta la
    optimalidad de BFS ni de A*.
    """
    movida = _caja_movida(nuevo, anterior)
    if movida is None:
        return False
    if movida in muertas:
        return True
    return _bloque_congelado(nuevo, mapa, movida)


def _caja_movida(nuevo: Estado, anterior: Estado) -> Optional[tuple]:
    diferencia = nuevo.cajas - anterior.cajas
    return next(iter(diferencia)) if diferencia else None


def _bloque_congelado(estado: Estado, mapa: Mapa, caja) -> bool:
    """Bloque de 2x2 de paredes y cajas con alguna caja fuera de objetivo."""
    x, y = caja
    for x0 in (x - 1, x):
        for y0 in (y - 1, y):
            celdas = [(x0, y0), (x0 + 1, y0), (x0, y0 + 1), (x0 + 1, y0 + 1)]
            if all(c in mapa.paredes or c in estado.cajas for c in celdas):
                cajas_del_bloque = [c for c in celdas if c in estado.cajas]
                if any(c not in mapa.objetivos for c in cajas_del_bloque):
                    return True
    return False
