"""Tablas precalculadas del mapa: se cachean porque `Mapa` es inmutable."""

from __future__ import annotations

from collections import deque
from functools import lru_cache
from typing import Dict, FrozenSet

from .estado import DIRECCIONES, Mapa, Posicion

INFINITO = float("inf")


def distancia_manhattan(a: Posicion, b: Posicion) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@lru_cache(maxsize=None)
def tabla_empujes(mapa: Mapa) -> Dict[Posicion, Dict[Posicion, int]]:
    """Para cada objetivo, los empujes minimos desde cada celda.

    BFS hacia atras: una caja en `q` pudo venir de `q - d` si `q - d` no es
    pared y el jugador pudo pararse en `q - 2d`. Ignora las otras cajas, por lo
    que el valor es un limite inferior de los empujes reales.
    """
    tabla: Dict[Posicion, Dict[Posicion, int]] = {}
    for objetivo in mapa.objetivos:
        distancias = {objetivo: 0}
        cola = deque([objetivo])
        while cola:
            actual = cola.popleft()
            for dx, dy in DIRECCIONES.values():
                origen_caja = (actual[0] - dx, actual[1] - dy)
                origen_jugador = (actual[0] - 2 * dx, actual[1] - 2 * dy)
                if origen_caja in mapa.paredes or origen_jugador in mapa.paredes:
                    continue
                if origen_caja in distancias:
                    continue
                distancias[origen_caja] = distancias[actual] + 1
                cola.append(origen_caja)
        tabla[objetivo] = distancias
    return tabla


@lru_cache(maxsize=None)
def distancia_minima_a_objetivo(mapa: Mapa) -> Dict[Posicion, int]:
    tabla = tabla_empujes(mapa)
    minimos: Dict[Posicion, int] = {}
    for distancias in tabla.values():
        for celda, distancia in distancias.items():
            if distancia < minimos.get(celda, INFINITO):
                minimos[celda] = distancia
    return minimos


@lru_cache(maxsize=None)
def celdas_muertas(mapa: Mapa) -> FrozenSet[Posicion]:
    """Celdas desde las que una caja ya no puede llegar a ningun objetivo.

    Si una celda no aparece en ninguna tabla de empujes, no hay forma de sacar
    la caja de ahi: empujarla a esa celda hace el nivel irresoluble.
    """
    alcanzables = set(distancia_minima_a_objetivo(mapa))
    muertas = set()
    for y in range(mapa.alto):
        for x in range(mapa.ancho):
            celda = (x, y)
            if celda in mapa.paredes or celda in alcanzables:
                continue
            muertas.add(celda)
    return frozenset(muertas)
