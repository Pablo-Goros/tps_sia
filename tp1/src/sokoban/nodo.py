"""Nodo del arbol de busqueda y reconstruccion del camino."""

from __future__ import annotations

from typing import Callable, List, Optional

from .estado import Direccion, Estado, Mapa
from .sucesores import generar_sucesores

Heuristica = Callable[[Estado, Mapa], float]


class Nodo:
    """Va separado de `Estado` para no romper la deteccion de repetidos.

    Dos nodos con el mismo tablero alcanzado por caminos distintos tienen
    distinto `g` y distinto padre, pero son el mismo estado.
    """

    __slots__ = ("estado", "padre", "accion", "g", "h")

    def __init__(
        self,
        estado: Estado,
        padre: Optional["Nodo"] = None,
        accion: Optional[Direccion] = None,
        g: int = 0,
        h: float = 0,
    ) -> None:
        self.estado = estado
        self.padre = padre
        self.accion = accion
        self.g = g
        self.h = h

    @property
    def f(self) -> float:
        return self.g + self.h

    def __lt__(self, otro: "Nodo") -> bool:
        return self.f < otro.f

    def __repr__(self) -> str:
        return "Nodo(g={}, h={}, accion={})".format(self.g, self.h, self.accion)


def expandir(
    nodo: Nodo,
    mapa: Mapa,
    heuristica: Optional[Heuristica] = None,
    podar_deadlocks: bool = True,
) -> List[Nodo]:
    hijos = []
    for estado, accion, costo in generar_sucesores(nodo.estado, mapa, podar_deadlocks):
        h = heuristica(estado, mapa) if heuristica is not None else 0
        hijos.append(Nodo(estado, padre=nodo, accion=accion, g=nodo.g + costo, h=h))
    return hijos


def reconstruir_camino(nodo: Optional[Nodo]) -> List[Direccion]:
    camino: List[Direccion] = []
    actual = nodo
    while actual is not None and actual.padre is not None:
        camino.append(actual.accion)
        actual = actual.padre
    camino.reverse()
    return camino


def reconstruir_estados(nodo: Optional[Nodo]) -> List[Estado]:
    estados: List[Estado] = []
    actual = nodo
    while actual is not None:
        estados.append(actual.estado)
        actual = actual.padre
    estados.reverse()
    return estados
