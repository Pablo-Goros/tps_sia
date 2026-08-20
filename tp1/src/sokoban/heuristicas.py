"""Heuristicas admisibles con asignacion uno a uno entre cajas y objetivos."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, FrozenSet, Sequence

from .distancias import distancia_manhattan
from .estado import Estado, Posicion
from .problema import ProblemaSokoban

Heuristica = Callable[[Estado], float]
DistanciaCajaObjetivo = Callable[[Posicion, Posicion], float]
FabricaHeuristica = Callable[[ProblemaSokoban], Heuristica]


def costo_asignacion_minima(matriz: Sequence[Sequence[float]]) -> float:
    """Costo minimo de una asignacion uno a uno mediante DP con bitmask."""
    costos = tuple(tuple(float(costo) for costo in fila) for fila in matriz)
    cantidad = len(costos)
    if cantidad == 0:
        return 0.0
    if any(len(fila) != cantidad for fila in costos):
        raise ValueError("la matriz de asignacion debe ser cuadrada")

    @lru_cache(maxsize=None)
    def resolver_fila(indice_caja: int, objetivos_usados: int) -> float:
        if indice_caja == cantidad:
            return 0.0

        mejor = math.inf
        for indice_objetivo, costo in enumerate(costos[indice_caja]):
            mascara = 1 << indice_objetivo
            if objetivos_usados & mascara:
                continue
            restante = resolver_fila(
                indice_caja + 1, objetivos_usados | mascara
            )
            mejor = min(mejor, costo + restante)
        return mejor

    return resolver_fila(0, 0)


class _HeuristicaAsignacion:
    """Heuristica con una cache privada para una unica ejecucion."""

    def __init__(
        self,
        problema: ProblemaSokoban,
        distancia: DistanciaCajaObjetivo,
    ) -> None:
        self._objetivos = tuple(sorted(problema.mapa.objetivos))
        self._distancia = distancia
        self._calcular_cacheado = lru_cache(maxsize=None)(self._calcular)

    def __call__(self, estado: Estado) -> float:
        return self._calcular_cacheado(estado.cajas)

    def _calcular(self, cajas: FrozenSet[Posicion]) -> float:
        cajas_ordenadas = tuple(sorted(cajas))
        matriz = tuple(
            tuple(self._distancia(caja, objetivo) for objetivo in self._objetivos)
            for caja in cajas_ordenadas
        )
        return costo_asignacion_minima(matriz)

    def estadisticas_cache(self):
        """Expone estadisticas para diagnostico y pruebas de aislamiento."""
        return self._calcular_cacheado.cache_info()


def _crear_manhattan(problema: ProblemaSokoban) -> Heuristica:
    return _HeuristicaAsignacion(
        problema,
        lambda caja, objetivo: float(distancia_manhattan(caja, objetivo)),
    )


def _crear_empujes_inversos(problema: ProblemaSokoban) -> Heuristica:
    return _HeuristicaAsignacion(
        problema,
        lambda caja, objetivo: float(
            problema.distancias_empujes[objetivo].get(caja, math.inf)
        ),
    )


@dataclass(frozen=True)
class DescripcionHeuristica:
    nombre: str
    fabrica: FabricaHeuristica
    admisible: bool
    descripcion: str


HEURISTICAS: Dict[str, DescripcionHeuristica] = {
    descripcion.nombre: descripcion
    for descripcion in (
        DescripcionHeuristica(
            "manhattan",
            _crear_manhattan,
            True,
            "asignacion optima caja-objetivo con distancia Manhattan",
        ),
        DescripcionHeuristica(
            "empujes_inversos",
            _crear_empujes_inversos,
            True,
            "asignacion optima con distancias de empujes que respetan paredes",
        ),
    )
}

HEURISTICA_POR_DEFECTO = "empujes_inversos"


def obtener_heuristica(nombre: str) -> DescripcionHeuristica:
    if nombre not in HEURISTICAS:
        raise KeyError(
            "heuristica desconocida: {!r}. Opciones: {}".format(
                nombre, ", ".join(sorted(HEURISTICAS))
            )
        )
    return HEURISTICAS[nombre]


def crear_heuristica(nombre: str, problema: ProblemaSokoban) -> Heuristica:
    """Crea una heuristica y su cache exclusiva para una ejecucion."""
    return obtener_heuristica(nombre).fabrica(problema)


__all__ = [
    "DescripcionHeuristica",
    "FabricaHeuristica",
    "HEURISTICAS",
    "HEURISTICA_POR_DEFECTO",
    "Heuristica",
    "costo_asignacion_minima",
    "crear_heuristica",
    "obtener_heuristica",
]
