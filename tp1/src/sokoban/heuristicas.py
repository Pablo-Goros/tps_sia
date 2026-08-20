"""Heuristicas para Greedy y A*. La justificacion de admisibilidad esta en el README."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, FrozenSet, List

from .distancias import distancia_manhattan
from .estado import Estado, Mapa, Posicion

# Cota superior usada como infinito dentro del metodo hungaro.
GRANDE = 10 ** 6


def h_manhattan(estado: Estado, mapa: Mapa) -> float:
    """Suma de distancias Manhattan de cada caja a su objetivo mas cercano."""
    return _manhattan_cacheada(estado.cajas, mapa)


def h_manhattan_jugador(estado: Estado, mapa: Mapa) -> float:
    """`manhattan` + los pasos que el jugador necesita para llegar a una caja."""
    base = _manhattan_cacheada(estado.cajas, mapa)
    if base == 0:
        return 0.0
    cercania = min(distancia_manhattan(estado.jugador, caja) for caja in estado.cajas)
    return base + max(0, cercania - 1)


def h_matching(estado: Estado, mapa: Mapa) -> float:
    """Asignacion optima caja<->objetivo sobre distancias Manhattan."""
    return _matching_cacheado(estado.cajas, mapa)


# Las heuristicas dependen solo de las cajas: muchos estados comparten la misma
# configuracion (el jugador caminando), asi que cachear por `cajas` evita recalcular.


@lru_cache(maxsize=2 ** 19)
def _manhattan_cacheada(cajas: FrozenSet[Posicion], mapa: Mapa) -> float:
    objetivos = mapa.objetivos
    return float(
        sum(
            min(distancia_manhattan(caja, objetivo) for objetivo in objetivos)
            for caja in cajas
        )
    )


@lru_cache(maxsize=2 ** 19)
def _matching_cacheado(cajas: FrozenSet[Posicion], mapa: Mapa) -> float:
    objetivos = sorted(mapa.objetivos)
    matriz = [
        [distancia_manhattan(caja, objetivo) for objetivo in objetivos]
        for caja in sorted(cajas)
    ]
    return float(costo_asignacion_minima(matriz))


def costo_asignacion_minima(matriz: List[List[int]]) -> int:
    """Metodo hungaro O(n^3) sobre una matriz cuadrada de costos."""
    n = len(matriz)
    if n == 0:
        return 0
    m = len(matriz[0])

    u = [0] * (n + 1)
    v = [0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [GRANDE] * (m + 1)
        usado = [False] * (m + 1)
        while True:
            usado[j0] = True
            i0 = p[j0]
            delta = GRANDE
            j1 = 0
            for j in range(1, m + 1):
                if usado[j]:
                    continue
                actual = matriz[i0 - 1][j - 1] - u[i0] - v[j]
                if actual < minv[j]:
                    minv[j] = actual
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if usado[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    return sum(matriz[p[j] - 1][j - 1] for j in range(1, m + 1) if p[j])


@dataclass(frozen=True)
class DescripcionHeuristica:
    nombre: str
    funcion: Callable[[Estado, Mapa], float]
    admisible: bool
    descripcion: str


HEURISTICAS: Dict[str, DescripcionHeuristica] = {
    d.nombre: d
    for d in (
        DescripcionHeuristica(
            "manhattan", h_manhattan, True, "suma de Manhattan al objetivo mas cercano"
        ),
        DescripcionHeuristica(
            "manhattan_jugador",
            h_manhattan_jugador,
            True,
            "manhattan + distancia del jugador a la caja mas cercana",
        ),
        DescripcionHeuristica(
            "matching", h_matching, True, "asignacion optima caja<->objetivo (Manhattan)"
        ),
    )
}

HEURISTICA_POR_DEFECTO = "matching"


def obtener_heuristica(nombre: str) -> DescripcionHeuristica:
    if nombre not in HEURISTICAS:
        raise KeyError(
            "heuristica desconocida: {!r}. Opciones: {}".format(
                nombre, ", ".join(sorted(HEURISTICAS))
            )
        )
    return HEURISTICAS[nombre]
