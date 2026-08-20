"""Bucle generico de busqueda, IDDFS y wrapper de metricas."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .estado import ABREVIATURAS, Estado, Mapa
from .frontera import (
    Frontera,
    FronteraFIFO,
    FronteraLIFO,
    frontera_a_estrella,
    frontera_greedy,
)
from .heuristicas import HEURISTICA_POR_DEFECTO, obtener_heuristica
from .nodo import Heuristica, Nodo, expandir, reconstruir_camino, reconstruir_estados

_FRECUENCIA_CHEQUEO = 2048

EXITO = "exito"
SIN_SOLUCION = "sin_solucion"
LIMITE_NODOS = "limite_nodos"
TIMEOUT = "timeout"


@dataclass
class Resultado:
    algoritmo: str
    heuristica: Optional[str]
    exito: bool
    motivo: str
    costo: Optional[int]
    nodos_expandidos: int
    nodos_frontera: int
    nodos_generados: int
    tiempo_seg: float
    camino: List[str] = field(default_factory=list)
    estados: List[Estado] = field(default_factory=list, repr=False)

    @property
    def camino_corto(self) -> str:
        return "".join(ABREVIATURAS[accion] for accion in self.camino)

    def como_dict(self) -> Dict:
        return {
            "algoritmo": self.algoritmo,
            "heuristica": self.heuristica,
            "exito": self.exito,
            "motivo": self.motivo,
            "costo": self.costo,
            "nodos_expandidos": self.nodos_expandidos,
            "nodos_frontera": self.nodos_frontera,
            "nodos_generados": self.nodos_generados,
            "tiempo_seg": round(self.tiempo_seg, 6),
            "camino": self.camino,
            "camino_corto": self.camino_corto,
        }


class _Limites:
    """Cortes de seguridad para que una corrida no quede colgada."""

    def __init__(self, max_nodos: Optional[int], timeout: Optional[float]) -> None:
        self.max_nodos = max_nodos
        self.timeout = timeout
        self.inicio = time.perf_counter()

    def excedido(self, expandidos: int) -> Optional[str]:
        if self.max_nodos is not None and expandidos >= self.max_nodos:
            return LIMITE_NODOS
        if self.timeout is not None:
            if time.perf_counter() - self.inicio > self.timeout:
                return TIMEOUT
        return None


def buscar(
    estado_inicial: Estado,
    mapa: Mapa,
    crear_frontera: Callable[[], Frontera],
    heuristica: Optional[Heuristica] = None,
    podar_deadlocks: bool = True,
    limites: Optional[_Limites] = None,
) -> Tuple[Optional[Nodo], str, int, int, int]:
    """Devuelve `(nodo_objetivo, motivo, expandidos, tam_frontera, generados)`."""
    limites = limites or _Limites(None, None)

    h_inicial = heuristica(estado_inicial, mapa) if heuristica is not None else 0
    frontera = crear_frontera()
    frontera.agregar(Nodo(estado_inicial, padre=None, accion=None, g=0, h=h_inicial))

    visitados = set()
    expandidos = 0
    generados = 1

    while not frontera.esta_vacia():
        nodo = frontera.sacar()

        # Un estado puede entrar a la frontera por varios caminos antes de ser
        # procesado: se descarta al sacarlo, no solo al agregarlo.
        if nodo.estado in visitados:
            continue
        visitados.add(nodo.estado)
        expandidos += 1

        if nodo.estado.es_objetivo(mapa):
            return nodo, EXITO, expandidos, len(frontera), generados

        if expandidos % _FRECUENCIA_CHEQUEO == 0:
            motivo = limites.excedido(expandidos)
            if motivo is not None:
                return None, motivo, expandidos, len(frontera), generados

        for hijo in expandir(nodo, mapa, heuristica, podar_deadlocks):
            if hijo.estado not in visitados:
                frontera.agregar(hijo)
                generados += 1

    return None, SIN_SOLUCION, expandidos, 0, generados


def buscar_iddfs(
    estado_inicial: Estado,
    mapa: Mapa,
    podar_deadlocks: bool = True,
    limites: Optional[_Limites] = None,
    profundidad_maxima: Optional[int] = None,
) -> Tuple[Optional[Nodo], str, int, int, int]:
    """DFS con limite de profundidad creciente: memoria de DFS, optimo como BFS."""
    limites = limites or _Limites(None, None)
    expandidos_total = 0
    generados_total = 1

    for limite in itertools.count(0):
        if profundidad_maxima is not None and limite > profundidad_maxima:
            return None, SIN_SOLUCION, expandidos_total, 0, generados_total

        pila: List[Nodo] = [Nodo(estado_inicial, padre=None, accion=None, g=0)]
        # Profundidad minima con la que se vio cada estado en esta iteracion.
        mejor_profundidad: Dict[Estado, int] = {estado_inicial: 0}
        hubo_corte = False

        while pila:
            nodo = pila.pop()
            expandidos_total += 1

            if nodo.estado.es_objetivo(mapa):
                return nodo, EXITO, expandidos_total, len(pila), generados_total

            if expandidos_total % _FRECUENCIA_CHEQUEO == 0:
                motivo = limites.excedido(expandidos_total)
                if motivo is not None:
                    return None, motivo, expandidos_total, len(pila), generados_total

            if nodo.g >= limite:
                hubo_corte = True
                continue

            for hijo in expandir(nodo, mapa, None, podar_deadlocks):
                previa = mejor_profundidad.get(hijo.estado)
                if previa is None or hijo.g < previa:
                    mejor_profundidad[hijo.estado] = hijo.g
                    pila.append(hijo)
                    generados_total += 1

        # Sin cortes por profundidad, el espacio alcanzable esta agotado.
        if not hubo_corte:
            return None, SIN_SOLUCION, expandidos_total, 0, generados_total

    raise AssertionError("inalcanzable")


@dataclass(frozen=True)
class DescripcionAlgoritmo:
    clave: str
    nombre: str
    usa_heuristica: bool
    optimo: bool
    crear_frontera: Optional[Callable[[], Frontera]]


ALGORITMOS: Dict[str, DescripcionAlgoritmo] = {
    d.clave: d
    for d in (
        DescripcionAlgoritmo("bfs", "BFS", False, True, FronteraFIFO),
        DescripcionAlgoritmo("dfs", "DFS", False, False, FronteraLIFO),
        DescripcionAlgoritmo("iddfs", "IDDFS", False, True, None),
        DescripcionAlgoritmo("greedy", "Greedy", True, False, frontera_greedy),
        DescripcionAlgoritmo("astar", "A*", True, True, frontera_a_estrella),
    )
}

ORDEN_ALGORITMOS: Tuple[str, ...] = ("bfs", "dfs", "iddfs", "greedy", "astar")


def ejecutar_busqueda(
    clave_algoritmo: str,
    estado_inicial: Estado,
    mapa: Mapa,
    heuristica: str = HEURISTICA_POR_DEFECTO,
    podar_deadlocks: bool = True,
    max_nodos: Optional[int] = None,
    timeout: Optional[float] = None,
    guardar_estados: bool = True,
) -> Resultado:
    """Corre un algoritmo y mide todo lo que pide el enunciado."""
    if clave_algoritmo not in ALGORITMOS:
        raise KeyError(
            "algoritmo desconocido: {!r}. Opciones: {}".format(
                clave_algoritmo, ", ".join(ORDEN_ALGORITMOS)
            )
        )

    descripcion = ALGORITMOS[clave_algoritmo]
    funcion_h = None
    nombre_h = None
    if descripcion.usa_heuristica:
        elegida = obtener_heuristica(heuristica)
        funcion_h = elegida.funcion
        nombre_h = elegida.nombre

    inicio = time.perf_counter()
    limites = _Limites(max_nodos, timeout)

    if descripcion.clave == "iddfs":
        nodo, motivo, expandidos, en_frontera, generados = buscar_iddfs(
            estado_inicial, mapa, podar_deadlocks, limites
        )
    else:
        nodo, motivo, expandidos, en_frontera, generados = buscar(
            estado_inicial,
            mapa,
            descripcion.crear_frontera,
            funcion_h,
            podar_deadlocks,
            limites,
        )

    tiempo = time.perf_counter() - inicio
    exito = nodo is not None

    return Resultado(
        algoritmo=descripcion.nombre,
        heuristica=nombre_h,
        exito=exito,
        motivo=motivo,
        costo=nodo.g if exito else None,
        nodos_expandidos=expandidos,
        nodos_frontera=en_frontera,
        nodos_generados=generados,
        tiempo_seg=tiempo,
        camino=reconstruir_camino(nodo) if exito else [],
        estados=reconstruir_estados(nodo) if (exito and guardar_estados) else [],
    )


def comparar_algoritmos(
    estado_inicial: Estado,
    mapa: Mapa,
    claves: Optional[List[str]] = None,
    heuristica: str = HEURISTICA_POR_DEFECTO,
    podar_deadlocks: bool = True,
    max_nodos: Optional[int] = None,
    timeout: Optional[float] = None,
) -> List[Resultado]:
    """Corre varios algoritmos sobre el mismo tablero (nada se muta entre corridas)."""
    claves = list(claves) if claves else list(ORDEN_ALGORITMOS)
    return [
        ejecutar_busqueda(
            clave,
            estado_inicial,
            mapa,
            heuristica=heuristica,
            podar_deadlocks=podar_deadlocks,
            max_nodos=max_nodos,
            timeout=timeout,
            guardar_estados=False,
        )
        for clave in claves
    ]
