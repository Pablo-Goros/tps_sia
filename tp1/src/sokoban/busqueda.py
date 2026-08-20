"""Bucle generico de busqueda, IDDFS y wrapper de metricas."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple

from .estado import ABREVIATURAS, Direccion, Estado, Mapa
from .frontera import (
    Frontera,
    FronteraFIFO,
    FronteraLIFO,
    frontera_a_estrella,
    frontera_greedy,
)
from .heuristicas import (
    HEURISTICA_POR_DEFECTO,
    Heuristica,
    crear_heuristica,
    obtener_heuristica,
)
from .nodo import Nodo, expandir, reconstruir_camino, reconstruir_estados
from .problema import ProblemaSokoban

EXITO = "exito"
SIN_SOLUCION = "sin_solucion"
LIMITE_NODOS = "limite_nodos"
TIMEOUT = "timeout"


EstadoBusqueda = Literal["exito", "fracaso", "corte"]


@dataclass(frozen=True)
class Resultado:
    estado: EstadoBusqueda
    algoritmo: str
    heuristica: Optional[str]
    costo: Optional[int]
    movimientos: Tuple[Direccion, ...]
    nodos_expandidos: int
    nodos_frontera: int
    max_nodos_frontera: int
    tiempo_segundos: float
    motivo: Optional[str]
    # Metricas y estados transitorios conservados hasta que las capas antiguas
    # de benchmark y visualizacion se retiren en fases posteriores.
    nodos_generados: int
    _estados: Tuple[Estado, ...] = field(default_factory=tuple, repr=False)

    @property
    def exito(self) -> bool:
        return self.estado == "exito"

    @property
    def camino(self) -> List[Direccion]:
        """Compatibilidad temporal con los consumidores anteriores."""
        return list(self.movimientos)

    @property
    def estados(self) -> List[Estado]:
        """Compatibilidad temporal con la capa de visualizacion anterior."""
        return list(self._estados)

    @property
    def tiempo_seg(self) -> float:
        """Nombre transitorio usado por los scripts anteriores."""
        return self.tiempo_segundos

    @property
    def camino_corto(self) -> str:
        return "".join(ABREVIATURAS[accion] for accion in self.movimientos)

    def como_dict(self) -> Dict:
        return {
            "algoritmo": self.algoritmo,
            "heuristica": self.heuristica,
            "estado": self.estado,
            "exito": self.exito,
            "motivo": self.motivo,
            "costo": self.costo,
            "movimientos": list(self.movimientos),
            "nodos_expandidos": self.nodos_expandidos,
            "nodos_frontera": self.nodos_frontera,
            "max_nodos_frontera": self.max_nodos_frontera,
            "nodos_generados": self.nodos_generados,
            "tiempo_segundos": round(self.tiempo_segundos, 6),
            "tiempo_seg": round(self.tiempo_segundos, 6),
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


@dataclass(frozen=True)
class _ResultadoBusqueda:
    nodo: Optional[Nodo]
    motivo: str
    expandidos: int
    en_frontera: int
    max_en_frontera: int
    generados: int

    def como_tupla_compatible(self) -> Tuple[Optional[Nodo], str, int, int, int]:
        return (
            self.nodo,
            self.motivo,
            self.expandidos,
            self.en_frontera,
            self.generados,
        )


def _validar_problema(
    problema: ProblemaSokoban, mapa: Mapa, estado_inicial: Estado
) -> None:
    if problema.mapa != mapa or problema.estado_inicial != estado_inicial:
        raise ValueError(
            "el problema debe contener el mapa y el estado inicial de la busqueda"
        )


def _resultado_busqueda(
    nodo: Optional[Nodo],
    motivo: str,
    expandidos: int,
    frontera: Frontera,
    max_en_frontera: int,
    generados: int,
) -> _ResultadoBusqueda:
    return _ResultadoBusqueda(
        nodo=nodo,
        motivo=motivo,
        expandidos=expandidos,
        en_frontera=len(frontera),
        max_en_frontera=max_en_frontera,
        generados=generados,
    )


def _buscar_bfs(
    problema: ProblemaSokoban,
    podar_deadlocks: bool,
    limites: _Limites,
) -> _ResultadoBusqueda:
    frontera = FronteraFIFO()
    frontera.agregar(Nodo(problema.estado_inicial))
    visitados = {problema.estado_inicial}
    expandidos = 0
    generados = 1
    max_en_frontera = 1

    while not frontera.esta_vacia():
        nodo = frontera.sacar()

        if nodo.estado.es_objetivo(problema.mapa):
            return _resultado_busqueda(
                nodo, EXITO, expandidos, frontera, max_en_frontera, generados
            )

        motivo = limites.excedido(expandidos)
        if motivo is not None:
            return _resultado_busqueda(
                None, motivo, expandidos, frontera, max_en_frontera, generados
            )

        hijos = expandir(nodo, problema, None, podar_deadlocks)
        expandidos += 1
        for hijo in hijos:
            if hijo.estado in visitados:
                continue
            visitados.add(hijo.estado)
            frontera.agregar(hijo)
            generados += 1
        max_en_frontera = max(max_en_frontera, len(frontera))

    return _resultado_busqueda(
        None,
        SIN_SOLUCION,
        expandidos,
        frontera,
        max_en_frontera,
        generados,
    )


def _buscar_dfs(
    problema: ProblemaSokoban,
    podar_deadlocks: bool,
    limites: _Limites,
) -> _ResultadoBusqueda:
    frontera = FronteraLIFO()
    frontera.agregar(Nodo(problema.estado_inicial))
    visitados = {problema.estado_inicial}
    expandidos = 0
    generados = 1
    max_en_frontera = 1

    while not frontera.esta_vacia():
        nodo = frontera.sacar()

        if nodo.estado.es_objetivo(problema.mapa):
            return _resultado_busqueda(
                nodo, EXITO, expandidos, frontera, max_en_frontera, generados
            )

        motivo = limites.excedido(expandidos)
        if motivo is not None:
            return _resultado_busqueda(
                None, motivo, expandidos, frontera, max_en_frontera, generados
            )

        hijos = expandir(nodo, problema, None, podar_deadlocks)
        expandidos += 1
        nuevos = []
        for hijo in hijos:
            if hijo.estado in visitados:
                continue
            visitados.add(hijo.estado)
            nuevos.append(hijo)

        # La pila debe recibir el ultimo sucesor primero para que la primera
        # accion configurada quede arriba y sea la proxima en explorarse.
        for hijo in reversed(nuevos):
            frontera.agregar(hijo)
            generados += 1
        max_en_frontera = max(max_en_frontera, len(frontera))

    return _resultado_busqueda(
        None,
        SIN_SOLUCION,
        expandidos,
        frontera,
        max_en_frontera,
        generados,
    )


def _buscar_greedy(
    problema: ProblemaSokoban,
    heuristica: Heuristica,
    podar_deadlocks: bool,
    limites: _Limites,
) -> _ResultadoBusqueda:
    frontera = frontera_greedy()
    h_inicial = heuristica(problema.estado_inicial)
    frontera.agregar(Nodo(problema.estado_inicial, h=h_inicial))
    visitados = {problema.estado_inicial}
    expandidos = 0
    generados = 1
    max_en_frontera = 1

    while not frontera.esta_vacia():
        nodo = frontera.sacar()

        if nodo.estado.es_objetivo(problema.mapa):
            return _resultado_busqueda(
                nodo, EXITO, expandidos, frontera, max_en_frontera, generados
            )

        motivo = limites.excedido(expandidos)
        if motivo is not None:
            return _resultado_busqueda(
                None, motivo, expandidos, frontera, max_en_frontera, generados
            )

        hijos = expandir(nodo, problema, heuristica, podar_deadlocks)
        expandidos += 1
        for hijo in hijos:
            if hijo.estado in visitados:
                continue
            visitados.add(hijo.estado)
            frontera.agregar(hijo)
            generados += 1
        max_en_frontera = max(max_en_frontera, len(frontera))

    return _resultado_busqueda(
        None,
        SIN_SOLUCION,
        expandidos,
        frontera,
        max_en_frontera,
        generados,
    )


def _buscar_a_estrella(
    problema: ProblemaSokoban,
    heuristica: Heuristica,
    podar_deadlocks: bool,
    limites: _Limites,
) -> _ResultadoBusqueda:
    frontera = frontera_a_estrella()
    h_inicial = heuristica(problema.estado_inicial)
    frontera.agregar(Nodo(problema.estado_inicial, h=h_inicial))
    mejor_g = {problema.estado_inicial: 0}
    expandidos = 0
    generados = 1
    max_en_frontera = 1

    while not frontera.esta_vacia():
        nodo = frontera.sacar()

        # Una mejora posterior deja en el heap la entrada anterior. Solo la
        # entrada que coincide con el mejor costo conocido puede procesarse.
        if nodo.g != mejor_g.get(nodo.estado):
            continue

        if nodo.estado.es_objetivo(problema.mapa):
            return _resultado_busqueda(
                nodo, EXITO, expandidos, frontera, max_en_frontera, generados
            )

        motivo = limites.excedido(expandidos)
        if motivo is not None:
            return _resultado_busqueda(
                None, motivo, expandidos, frontera, max_en_frontera, generados
            )

        hijos = expandir(nodo, problema, heuristica, podar_deadlocks)
        expandidos += 1
        for hijo in hijos:
            costo_anterior = mejor_g.get(hijo.estado)
            if costo_anterior is not None and hijo.g >= costo_anterior:
                continue
            mejor_g[hijo.estado] = hijo.g
            frontera.agregar(hijo)
            generados += 1
        max_en_frontera = max(max_en_frontera, len(frontera))

    return _resultado_busqueda(
        None,
        SIN_SOLUCION,
        expandidos,
        frontera,
        max_en_frontera,
        generados,
    )


def buscar(
    estado_inicial: Estado,
    mapa: Mapa,
    crear_frontera: Callable[[], Frontera],
    heuristica: Optional[Heuristica] = None,
    podar_deadlocks: bool = True,
    limites: Optional[_Limites] = None,
    problema: Optional[ProblemaSokoban] = None,
) -> Tuple[Optional[Nodo], str, int, int, int]:
    """Compatibilidad con la API anterior basada en fabricas de frontera."""
    limites = limites or _Limites(None, None)
    problema = problema or ProblemaSokoban(mapa, estado_inicial)
    _validar_problema(problema, mapa, estado_inicial)

    if crear_frontera is FronteraFIFO:
        return _buscar_bfs(
            problema, podar_deadlocks, limites
        ).como_tupla_compatible()
    if crear_frontera is FronteraLIFO:
        return _buscar_dfs(
            problema, podar_deadlocks, limites
        ).como_tupla_compatible()

    funcion_h = heuristica or (lambda estado: 0)
    if crear_frontera is frontera_greedy:
        return _buscar_greedy(
            problema, funcion_h, podar_deadlocks, limites
        ).como_tupla_compatible()
    if crear_frontera is frontera_a_estrella:
        return _buscar_a_estrella(
            problema, funcion_h, podar_deadlocks, limites
        ).como_tupla_compatible()

    # Soporte transitorio para fabricas de frontera externas. Sigue las
    # mismas reglas de metricas y orden que los algoritmos incorporados.
    h_inicial = funcion_h(estado_inicial)
    frontera = crear_frontera()
    frontera.agregar(Nodo(estado_inicial, padre=None, accion=None, g=0, h=h_inicial))

    visitados = {estado_inicial}
    expandidos = 0
    generados = 1

    while not frontera.esta_vacia():
        nodo = frontera.sacar()

        if nodo.estado.es_objetivo(mapa):
            return nodo, EXITO, expandidos, len(frontera), generados

        motivo = limites.excedido(expandidos)
        if motivo is not None:
            return None, motivo, expandidos, len(frontera), generados

        hijos = expandir(nodo, problema, heuristica, podar_deadlocks)
        expandidos += 1
        nuevos = []
        for hijo in hijos:
            if hijo.estado not in visitados:
                visitados.add(hijo.estado)
                nuevos.append(hijo)

        if isinstance(frontera, FronteraLIFO):
            nuevos.reverse()
        for hijo in nuevos:
            frontera.agregar(hijo)
            generados += 1

    return None, SIN_SOLUCION, expandidos, 0, generados


def buscar_iddfs(
    estado_inicial: Estado,
    mapa: Mapa,
    podar_deadlocks: bool = True,
    limites: Optional[_Limites] = None,
    profundidad_maxima: Optional[int] = None,
    problema: Optional[ProblemaSokoban] = None,
) -> Tuple[Optional[Nodo], str, int, int, int]:
    """DFS con limite de profundidad creciente: memoria de DFS, optimo como BFS."""
    limites = limites or _Limites(None, None)
    problema = problema or ProblemaSokoban(mapa, estado_inicial)
    _validar_problema(problema, mapa, estado_inicial)
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

            if nodo.estado.es_objetivo(mapa):
                return nodo, EXITO, expandidos_total, len(pila), generados_total

            if nodo.g >= limite:
                hubo_corte = True
                continue

            motivo = limites.excedido(expandidos_total)
            if motivo is not None:
                return None, motivo, expandidos_total, len(pila), generados_total

            hijos = expandir(nodo, problema, None, podar_deadlocks)
            expandidos_total += 1
            nuevos = []
            for hijo in hijos:
                previa = mejor_profundidad.get(hijo.estado)
                if previa is None or hijo.g < previa:
                    mejor_profundidad[hijo.estado] = hijo.g
                    nuevos.append(hijo)
            for hijo in reversed(nuevos):
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
    problema: Optional[ProblemaSokoban] = None,
) -> Resultado:
    """Corre un algoritmo y mide todo lo que pide el enunciado."""
    if clave_algoritmo not in ALGORITMOS:
        raise KeyError(
            "algoritmo desconocido: {!r}. Opciones: {}".format(
                clave_algoritmo, ", ".join(ORDEN_ALGORITMOS)
            )
        )

    descripcion = ALGORITMOS[clave_algoritmo]
    problema = problema or ProblemaSokoban(mapa, estado_inicial)
    _validar_problema(problema, mapa, estado_inicial)
    funcion_h = None
    nombre_h = None
    if descripcion.usa_heuristica:
        elegida = obtener_heuristica(heuristica)
        funcion_h = crear_heuristica(elegida.nombre, problema)
        nombre_h = elegida.nombre

    inicio = time.perf_counter()
    limites = _Limites(max_nodos, timeout)

    if descripcion.clave == "bfs":
        resultado_busqueda = _buscar_bfs(
            problema, podar_deadlocks, limites
        )
    elif descripcion.clave == "dfs":
        resultado_busqueda = _buscar_dfs(
            problema, podar_deadlocks, limites
        )
    elif descripcion.clave == "greedy":
        if funcion_h is None:
            raise AssertionError("Greedy requiere una heuristica")
        resultado_busqueda = _buscar_greedy(
            problema, funcion_h, podar_deadlocks, limites
        )
    elif descripcion.clave == "astar":
        if funcion_h is None:
            raise AssertionError("A* requiere una heuristica")
        resultado_busqueda = _buscar_a_estrella(
            problema, funcion_h, podar_deadlocks, limites
        )
    elif descripcion.clave == "iddfs":
        nodo, motivo, expandidos, en_frontera, generados = buscar_iddfs(
            estado_inicial,
            mapa,
            podar_deadlocks,
            limites,
            problema=problema,
        )
        resultado_busqueda = _ResultadoBusqueda(
            nodo=nodo,
            motivo=motivo,
            expandidos=expandidos,
            en_frontera=en_frontera,
            max_en_frontera=max(1, en_frontera),
            generados=generados,
        )
    else:
        raise AssertionError("algoritmo registrado sin implementacion")

    nodo = resultado_busqueda.nodo
    motivo = resultado_busqueda.motivo
    expandidos = resultado_busqueda.expandidos
    en_frontera = resultado_busqueda.en_frontera
    generados = resultado_busqueda.generados

    tiempo = time.perf_counter() - inicio
    exito = nodo is not None

    if nodo is not None:
        estado_resultado: EstadoBusqueda = "exito"
    elif motivo in (LIMITE_NODOS, TIMEOUT):
        estado_resultado = "corte"
    else:
        estado_resultado = "fracaso"

    return Resultado(
        estado=estado_resultado,
        algoritmo=descripcion.nombre,
        heuristica=nombre_h,
        costo=nodo.g if exito else None,
        movimientos=tuple(reconstruir_camino(nodo)) if exito else (),
        nodos_expandidos=expandidos,
        nodos_frontera=en_frontera,
        max_nodos_frontera=resultado_busqueda.max_en_frontera,
        tiempo_segundos=tiempo,
        motivo=motivo,
        nodos_generados=generados,
        _estados=(
            tuple(reconstruir_estados(nodo))
            if exito and guardar_estados
            else ()
        ),
    )


def resolver(
    problema: ProblemaSokoban,
    algoritmo: str,
    heuristica: Optional[str] = None,
    max_expandidos: Optional[int] = None,
) -> Resultado:
    """Punto de entrada publico para resolver un problema de Sokoban."""
    return ejecutar_busqueda(
        algoritmo,
        problema.estado_inicial,
        problema.mapa,
        heuristica=heuristica or HEURISTICA_POR_DEFECTO,
        max_nodos=max_expandidos,
        problema=problema,
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
