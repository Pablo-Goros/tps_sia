from pathlib import Path

import pytest

import sokoban.busqueda as modulo_busqueda
from sokoban import ProblemaSokoban, cargar_nivel, parsear_tablero, resolver
from sokoban.busqueda import (
    EXITO,
    LIMITE_NODOS,
    buscar,
    ejecutar_busqueda,
)
from sokoban.estado import Estado, Mapa
from sokoban.frontera import FronteraFIFO, frontera_a_estrella, frontera_greedy
from sokoban.nodo import Nodo


NIVELES = Path(__file__).resolve().parents[1] / "niveles"
ALGORITMOS_REQUERIDOS = ("bfs", "dfs", "greedy", "astar")

UN_MOVIMIENTO = "#####\n#@$.#\n#####"
DOS_MOVIMIENTOS = "######\n#@ $.#\n######"
RESUELTO = "#####\n#@* #\n#####"
SIN_SOLUCION = "#####\n#$@.#\n#####"


@pytest.mark.parametrize("algoritmo", ALGORITMOS_REQUERIDOS)
def test_un_objetivo_extraido_no_se_cuenta_como_expandido(algoritmo: str) -> None:
    mapa, estado = parsear_tablero(UN_MOVIMIENTO)

    resultado = ejecutar_busqueda(algoritmo, estado, mapa)

    assert resultado.estado == "exito"
    assert resultado.costo == 1
    assert resultado.nodos_expandidos == 1
    assert resultado.nodos_frontera == 0
    assert resultado.max_nodos_frontera == 1


@pytest.mark.parametrize("algoritmo", ALGORITMOS_REQUERIDOS)
def test_el_limite_de_expansiones_es_exacto(algoritmo: str) -> None:
    mapa, estado = parsear_tablero(DOS_MOVIMIENTOS)

    resultado = ejecutar_busqueda(algoritmo, estado, mapa, max_nodos=1)

    assert resultado.estado == "corte"
    assert resultado.motivo == LIMITE_NODOS
    assert resultado.nodos_expandidos == 1


@pytest.mark.parametrize("algoritmo", ALGORITMOS_REQUERIDOS)
def test_un_estado_inicial_resuelto_precede_un_limite_cero(algoritmo: str) -> None:
    mapa, estado = parsear_tablero(RESUELTO)

    resultado = ejecutar_busqueda(algoritmo, estado, mapa, max_nodos=0)

    assert resultado.estado == "exito"
    assert resultado.costo == 0
    assert resultado.nodos_expandidos == 0
    assert resultado.max_nodos_frontera == 1


def test_metricas_de_frontera_cuentan_entradas_y_el_maximo_historico() -> None:
    mapa, estado = parsear_tablero(
        """\
#######
#  .  #
#     #
#  @  #
#  $  #
#     #
#######
"""
    )

    resultado = ejecutar_busqueda(
        "bfs",
        estado,
        mapa,
        podar_deadlocks=False,
        max_nodos=1,
    )

    assert resultado.estado == "corte"
    assert resultado.nodos_expandidos == 1
    assert resultado.nodos_frontera == 3
    assert resultado.max_nodos_frontera == 4


@pytest.mark.parametrize("algoritmo", ALGORITMOS_REQUERIDOS)
def test_un_problema_agotado_termina_en_fracaso(algoritmo: str) -> None:
    mapa, estado = parsear_tablero(SIN_SOLUCION)

    resultado = ejecutar_busqueda(algoritmo, estado, mapa)

    assert resultado.estado == "fracaso"
    assert resultado.costo is None
    assert resultado.movimientos == ()
    assert resultado.nodos_frontera == 0


@pytest.mark.parametrize(
    ("archivo", "costo_optimo"),
    [
        ("nivel_01_trivial.txt", 6),
        ("nivel_02_facil.txt", 13),
        ("nivel_03_medio.txt", 18),
        ("nivel_04_dificil.txt", 24),
    ],
)
def test_bfs_y_a_estrella_coinciden_en_costos_optimos(
    archivo: str, costo_optimo: int
) -> None:
    mapa, estado = cargar_nivel(NIVELES / archivo)

    bfs = ejecutar_busqueda("bfs", estado, mapa)
    a_estrella = ejecutar_busqueda("astar", estado, mapa)

    assert bfs.costo == costo_optimo
    assert a_estrella.costo == bfs.costo


@pytest.mark.parametrize("algoritmo", ALGORITMOS_REQUERIDOS)
def test_todo_camino_devuelto_se_puede_reproducir(algoritmo: str) -> None:
    mapa, estado = cargar_nivel(NIVELES / "nivel_01_trivial.txt")
    problema = ProblemaSokoban(mapa, estado)

    resultado = resolver(problema, algoritmo)

    actual = estado
    for accion in resultado.movimientos:
        siguiente = problema.aplicar_accion(actual, accion)
        assert siguiente is not None
        actual = siguiente
    assert actual.es_objetivo(mapa)
    assert len(resultado.movimientos) == resultado.costo


def test_bfs_marca_visitados_al_encolar(monkeypatch) -> None:
    raiz, izquierda, derecha, objetivo, mapa = _estados_de_grafo()

    def expandir_controlado(nodo, problema, heuristica, podar_deadlocks):
        if nodo.estado == raiz:
            return [
                Nodo(izquierda, nodo, "arriba", 1),
                Nodo(derecha, nodo, "abajo", 1),
            ]
        if nodo.estado in (izquierda, derecha):
            return [Nodo(objetivo, nodo, "derecha", nodo.g + 1)]
        return []

    monkeypatch.setattr(modulo_busqueda, "expandir", expandir_controlado)

    nodo, motivo, expandidos, _, generados = buscar(
        raiz, mapa, FronteraFIFO
    )

    assert nodo is not None
    assert motivo == EXITO
    assert expandidos == 3
    assert generados == 4


def test_greedy_desempata_por_orden_de_insercion(monkeypatch) -> None:
    raiz, _, alternativa, objetivo, mapa = _estados_de_grafo()
    expandidos = []

    def expandir_controlado(nodo, problema, heuristica, podar_deadlocks):
        expandidos.append(nodo.estado)
        if nodo.estado == raiz:
            return [
                Nodo(objetivo, nodo, "arriba", 1, 0),
                Nodo(alternativa, nodo, "derecha", 1, 0),
            ]
        return []

    monkeypatch.setattr(modulo_busqueda, "expandir", expandir_controlado)

    nodo, motivo, _, _, _ = buscar(
        raiz,
        mapa,
        frontera_greedy,
        heuristica=lambda estado: 0,
    )

    assert nodo is not None
    assert motivo == EXITO
    assert nodo.accion == "arriba"
    assert expandidos == [raiz]


def test_a_estrella_reabre_estados_mejorados_y_descarta_entradas_obsoletas(
    monkeypatch,
) -> None:
    raiz, izquierda, derecha, objetivo, mapa = _estados_de_grafo()
    intermedio = Estado(jugador=(4, 0), cajas=frozenset({(5, 5)}))
    veces_expandido = []
    valores_h = {
        raiz: 0,
        izquierda: 0,
        derecha: 5,
        intermedio: 0,
        objetivo: 0,
    }

    def expandir_controlado(nodo, problema, heuristica, podar_deadlocks):
        veces_expandido.append(nodo.estado)
        if nodo.estado == raiz:
            return [
                Nodo(izquierda, nodo, "arriba", 5, valores_h[izquierda]),
                Nodo(derecha, nodo, "abajo", 1, valores_h[derecha]),
            ]
        if nodo.estado == izquierda:
            return [
                Nodo(
                    intermedio,
                    nodo,
                    "derecha",
                    nodo.g + 1,
                    valores_h[intermedio],
                )
            ]
        if nodo.estado == derecha:
            return [
                Nodo(
                    izquierda,
                    nodo,
                    "izquierda",
                    nodo.g + 1,
                    valores_h[izquierda],
                )
            ]
        if nodo.estado == intermedio:
            return [
                Nodo(
                    objetivo,
                    nodo,
                    "derecha",
                    nodo.g + 1,
                    valores_h[objetivo],
                )
            ]
        return []

    monkeypatch.setattr(modulo_busqueda, "expandir", expandir_controlado)

    nodo, motivo, _, en_frontera, _ = buscar(
        raiz,
        mapa,
        frontera_a_estrella,
        heuristica=valores_h.__getitem__,
    )

    assert nodo is not None
    assert motivo == EXITO
    assert nodo.g == 4
    assert veces_expandido.count(izquierda) == 2
    assert en_frontera == 1


def _estados_de_grafo() -> tuple[Estado, Estado, Estado, Estado, Mapa]:
    raiz = Estado(jugador=(0, 0), cajas=frozenset({(9, 9)}))
    izquierda = Estado(jugador=(1, 0), cajas=frozenset({(8, 8)}))
    derecha = Estado(jugador=(2, 0), cajas=frozenset({(7, 7)}))
    objetivo = Estado(jugador=(3, 0), cajas=frozenset({(1, 1)}))
    mapa = Mapa(
        ancho=10,
        alto=10,
        pisos=frozenset((x, y) for y in range(10) for x in range(10)),
        paredes=frozenset(),
        objetivos=frozenset({(1, 1)}),
    )
    return raiz, izquierda, derecha, objetivo, mapa
