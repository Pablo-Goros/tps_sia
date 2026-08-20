import os
from pathlib import Path
import subprocess
import sys

import pytest

import sokoban.busqueda as modulo_busqueda
from sokoban.busqueda import LIMITE_NODOS, buscar, ejecutar_busqueda
from sokoban.estado import Estado, Mapa, aplicar_accion
from sokoban.frontera import FronteraLIFO
from sokoban.nivel import parsear_tablero
from sokoban.nodo import Nodo


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ONE_MOVE_BOARD = """\
#####
#@$.#
#####
"""


def test_walking_beyond_board_boundaries_is_illegal() -> None:
    mapa = Mapa(
        ancho=2,
        alto=1,
        pisos=frozenset({(0, 0), (1, 0)}),
        paredes=frozenset(),
        objetivos=frozenset({(0, 0)}),
    )
    estado = Estado(jugador=(1, 0), cajas=frozenset({(0, 0)}))

    assert aplicar_accion(estado, mapa, "derecha") is None


def test_missing_cells_in_uneven_rows_are_not_floor() -> None:
    mapa, estado = parsear_tablero(
        """\
#######
# @
# $.###
#######
"""
    )

    assert aplicar_accion(estado, mapa, "derecha") is None


def test_reverse_push_preprocessing_cannot_escape_the_board() -> None:
    script = """
from sokoban.distancias import tabla_empujes
from sokoban.estado import Mapa

mapa = Mapa(
    ancho=1,
    alto=1,
    pisos=frozenset({(0, 0)}),
    paredes=frozenset(),
    objetivos=frozenset({(0, 0)}),
)
tabla_empujes(mapa)
"""
    env = os.environ.copy()
    existing_path = env.get("PYTHONPATH")
    source_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = (
        source_path + os.pathsep + existing_path if existing_path else source_path
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1.0,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.xfail(
    strict=True,
    reason="expansion limits are currently checked only every 2048 expansions",
)
def test_expansion_cutoff_is_exact() -> None:
    mapa, estado = parsear_tablero(ONE_MOVE_BOARD)

    resultado = ejecutar_busqueda("bfs", estado, mapa, max_nodos=1)

    assert not resultado.exito
    assert resultado.motivo == LIMITE_NODOS
    assert resultado.nodos_expandidos == 1


@pytest.mark.xfail(
    strict=True,
    reason="a goal node is currently counted as expanded before its goal check",
)
def test_goal_nodes_are_not_counted_as_expanded() -> None:
    mapa, estado = parsear_tablero(ONE_MOVE_BOARD)

    resultado = ejecutar_busqueda("bfs", estado, mapa)

    assert resultado.exito
    assert resultado.nodos_expandidos == 1


@pytest.mark.xfail(
    strict=True,
    reason="DFS currently pushes U,D,L,R directly and therefore explores R first",
)
def test_dfs_explores_successors_in_configured_order(monkeypatch) -> None:
    raiz = Estado(jugador=(0, 0), cajas=frozenset({(9, 9)}))
    arriba_objetivo = Estado(jugador=(0, -1), cajas=frozenset({(1, 1)}))
    derecha_sin_salida = Estado(jugador=(1, 0), cajas=frozenset({(8, 8)}))
    mapa = Mapa(
        ancho=10,
        alto=10,
        pisos=frozenset(
            (x, y)
            for y in range(10)
            for x in range(10)
        ),
        paredes=frozenset(),
        objetivos=frozenset({(1, 1)}),
    )
    estados_expandidos = []

    def expandir_controlado(nodo, problema, heuristica, podar_deadlocks):
        estados_expandidos.append(nodo.estado)
        if nodo.estado == raiz:
            return [
                Nodo(
                    arriba_objetivo,
                    padre=nodo,
                    accion="arriba",
                    g=nodo.g + 1,
                ),
                Nodo(
                    derecha_sin_salida,
                    padre=nodo,
                    accion="derecha",
                    g=nodo.g + 1,
                ),
            ]
        return []

    monkeypatch.setattr(modulo_busqueda, "expandir", expandir_controlado)

    nodo, motivo, _, _, _ = buscar(raiz, mapa, FronteraLIFO)

    assert nodo is not None
    assert motivo == modulo_busqueda.EXITO
    assert nodo.accion == "arriba"
    assert estados_expandidos == [raiz]


@pytest.mark.xfail(
    strict=True,
    reason="search results currently report only the final frontier size",
)
def test_maximum_frontier_size_is_tracked_from_initialization() -> None:
    mapa, estado = parsear_tablero(ONE_MOVE_BOARD)

    resultado = ejecutar_busqueda("bfs", estado, mapa)

    assert resultado.max_nodos_frontera == 1
