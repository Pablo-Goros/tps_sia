from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from sokoban.busqueda import ejecutar_busqueda
from sokoban.estado import Estado, aplicar_accion, cargar_nivel, parsear_tablero
from sokoban.nodo import Nodo, reconstruir_camino, reconstruir_estados


LEVELS = Path(__file__).resolve().parents[1] / "niveles"

ONE_MOVE_BOARD = """\
#####
#@$.#
#####
"""


def test_parser_supports_compound_symbols() -> None:
    mapa, estado = parsear_tablero(
        """\
######
#+*$ #
######
"""
    )

    assert estado.jugador == (1, 1)
    assert estado.cajas == frozenset({(2, 1), (3, 1)})
    assert mapa.objetivos == frozenset({(1, 1), (2, 1)})


def test_states_are_immutable_hashable_values() -> None:
    estado = Estado(jugador=(1, 2), cajas=frozenset({(3, 4)}))
    estado_equivalente = Estado(jugador=(1, 2), cajas=frozenset({(3, 4)}))

    assert estado == estado_equivalente
    assert hash(estado) == hash(estado_equivalente)
    assert {estado, estado_equivalente} == {estado}

    with pytest.raises(FrozenInstanceError):
        estado.jugador = (2, 2)  # type: ignore[misc]
    with pytest.raises(AttributeError):
        estado.cajas.add((4, 4))  # type: ignore[attr-defined]


def test_legal_walking_and_pushing_create_new_states() -> None:
    mapa, estado = parsear_tablero(
        """\
#######
# @$. #
#######
"""
    )

    caminando = aplicar_accion(estado, mapa, "izquierda")
    empujando = aplicar_accion(estado, mapa, "derecha")

    assert caminando == Estado(jugador=(1, 1), cajas=frozenset({(3, 1)}))
    assert empujando == Estado(jugador=(3, 1), cajas=frozenset({(4, 1)}))
    assert estado == Estado(jugador=(2, 1), cajas=frozenset({(3, 1)}))


def test_path_and_state_reconstruction_follow_parent_links() -> None:
    inicial = Estado(jugador=(1, 1), cajas=frozenset({(2, 1)}))
    intermedio = Estado(jugador=(1, 2), cajas=frozenset({(2, 1)}))
    final = Estado(jugador=(2, 2), cajas=frozenset({(2, 1)}))
    raiz = Nodo(inicial)
    hijo = Nodo(intermedio, padre=raiz, accion="abajo", g=1)
    objetivo = Nodo(final, padre=hijo, accion="derecha", g=2)

    assert reconstruir_camino(objetivo) == ["abajo", "derecha"]
    assert reconstruir_estados(objetivo) == [inicial, intermedio, final]


@pytest.mark.parametrize("algoritmo", ["bfs", "dfs", "greedy", "astar"])
def test_required_algorithms_solve_one_move_board(algoritmo: str) -> None:
    mapa, estado = parsear_tablero(ONE_MOVE_BOARD)

    resultado = ejecutar_busqueda(algoritmo, estado, mapa)

    assert resultado.exito
    assert resultado.costo == 1
    assert resultado.camino == ["derecha"]


@pytest.mark.parametrize(
    ("archivo", "costo_optimo"),
    [
        ("nivel_01_trivial.txt", 6),
        ("nivel_02_facil.txt", 13),
        ("nivel_03_medio.txt", 18),
        ("nivel_04_dificil.txt", 24),
    ],
)
def test_existing_small_levels_keep_known_bfs_costs(
    archivo: str, costo_optimo: int
) -> None:
    mapa, estado = cargar_nivel(str(LEVELS / archivo))

    resultado = ejecutar_busqueda("bfs", estado, mapa)

    assert resultado.exito
    assert resultado.costo == costo_optimo
