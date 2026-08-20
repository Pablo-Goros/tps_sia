import math
from pathlib import Path

import pytest

from sokoban import ProblemaSokoban, cargar_nivel, parsear_tablero
from sokoban.estado import Estado, Mapa
from sokoban.heuristicas import (
    HEURISTICAS,
    costo_asignacion_minima,
    crear_heuristica,
)


NIVELES = Path(__file__).resolve().parents[1] / "niveles"


def test_asignacion_minima_usa_programacion_dinamica_uno_a_uno() -> None:
    matriz = (
        (4, 1, 3),
        (2, 0, 5),
        (3, 2, 2),
    )

    assert costo_asignacion_minima(matriz) == 5
    assert costo_asignacion_minima(()) == 0


def test_asignacion_incompleta_devuelve_infinito() -> None:
    matriz = (
        (0, math.inf),
        (math.inf, math.inf),
    )

    assert math.isinf(costo_asignacion_minima(matriz))


def test_asignacion_rechaza_una_matriz_no_cuadrada() -> None:
    with pytest.raises(ValueError, match="debe ser cuadrada"):
        costo_asignacion_minima(((1, 2),))


def test_manhattan_realiza_matching_uno_a_uno() -> None:
    mapa = Mapa(
        ancho=11,
        alto=2,
        pisos=frozenset((x, y) for y in range(2) for x in range(11)),
        paredes=frozenset(),
        objetivos=frozenset({(0, 0), (10, 0)}),
    )
    estado = Estado(jugador=(5, 1), cajas=frozenset({(1, 0), (2, 0)}))
    problema = ProblemaSokoban(mapa, estado)

    heuristica = crear_heuristica("manhattan", problema)

    assert heuristica(estado) == 9


@pytest.mark.parametrize("nombre", ["manhattan", "empujes_inversos"])
def test_heuristicas_valen_cero_en_un_estado_resuelto(nombre: str) -> None:
    mapa, estado = parsear_tablero("######\n#@** #\n######")
    problema = ProblemaSokoban(mapa, estado)

    assert crear_heuristica(nombre, problema)(estado) == 0


@pytest.mark.parametrize(
    ("archivo", "costo_optimo"),
    [
        ("nivel_01_trivial.txt", 6),
        ("nivel_02_facil.txt", 13),
        ("nivel_03_medio.txt", 18),
        ("nivel_04_dificil.txt", 24),
    ],
)
@pytest.mark.parametrize("nombre", ["manhattan", "empujes_inversos"])
def test_heuristicas_no_superan_costos_optimos_conocidos(
    archivo: str, costo_optimo: int, nombre: str
) -> None:
    mapa, estado = cargar_nivel(NIVELES / archivo)
    problema = ProblemaSokoban(mapa, estado)

    valor = crear_heuristica(nombre, problema)(estado)

    assert 0 <= valor <= costo_optimo


def test_empujes_inversos_refleja_rodeos_impuestos_por_paredes() -> None:
    mapa, estado = parsear_tablero(
        """\
#########
#       #
# . # $ #
#   #   #
#   #   #
#   @   #
#       #
#########
"""
    )
    problema = ProblemaSokoban(mapa, estado)

    manhattan = crear_heuristica("manhattan", problema)(estado)
    empujes = crear_heuristica("empujes_inversos", problema)(estado)

    assert manhattan == 4
    assert empujes == 10


def test_empujes_inversos_devuelve_infinito_si_no_hay_matching_completo() -> None:
    mapa, estado = parsear_tablero(
        """\
#####
# . #
#   #
#$@ #
#####
"""
    )
    problema = ProblemaSokoban(mapa, estado)

    valor = crear_heuristica("empujes_inversos", problema)(estado)

    assert math.isinf(valor)


def test_cada_fabrica_crea_una_cache_independiente() -> None:
    mapa, estado = parsear_tablero("#####\n#@$.#\n#####")
    problema = ProblemaSokoban(mapa, estado)
    primera = crear_heuristica("manhattan", problema)
    segunda = crear_heuristica("manhattan", problema)

    assert primera is not segunda
    assert primera.estadisticas_cache().currsize == 0
    assert segunda.estadisticas_cache().currsize == 0

    assert primera(estado) == 1
    assert primera(estado) == 1

    assert primera.estadisticas_cache().misses == 1
    assert primera.estadisticas_cache().hits == 1
    assert segunda.estadisticas_cache().currsize == 0


def test_interfaz_requerida_no_expone_heuristicas_obsoletas() -> None:
    assert set(HEURISTICAS) == {"manhattan", "empujes_inversos"}

    with pytest.raises(KeyError):
        crear_heuristica(
            "matching",
            ProblemaSokoban(*parsear_tablero("#####\n#@$.#\n#####")),
        )
