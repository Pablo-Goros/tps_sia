from dataclasses import FrozenInstanceError
import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest

from sokoban import (
    DIRECCIONES,
    Direccion,
    Estado,
    Mapa,
    ProblemaSokoban,
    Resultado,
    parsear_tablero,
    resolver,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ONE_MOVE_BOARD = """\
#####
#@$.#
#####
"""


def test_public_types_use_the_spanish_interface() -> None:
    mapa, estado = parsear_tablero(ONE_MOVE_BOARD)
    problema = ProblemaSokoban(mapa=mapa, estado_inicial=estado)

    assert isinstance(mapa, Mapa)
    assert isinstance(estado, Estado)
    assert mapa.pisos
    assert problema.orden_acciones == (
        "arriba",
        "abajo",
        "izquierda",
        "derecha",
    )

    with pytest.raises(FrozenInstanceError):
        problema.estado_inicial = estado  # type: ignore[misc]


def test_resolver_exposes_the_planned_signature_and_result() -> None:
    parametros = inspect.signature(resolver).parameters
    assert tuple(parametros) == (
        "problema",
        "algoritmo",
        "heuristica",
        "max_expandidos",
    )

    mapa, estado = parsear_tablero(ONE_MOVE_BOARD)
    resultado = resolver(ProblemaSokoban(mapa, estado), "bfs")

    assert isinstance(resultado, Resultado)
    assert resultado.estado == "exito"
    assert resultado.costo == 1
    assert resultado.movimientos == ("derecha",)
    assert isinstance(resultado.max_nodos_frontera, int)
    assert resultado.tiempo_segundos >= 0

    with pytest.raises(FrozenInstanceError):
        resultado.costo = 2  # type: ignore[misc]


def test_direccion_describes_the_four_public_actions() -> None:
    assert set(Direccion.__args__) == {
        "arriba",
        "abajo",
        "izquierda",
        "derecha",
    }
    assert DIRECCIONES == {
        "arriba": (0, -1),
        "abajo": (0, 1),
        "izquierda": (-1, 0),
        "derecha": (1, 0),
    }

    mapa, estado = parsear_tablero(ONE_MOVE_BOARD)
    assert estado.jugador == (1, 1)
    assert estado.cajas == frozenset({(2, 1)})
    assert mapa.objetivos == frozenset({(3, 1)})


def test_python_module_help_succeeds() -> None:
    env = os.environ.copy()
    existing_path = env.get("PYTHONPATH")
    source_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = (
        source_path + os.pathsep + existing_path if existing_path else source_path
    )

    completed = subprocess.run(
        [sys.executable, "-m", "sokoban", "--help"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "uso:" in completed.stdout
    assert "opciones:" in completed.stdout
    assert "usage:" not in completed.stdout
    assert "resolver" in completed.stdout
    assert "experimentar" in completed.stdout
    assert "graficar" in completed.stdout
