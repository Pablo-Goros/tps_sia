import pytest

from sokoban.estado import Estado, Mapa, aplicar_accion
from sokoban.nivel import NivelInvalido, parsear_tablero, render_texto


def test_filas_desiguales_no_crean_pisos_implicitamente() -> None:
    mapa, estado = parsear_tablero(
        """\
#######
# @
# $.###
#######
"""
    )

    assert mapa.ancho == 7
    assert mapa.alto == 4
    assert (2, 1) in mapa.pisos
    assert (3, 1) not in mapa.pisos
    assert aplicar_accion(estado, mapa, "derecha") is None


def test_espacios_iniciales_e_internos_son_pisos_explicitos() -> None:
    mapa, _ = parsear_tablero(
        """\
  ####
  #@ #
###$ #
#  . #
######
"""
    )

    assert (0, 0) in mapa.pisos
    assert (1, 0) in mapa.pisos
    assert (3, 1) in mapa.pisos


@pytest.mark.parametrize("caracter", ["-", "_", "x", "á"])
def test_rechaza_simbolos_desconocidos_con_su_ubicacion(caracter: str) -> None:
    texto = "#####\n#@$.#\n# {} #\n#####".format(caracter)

    with pytest.raises(
        NivelInvalido,
        match=r"caracter desconocido .* en fila 2, columna 2",
    ):
        parsear_tablero(texto)


@pytest.mark.parametrize(
    ("texto", "mensaje"),
    [
        ("#####\n# $.#\n#####", "exactamente 1 jugador"),
        ("######\n#@@$.#\n######", "exactamente 1 jugador"),
        ("#####\n#@ .#\n#####", "no tiene cajas"),
        ("#####\n#@ $#\n#####", "no tiene objetivos"),
        ("######\n#@$$.#\n######", "cantidad de cajas"),
    ],
)
def test_valida_jugador_cajas_y_objetivos(texto: str, mensaje: str) -> None:
    with pytest.raises(NivelInvalido, match=mensaje):
        parsear_tablero(texto)


def test_mapa_rechaza_objetivos_fuera_de_los_pisos() -> None:
    with pytest.raises(ValueError, match="objetivos deben estar sobre pisos"):
        Mapa(
            ancho=2,
            alto=1,
            pisos=frozenset({(0, 0)}),
            paredes=frozenset(),
            objetivos=frozenset({(1, 0)}),
        )


def test_render_y_parseo_preservan_mapa_y_estado() -> None:
    mapa, estado = parsear_tablero(
        """\
  #######
  # +*  #
### $$ .#
#######
"""
    )

    texto_renderizado = render_texto(estado, mapa)
    mapa_reparseado, estado_reparseado = parsear_tablero(texto_renderizado)

    assert mapa_reparseado == mapa
    assert estado_reparseado == estado


def test_render_preserva_pisos_que_son_espacios_finales() -> None:
    mapa, estado = parsear_tablero("#####  \n#@$.#\n#####")

    texto_renderizado = render_texto(estado, mapa)
    mapa_reparseado, estado_reparseado = parsear_tablero(texto_renderizado)

    assert texto_renderizado.splitlines()[0] == "#####  "
    assert mapa_reparseado == mapa
    assert estado_reparseado == estado


def test_render_rechaza_entidades_fuera_de_pisos() -> None:
    mapa, estado = parsear_tablero("#####\n#@$.#\n#####")
    estado_invalido = Estado(jugador=(8, 8), cajas=estado.cajas)

    with pytest.raises(ValueError, match="jugador debe estar sobre un piso"):
        render_texto(estado_invalido, mapa)
