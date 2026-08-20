"""Carga, validacion, parseo y renderizado de niveles de Sokoban."""

from __future__ import annotations

from os import PathLike
from typing import Union

from .estado import (
    CAJA,
    CAJA_EN_OBJETIVO,
    JUGADOR,
    JUGADOR_EN_OBJETIVO,
    OBJETIVO,
    PARED,
    PISO,
    Estado,
    Mapa,
)

Ruta = Union[str, PathLike[str]]
SIMBOLOS_VALIDOS = frozenset(
    {PARED, PISO, OBJETIVO, JUGADOR, CAJA, JUGADOR_EN_OBJETIVO, CAJA_EN_OBJETIVO}
)


class NivelInvalido(ValueError):
    pass


def parsear_tablero(texto: str) -> tuple[Mapa, Estado]:
    """Convierte texto XSB en un mapa y su estado inicial."""
    lineas = [
        linea
        for linea in texto.splitlines()
        if not linea.lstrip().startswith(";")
    ]
    # Se omiten solo las lineas vacias usadas para delimitar strings o archivos.
    # Una linea con espacios representa pisos explicitos y debe conservarse.
    while lineas and lineas[0] == "":
        lineas.pop(0)
    while lineas and lineas[-1] == "":
        lineas.pop()
    if not lineas:
        raise NivelInvalido("el nivel esta vacio")

    pisos = set()
    paredes = set()
    objetivos = set()
    cajas = set()
    jugadores = []

    for y, linea in enumerate(lineas):
        for x, caracter in enumerate(linea):
            if caracter not in SIMBOLOS_VALIDOS:
                raise NivelInvalido(
                    "caracter desconocido {!r} en fila {}, columna {}".format(
                        caracter, y, x
                    )
                )

            posicion = (x, y)
            if caracter == PARED:
                paredes.add(posicion)
                continue

            pisos.add(posicion)
            if caracter == OBJETIVO:
                objetivos.add(posicion)
            elif caracter == CAJA:
                cajas.add(posicion)
            elif caracter == CAJA_EN_OBJETIVO:
                cajas.add(posicion)
                objetivos.add(posicion)
            elif caracter == JUGADOR:
                jugadores.append(posicion)
            elif caracter == JUGADOR_EN_OBJETIVO:
                jugadores.append(posicion)
                objetivos.add(posicion)

    if len(jugadores) != 1:
        raise NivelInvalido(
            "el nivel debe tener exactamente 1 jugador (tiene {})".format(
                len(jugadores)
            )
        )
    if not cajas:
        raise NivelInvalido("el nivel no tiene cajas")
    if not objetivos:
        raise NivelInvalido("el nivel no tiene objetivos")
    if len(cajas) != len(objetivos):
        raise NivelInvalido(
            "cantidad de cajas ({}) distinta de cantidad de objetivos ({})".format(
                len(cajas), len(objetivos)
            )
        )

    jugador = jugadores[0]
    if jugador not in pisos or not cajas <= pisos or not objetivos <= pisos:
        raise NivelInvalido("todas las entidades deben estar sobre pisos validos")

    mapa = Mapa(
        ancho=max(len(linea) for linea in lineas),
        alto=len(lineas),
        pisos=frozenset(pisos),
        paredes=frozenset(paredes),
        objetivos=frozenset(objetivos),
    )
    return mapa, Estado(jugador=jugador, cajas=frozenset(cajas))


def cargar_nivel(ruta: Ruta) -> tuple[Mapa, Estado]:
    with open(ruta, "r", encoding="utf-8") as archivo:
        return parsear_tablero(archivo.read())


def render_texto(estado: Estado, mapa: Mapa) -> str:
    """Renderiza un nivel XSB sin convertir las posiciones ausentes en pisos."""
    _validar_entidades(estado, mapa)
    ocupadas = mapa.pisos | mapa.paredes
    filas = []

    for y in range(mapa.alto):
        posiciones_fila = [x for x in range(mapa.ancho) if (x, y) in ocupadas]
        if not posiciones_fila:
            filas.append("")
            continue

        ultimo_x = max(posiciones_fila)
        faltantes = [x for x in range(ultimo_x + 1) if (x, y) not in ocupadas]
        if faltantes:
            raise ValueError(
                "no se puede renderizar una fila con huecos internos: ({}, {})".format(
                    faltantes[0], y
                )
            )

        caracteres = []
        for x in range(ultimo_x + 1):
            posicion = (x, y)
            if posicion in mapa.paredes:
                caracteres.append(PARED)
            elif posicion == estado.jugador:
                caracteres.append(
                    JUGADOR_EN_OBJETIVO
                    if posicion in mapa.objetivos
                    else JUGADOR
                )
            elif posicion in estado.cajas:
                caracteres.append(
                    CAJA_EN_OBJETIVO if posicion in mapa.objetivos else CAJA
                )
            elif posicion in mapa.objetivos:
                caracteres.append(OBJETIVO)
            else:
                caracteres.append(PISO)
        filas.append("".join(caracteres))

    return "\n".join(filas)


def _validar_entidades(estado: Estado, mapa: Mapa) -> None:
    if estado.jugador not in mapa.pisos:
        raise ValueError("el jugador debe estar sobre un piso valido")
    if not estado.cajas <= mapa.pisos:
        raise ValueError("todas las cajas deben estar sobre pisos validos")
    if len(estado.cajas) != len(mapa.objetivos):
        raise ValueError("la cantidad de cajas y objetivos debe coincidir")


__all__ = [
    "NivelInvalido",
    "SIMBOLOS_VALIDOS",
    "cargar_nivel",
    "parsear_tablero",
    "render_texto",
]
