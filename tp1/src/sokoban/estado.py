"""Mapa (parte estatica), Estado (parte dinamica) y reglas del juego."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Literal, Optional, Tuple

Posicion = Tuple[int, int]
Direccion = Literal["arriba", "abajo", "izquierda", "derecha"]

# Coordenadas cartesianas (x, y), con x horizontal e y creciendo hacia abajo.
DIRECCIONES: Dict[Direccion, Posicion] = {
    "arriba": (0, -1),
    "abajo": (0, 1),
    "izquierda": (-1, 0),
    "derecha": (1, 0),
}

# Orden fijo de expansion: hace reproducibles las corridas.
ORDEN_ACCIONES: Tuple[Direccion, ...] = (
    "arriba",
    "abajo",
    "izquierda",
    "derecha",
)

ABREVIATURAS: Dict[Direccion, str] = {
    "arriba": "U",
    "abajo": "D",
    "izquierda": "L",
    "derecha": "R",
}

PARED = "#"
PISO = " "
OBJETIVO = "."
CAJA = "$"
CAJA_EN_OBJETIVO = "*"
JUGADOR = "@"
JUGADOR_EN_OBJETIVO = "+"


class NivelInvalido(ValueError):
    pass


@dataclass(frozen=True)
class Mapa:
    ancho: int
    alto: int
    pisos: FrozenSet[Posicion]
    paredes: FrozenSet[Posicion]
    objetivos: FrozenSet[Posicion]

    def es_pared(self, pos: Posicion) -> bool:
        return pos in self.paredes

    def dentro(self, pos: Posicion) -> bool:
        x, y = pos
        return 0 <= x < self.ancho and 0 <= y < self.alto


@dataclass(frozen=True)
class Estado:
    """Inmutable y hasheable: es lo que entra al set de visitados."""

    jugador: Posicion
    cajas: FrozenSet[Posicion]

    def es_objetivo(self, mapa: Mapa) -> bool:
        return self.cajas == mapa.objetivos

    def cajas_fuera(self, mapa: Mapa) -> FrozenSet[Posicion]:
        return self.cajas - mapa.objetivos


def aplicar_accion(
    estado: Estado, mapa: Mapa, accion: Direccion
) -> Optional[Estado]:
    """Estado resultante de aplicar `accion`, o None si es invalida."""
    desplazamiento_x, desplazamiento_y = DIRECCIONES[accion]
    jugador_x, jugador_y = estado.jugador
    destino = (jugador_x + desplazamiento_x, jugador_y + desplazamiento_y)

    if destino in mapa.paredes:
        return None

    if destino in estado.cajas:
        siguiente = (
            destino[0] + desplazamiento_x,
            destino[1] + desplazamiento_y,
        )
        if siguiente in mapa.paredes or siguiente in estado.cajas:
            return None
        # Se arma un frozenset nuevo: mutar el original corromperia al padre,
        # que sigue vivo en la frontera y en visitados.
        cajas = (estado.cajas - {destino}) | {siguiente}
        return Estado(jugador=destino, cajas=frozenset(cajas))

    return Estado(jugador=destino, cajas=estado.cajas)


def empuja(estado: Estado, accion: Direccion) -> bool:
    desplazamiento_x, desplazamiento_y = DIRECCIONES[accion]
    return (
        estado.jugador[0] + desplazamiento_x,
        estado.jugador[1] + desplazamiento_y,
    ) in estado.cajas


def parsear_tablero(texto: str) -> Tuple[Mapa, Estado]:
    """Convierte el texto de un nivel (formato XSB) en `(Mapa, Estado)`."""
    lineas = [
        linea.rstrip("\n\r")
        for linea in texto.splitlines()
        if not linea.lstrip().startswith(";")
    ]
    while lineas and not lineas[0].strip():
        lineas.pop(0)
    while lineas and not lineas[-1].strip():
        lineas.pop()
    if not lineas:
        raise NivelInvalido("el nivel esta vacio")

    paredes = set()
    objetivos = set()
    cajas = set()
    jugadores = []

    for y, linea in enumerate(lineas):
        for x, caracter in enumerate(linea):
            pos = (x, y)
            if caracter == PARED:
                paredes.add(pos)
            elif caracter == OBJETIVO:
                objetivos.add(pos)
            elif caracter == CAJA:
                cajas.add(pos)
            elif caracter == CAJA_EN_OBJETIVO:
                cajas.add(pos)
                objetivos.add(pos)
            elif caracter == JUGADOR:
                jugadores.append(pos)
            elif caracter == JUGADOR_EN_OBJETIVO:
                jugadores.append(pos)
                objetivos.add(pos)
            elif caracter in (PISO, "-", "_"):
                continue
            else:
                raise NivelInvalido(
                    "caracter desconocido {!r} en fila {}, columna {}".format(
                        caracter, y, x
                    )
                )

    if len(jugadores) != 1:
        raise NivelInvalido(
            "el nivel debe tener exactamente 1 jugador (tiene {})".format(len(jugadores))
        )
    if not cajas:
        raise NivelInvalido("el nivel no tiene cajas")
    if len(cajas) != len(objetivos):
        raise NivelInvalido(
            "cantidad de cajas ({}) distinta de cantidad de objetivos ({})".format(
                len(cajas), len(objetivos)
            )
        )

    ancho = max(len(linea) for linea in lineas)
    alto = len(lineas)
    mapa = Mapa(
        paredes=frozenset(paredes),
        objetivos=frozenset(objetivos),
        ancho=ancho,
        alto=alto,
        pisos=frozenset(
            (x, y)
            for y in range(alto)
            for x in range(ancho)
            if (x, y) not in paredes
        ),
    )
    estado = Estado(jugador=jugadores[0], cajas=frozenset(cajas))
    _verificar_cerrado(mapa, estado)
    return mapa, estado


def _verificar_cerrado(mapa: Mapa, estado: Estado) -> None:
    """Un tablero abierto haria que el espacio de estados sea infinito."""
    pendientes = [estado.jugador]
    vistos = {estado.jugador}
    while pendientes:
        x, y = pendientes.pop()
        for desplazamiento_x, desplazamiento_y in DIRECCIONES.values():
            vecino = (x + desplazamiento_x, y + desplazamiento_y)
            if vecino in mapa.paredes or vecino in vistos:
                continue
            if not mapa.dentro(vecino):
                raise NivelInvalido(
                    "el tablero no esta cerrado: se puede salir por {}".format(vecino)
                )
            vistos.add(vecino)
            pendientes.append(vecino)


def cargar_nivel(ruta: str) -> Tuple[Mapa, Estado]:
    with open(ruta, "r", encoding="utf-8") as archivo:
        return parsear_tablero(archivo.read())
