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


@dataclass(frozen=True)
class Mapa:
    ancho: int
    alto: int
    pisos: FrozenSet[Posicion]
    paredes: FrozenSet[Posicion]
    objetivos: FrozenSet[Posicion]

    def __post_init__(self) -> None:
        if self.ancho <= 0 or self.alto <= 0:
            raise ValueError("el ancho y el alto del mapa deben ser positivos")
        if self.pisos & self.paredes:
            raise ValueError("una posicion no puede ser piso y pared a la vez")
        if not self.objetivos <= self.pisos:
            raise ValueError("todos los objetivos deben estar sobre pisos validos")

        posiciones = self.pisos | self.paredes
        fuera = [pos for pos in posiciones if not self.dentro(pos)]
        if fuera:
            raise ValueError(
                "hay posiciones fuera de los limites del mapa: {!r}".format(fuera[0])
            )

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

    if destino not in mapa.pisos:
        return None

    if destino in estado.cajas:
        siguiente = (
            destino[0] + desplazamiento_x,
            destino[1] + desplazamiento_y,
        )
        if siguiente not in mapa.pisos or siguiente in estado.cajas:
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


__all__ = [
    "ABREVIATURAS",
    "DIRECCIONES",
    "Direccion",
    "Estado",
    "Mapa",
    "ORDEN_ACCIONES",
    "Posicion",
    "aplicar_accion",
    "empuja",
]
