"""Definicion publica del problema de Sokoban."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import FrozenSet, List, Mapping, Optional, Tuple

from .distancias import celdas_muertas as calcular_celdas_muertas
from .distancias import tabla_empujes
from .estado import DIRECCIONES, Direccion, Estado, Mapa, ORDEN_ACCIONES, Posicion

Sucesor = Tuple[Estado, Direccion, int]
DistanciasEmpujes = Mapping[Posicion, Mapping[Posicion, int]]
COSTO_MOVIMIENTO = 1


@dataclass(frozen=True)
class ProblemaSokoban:
    """Mapa, estado inicial y orden determinista de acciones de un problema."""

    mapa: Mapa
    estado_inicial: Estado
    orden_acciones: Tuple[Direccion, ...] = ORDEN_ACCIONES
    distancias_empujes: DistanciasEmpujes = field(
        init=False, compare=False, hash=False, repr=False
    )
    celdas_muertas: FrozenSet[Posicion] = field(
        init=False, compare=False, hash=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.estado_inicial.jugador not in self.mapa.pisos:
            raise ValueError("el jugador debe estar sobre un piso valido")
        if not self.estado_inicial.cajas <= self.mapa.pisos:
            raise ValueError("todas las cajas deben estar sobre pisos validos")
        if not self.estado_inicial.cajas:
            raise ValueError("el problema debe tener al menos una caja")
        if len(self.estado_inicial.cajas) != len(self.mapa.objetivos):
            raise ValueError("la cantidad de cajas y objetivos debe coincidir")
        if set(self.orden_acciones) != set(ORDEN_ACCIONES):
            raise ValueError(
                "orden_acciones debe contener arriba, abajo, izquierda y derecha"
            )
        if len(self.orden_acciones) != len(ORDEN_ACCIONES):
            raise ValueError("orden_acciones no puede contener acciones repetidas")

        distancias = {
            objetivo: MappingProxyType(dict(por_posicion))
            for objetivo, por_posicion in tabla_empujes(self.mapa).items()
        }
        object.__setattr__(
            self, "distancias_empujes", MappingProxyType(distancias)
        )
        object.__setattr__(
            self, "celdas_muertas", calcular_celdas_muertas(self.mapa)
        )

    def aplicar_accion(
        self, estado: Estado, accion: Direccion
    ) -> Optional[Estado]:
        """Devuelve un estado nuevo o ``None`` si el movimiento es ilegal."""
        return self.aplicar_en_mapa(estado, self.mapa, accion)

    @staticmethod
    def aplicar_en_mapa(
        estado: Estado, mapa: Mapa, accion: Direccion
    ) -> Optional[Estado]:
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
            cajas = (estado.cajas - {destino}) | {siguiente}
            return Estado(jugador=destino, cajas=frozenset(cajas))

        return Estado(jugador=destino, cajas=estado.cajas)

    @staticmethod
    def accion_empuja(estado: Estado, accion: Direccion) -> bool:
        desplazamiento_x, desplazamiento_y = DIRECCIONES[accion]
        return (
            estado.jugador[0] + desplazamiento_x,
            estado.jugador[1] + desplazamiento_y,
        ) in estado.cajas

    def generar_sucesores(
        self, estado: Estado, podar_deadlocks: bool = True
    ) -> List[Sucesor]:
        """Genera sucesores legales en el orden configurado, todos con costo 1."""
        sucesores: List[Sucesor] = []
        for accion in self.orden_acciones:
            hubo_empuje = self.accion_empuja(estado, accion)
            nuevo = self.aplicar_accion(estado, accion)
            if nuevo is None:
                continue
            if podar_deadlocks and hubo_empuje and self.es_deadlock(nuevo, estado):
                continue
            sucesores.append((nuevo, accion, COSTO_MOVIMIENTO))
        return sucesores

    def es_deadlock(self, nuevo: Estado, anterior: Estado) -> bool:
        """Indica si el ultimo empuje dejo una caja inmovilizable."""
        movida = self._caja_movida(nuevo, anterior)
        if movida is None:
            return False
        if movida in self.celdas_muertas:
            return True
        return self._bloque_congelado(nuevo, movida)

    @staticmethod
    def _caja_movida(nuevo: Estado, anterior: Estado) -> Optional[Posicion]:
        diferencia = nuevo.cajas - anterior.cajas
        return next(iter(diferencia)) if diferencia else None

    def _bloque_congelado(self, estado: Estado, caja: Posicion) -> bool:
        """Detecta un bloque 2x2 inmovil de paredes y cajas fuera de objetivo."""
        x, y = caja
        for x_inicial in (x - 1, x):
            for y_inicial in (y - 1, y):
                celdas = (
                    (x_inicial, y_inicial),
                    (x_inicial + 1, y_inicial),
                    (x_inicial, y_inicial + 1),
                    (x_inicial + 1, y_inicial + 1),
                )
                if all(
                    posicion in self.mapa.paredes or posicion in estado.cajas
                    for posicion in celdas
                ):
                    cajas_del_bloque = (
                        posicion for posicion in celdas if posicion in estado.cajas
                    )
                    if any(
                        posicion not in self.mapa.objetivos
                        for posicion in cajas_del_bloque
                    ):
                        return True
        return False


__all__ = [
    "COSTO_MOVIMIENTO",
    "DistanciasEmpujes",
    "ProblemaSokoban",
    "Sucesor",
]
