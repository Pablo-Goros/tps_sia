"""Fronteras: lo unico que cambia entre BFS, DFS, Greedy y A*."""

from __future__ import annotations

import heapq
import itertools
from collections import deque
from typing import Callable, List, Tuple

from .nodo import Nodo


class Frontera:
    def agregar(self, nodo: Nodo) -> None:
        raise NotImplementedError

    def sacar(self) -> Nodo:
        raise NotImplementedError

    def esta_vacia(self) -> bool:
        return len(self) == 0

    def __len__(self) -> int:
        raise NotImplementedError


class FronteraFIFO(Frontera):
    """BFS."""

    def __init__(self) -> None:
        self._items: deque = deque()

    def agregar(self, nodo: Nodo) -> None:
        self._items.append(nodo)

    def sacar(self) -> Nodo:
        return self._items.popleft()

    def __len__(self) -> int:
        return len(self._items)


class FronteraLIFO(Frontera):
    """DFS."""

    def __init__(self) -> None:
        self._items: List[Nodo] = []

    def agregar(self, nodo: Nodo) -> None:
        self._items.append(nodo)

    def sacar(self) -> Nodo:
        return self._items.pop()

    def __len__(self) -> int:
        return len(self._items)


class FronteraPrioridad(Frontera):
    """Greedy (clave = h) y A* (clave = f)."""

    def __init__(self, clave: Callable[[Nodo], float]) -> None:
        self._items: List[Tuple[float, int, Nodo]] = []
        self._clave = clave
        # Desempate estable, y evita que heapq compare objetos Nodo.
        self._contador = itertools.count()

    def agregar(self, nodo: Nodo) -> None:
        heapq.heappush(self._items, (self._clave(nodo), next(self._contador), nodo))

    def sacar(self) -> Nodo:
        return heapq.heappop(self._items)[2]

    def __len__(self) -> int:
        return len(self._items)


def frontera_greedy() -> FronteraPrioridad:
    return FronteraPrioridad(clave=lambda nodo: nodo.h)


def frontera_a_estrella() -> FronteraPrioridad:
    return FronteraPrioridad(clave=lambda nodo: nodo.f)
