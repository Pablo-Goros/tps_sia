"""Estrategias de supervivencia: Aditiva, Exclusiva y Brecha Generacional."""
from __future__ import annotations
import math
from abc import ABC, abstractmethod
from typing import List
from tp2.src.models.individual import Individual
from tp2.src.operators.selection import SelectionMethod

class SurvivalStrategy(ABC):
    def __init__(self, selector: SelectionMethod) -> None:
        self.selector = selector

    @abstractmethod
    def form_next_generation(
        self,
        current_pop: List[Individual],
        offspring: List[Individual],
        n: int,
        generation: int = 0
    ) -> List[Individual]:
        """Forma la nueva población de tamaño N a partir de la población actual e hijos."""
        pass


class AdditiveSurvival(SurvivalStrategy):
    """Supervivencia Aditiva: forma un conjunto combinado [N + K] y selecciona N."""

    def form_next_generation(
        self,
        current_pop: List[Individual],
        offspring: List[Individual],
        n: int,
        generation: int = 0
    ) -> List[Individual]:
        pool = current_pop + offspring
        return self.selector.select(pool, k=n, generation=generation)


class ExclusiveSurvival(SurvivalStrategy):
    """Supervivencia Exclusiva:
    Si K > N: selecciona N de los K hijos exclusivamente.
    Si K <= N: toma los K hijos + selecciona (N - K) de los padres.
    """

    def form_next_generation(
        self,
        current_pop: List[Individual],
        offspring: List[Individual],
        n: int,
        generation: int = 0
    ) -> List[Individual]:
        k = len(offspring)
        if k > n:
            return self.selector.select(offspring, k=n, generation=generation)
        else:
            needed = n - k
            parents_selected = (
                self.selector.select(current_pop, k=needed, generation=generation)
                if needed > 0
                else []
            )
            return [ind.copy(keep_eval=True) for ind in offspring] + parents_selected


class GenerationalGapSurvival(SurvivalStrategy):
    """Supervivencia por Brecha Generacional G en [0, 1]:
    (1 - G) * N seleccionados de la población previa + G * N seleccionados de los hijos.
    """

    def __init__(self, selector: SelectionMethod, gap: float = 0.8) -> None:
        super().__init__(selector)
        self.gap = max(0.0, min(1.0, gap))

    def form_next_generation(
        self,
        current_pop: List[Individual],
        offspring: List[Individual],
        n: int,
        generation: int = 0
    ) -> List[Individual]:
        n_offspring = int(round(self.gap * n))
        n_parents = n - n_offspring

        selected_parents = (
            self.selector.select(current_pop, k=n_parents, generation=generation)
            if n_parents > 0
            else []
        )
        selected_offspring = (
            self.selector.select(offspring, k=n_offspring, generation=generation)
            if n_offspring > 0
            else []
        )
        return selected_parents + selected_offspring


def get_survival_strategy(
    name: str,
    selector: SelectionMethod,
    gap: float = 0.8
) -> SurvivalStrategy:
    """Factory para instanciar la estrategia de supervivencia."""
    normalized = name.strip().lower()
    if normalized in ["aditiva", "additive"]:
        return AdditiveSurvival(selector=selector)
    elif normalized in ["exclusiva", "exclusive"]:
        return ExclusiveSurvival(selector=selector)
    elif normalized in ["brecha", "gap", "generational_gap"]:
        return GenerationalGapSurvival(selector=selector, gap=gap)
    else:
        raise ValueError(
            f"Estrategia de supervivencia desconocida: '{name}'. "
            f"Opciones: 'aditiva', 'exclusiva', 'brecha'."
        )
