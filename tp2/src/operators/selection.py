"""Métodos de selección de Algoritmos Genéticos."""
from __future__ import annotations
import math
import random
from abc import ABC, abstractmethod
from typing import List
from tp2.src.models.individual import Individual

class SelectionMethod(ABC):
    @abstractmethod
    def select(
        self,
        population: List[Individual],
        k: int,
        generation: int = 0
    ) -> List[Individual]:
        """Selecciona K individuos de la población."""
        pass


class EliteSelection(SelectionMethod):
    """Selección Elite según fórmula: n(i) = ceil((K - i) / N)."""

    def select(
        self,
        population: List[Individual],
        k: int,
        generation: int = 0
    ) -> List[Individual]:
        if not population or k <= 0:
            return []
        
        # Ordenar de mayor a menor fitness
        sorted_pop = sorted(population, key=lambda ind: (ind.fitness or 0.0), reverse=True)
        n_pop = len(sorted_pop)
        selected: List[Individual] = []

        for i, ind in enumerate(sorted_pop):
            # n(i) = ceil((k - i) / n_pop)
            times = math.ceil((k - i) / n_pop)
            if times <= 0:
                break
            for _ in range(times):
                selected.append(ind.copy(keep_eval=True))
                if len(selected) == k:
                    return selected

        # En caso de necesitar completar hasta k
        while len(selected) < k:
            selected.append(sorted_pop[len(selected) % n_pop].copy(keep_eval=True))
            
        return selected[:k]


class RouletteSelection(SelectionMethod):
    """Selección por Ruleta proporcional al fitness relativo."""

    def select(
        self,
        population: List[Individual],
        k: int,
        generation: int = 0
    ) -> List[Individual]:
        if not population or k <= 0:
            return []

        fitnesses = [max(0.0, ind.fitness or 0.0) for ind in population]
        total_fit = sum(fitnesses)
        if total_fit <= 0:
            return [random.choice(population).copy(keep_eval=True) for _ in range(k)]

        # Probabilidades acumuladas q(i)
        q = []
        acc = 0.0
        for f in fitnesses:
            acc += f / total_fit
            q.append(acc)
        q[-1] = 1.0  # Asegurar cierre numérico

        selected: List[Individual] = []
        for _ in range(k):
            r = random.random()
            # Búsqueda de q_i-1 < r <= q_i
            chosen_idx = 0
            for idx, cum in enumerate(q):
                if r <= cum:
                    chosen_idx = idx
                    break
            selected.append(population[chosen_idx].copy(keep_eval=True))

        return selected


class UniversalSelection(SelectionMethod):
    """Selección Universal Estocástica (SUS) con punteros equidistantes."""

    def select(
        self,
        population: List[Individual],
        k: int,
        generation: int = 0
    ) -> List[Individual]:
        if not population or k <= 0:
            return []

        fitnesses = [max(0.0, ind.fitness or 0.0) for ind in population]
        total_fit = sum(fitnesses)
        if total_fit <= 0:
            return [random.choice(population).copy(keep_eval=True) for _ in range(k)]

        q = []
        acc = 0.0
        for f in fitnesses:
            acc += f / total_fit
            q.append(acc)
        q[-1] = 1.0

        r_base = random.random()
        selected: List[Individual] = []
        for j in range(k):
            # r_j = (r + j) / K
            r_j = (r_base + j) / k
            chosen_idx = 0
            for idx, cum in enumerate(q):
                if r_j <= cum:
                    chosen_idx = idx
                    break
            selected.append(population[chosen_idx].copy(keep_eval=True))

        return selected


class RankingSelection(SelectionMethod):
    """Selección por Ranking utilizando pseudo-aptitud: f'(i) = (N - rank(i) + 1) / N."""

    def select(
        self,
        population: List[Individual],
        k: int,
        generation: int = 0
    ) -> List[Individual]:
        if not population or k <= 0:
            return []

        n_pop = len(population)
        sorted_pop = sorted(population, key=lambda ind: (ind.fitness or 0.0), reverse=True)

        # rank(i) de 1 a N (1 es el mejor).
        # Para garantizar que el peor tenga pseudo-aptitud positiva no nula:
        # f'(i) = (N - rank + 1) / N
        pseudo_fits = [(n_pop - rank + 1) / n_pop for rank in range(1, n_pop + 1)]
        total_pseudo = sum(pseudo_fits)

        q = []
        acc = 0.0
        for pf in pseudo_fits:
            acc += pf / total_pseudo
            q.append(acc)
        q[-1] = 1.0

        selected: List[Individual] = []
        for _ in range(k):
            r = random.random()
            chosen_idx = 0
            for idx, cum in enumerate(q):
                if r <= cum:
                    chosen_idx = idx
                    break
            selected.append(sorted_pop[chosen_idx].copy(keep_eval=True))

        return selected


class BoltzmannSelection(SelectionMethod):
    """Selección de Boltzmann con temperatura exponencial decreciente:
    T(t) = Tc + (T0 - Tc) * e^(-k_rate * t)
    ExpVal(i, g, T) = e^(f(i)/T) / <e^(f(x)/T)>_g
    """

    def __init__(
        self,
        t0: float = 1.0,
        tc: float = 0.01,
        k_rate: float = 0.01
    ) -> None:
        self.t0 = t0
        self.tc = tc
        self.k_rate = k_rate

    def get_temperature(self, t: int) -> float:
        return self.tc + (self.t0 - self.tc) * math.exp(-self.k_rate * t)

    def select(
        self,
        population: List[Individual],
        k: int,
        generation: int = 0
    ) -> List[Individual]:
        if not population or k <= 0:
            return []

        temp = max(1e-5, self.get_temperature(generation))
        fitnesses = [ind.fitness or 0.0 for ind in population]
        max_f = max(fitnesses) if fitnesses else 0.0

        # Estabilidad numérica restando max_f
        exp_vals = [math.exp((f - max_f) / temp) for f in fitnesses]
        total_exp = sum(exp_vals)
        if total_exp <= 0:
            return [random.choice(population).copy(keep_eval=True) for _ in range(k)]

        q = []
        acc = 0.0
        for ev in exp_vals:
            acc += ev / total_exp
            q.append(acc)
        q[-1] = 1.0

        selected: List[Individual] = []
        for _ in range(k):
            r = random.random()
            chosen_idx = 0
            for idx, cum in enumerate(q):
                if r <= cum:
                    chosen_idx = idx
                    break
            selected.append(population[chosen_idx].copy(keep_eval=True))

        return selected


class DeterministicTournamentSelection(SelectionMethod):
    """Torneo Determinístico: toma M individuos al azar y elige el mejor."""

    def __init__(self, m: int = 3) -> None:
        self.m = max(2, m)

    def select(
        self,
        population: List[Individual],
        k: int,
        generation: int = 0
    ) -> List[Individual]:
        if not population or k <= 0:
            return []

        selected: List[Individual] = []
        n_pop = len(population)
        m = min(self.m, n_pop)

        for _ in range(k):
            candidates = random.sample(population, m)
            best = max(candidates, key=lambda ind: (ind.fitness or 0.0))
            selected.append(best.copy(keep_eval=True))

        return selected


class ProbabilisticTournamentSelection(SelectionMethod):
    """Torneo Probabilístico: toma 2 individuos al azar y elige el más apto
    con probabilidad threshold en [0.5, 1.0], o el menos apto con 1 - threshold."""

    def __init__(self, threshold: float = 0.75) -> None:
        self.threshold = max(0.5, min(1.0, threshold))

    def select(
        self,
        population: List[Individual],
        k: int,
        generation: int = 0
    ) -> List[Individual]:
        if not population or k <= 0:
            return []

        selected: List[Individual] = []
        n_pop = len(population)

        for _ in range(k):
            ind1, ind2 = random.sample(population, 2 if n_pop >= 2 else 1)
            fit1 = ind1.fitness or 0.0
            fit2 = ind2.fitness or 0.0
            
            fittest = ind1 if fit1 >= fit2 else ind2
            less_fit = ind2 if fit1 >= fit2 else ind1

            r = random.random()
            if r < self.threshold:
                selected.append(fittest.copy(keep_eval=True))
            else:
                selected.append(less_fit.copy(keep_eval=True))

        return selected


def get_selection_operator(name: str, **kwargs) -> SelectionMethod:
    """Factory para instanciar métodos de selección por nombre."""
    normalized = name.strip().lower()
    if normalized in ["elite", "elitist"]:
        return EliteSelection()
    elif normalized in ["ruleta", "roulette"]:
        return RouletteSelection()
    elif normalized in ["universal", "sus"]:
        return UniversalSelection()
    elif normalized in ["ranking"]:
        return RankingSelection()
    elif normalized in ["boltzmann"]:
        t0 = float(kwargs.get("t0", 1.0))
        tc = float(kwargs.get("tc", 0.01))
        k_rate = float(kwargs.get("k_rate", 0.01))
        return BoltzmannSelection(t0=t0, tc=tc, k_rate=k_rate)
    elif normalized in ["torneo_deterministico", "tournament_deterministic", "deterministic_tournament"]:
        m = int(kwargs.get("tournament_m", 3))
        return DeterministicTournamentSelection(m=m)
    elif normalized in ["torneo_probabilistico", "tournament_probabilistic", "probabilistic_tournament"]:
        thresh = float(kwargs.get("tournament_threshold", 0.75))
        return ProbabilisticTournamentSelection(threshold=thresh)
    else:
        raise ValueError(f"Método de selección desconocido: '{name}'")
