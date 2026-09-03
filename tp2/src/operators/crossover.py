"""Operadores de cruce (recombinación): Dos Puntos y Uniforme."""
from __future__ import annotations
import random
from abc import ABC, abstractmethod
from typing import Tuple
from tp2.src.models.chromosome import Chromosome
from tp2.src.models.individual import Individual

class CrossoverMethod(ABC):
    def __init__(self, pc: float = 0.85) -> None:
        self.pc = pc

    @abstractmethod
    def cross(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Aplica cruce a dos padres y retorna dos hijos."""
        pass


class TwoPointCrossover(CrossoverMethod):
    """Cruce de dos puntos: elige dos locus al azar P1 <= P2 e intercambia
    los segmentos centrales."""

    def cross(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> Tuple[Individual, Individual]:
        if random.random() > self.pc:
            return parent1.copy(), parent2.copy()

        genes1 = [g.copy() for g in parent1.chromosome.genes]
        genes2 = [g.copy() for g in parent2.chromosome.genes]
        length = len(genes1)

        if length <= 1:
            return Individual(chromosome=Chromosome(genes=genes1)), Individual(chromosome=Chromosome(genes=genes2))

        # Seleccionar dos puntos P1 <= P2 en [0, length]
        pt1 = random.randint(0, length)
        pt2 = random.randint(0, length)
        if pt1 > pt2:
            pt1, pt2 = pt2, pt1

        child1_genes = genes1[:pt1] + genes2[pt1:pt2] + genes1[pt2:]
        child2_genes = genes2[:pt1] + genes1[pt1:pt2] + genes2[pt2:]

        return (
            Individual(chromosome=Chromosome(genes=child1_genes)),
            Individual(chromosome=Chromosome(genes=child2_genes))
        )


class UniformCrossover(CrossoverMethod):
    """Cruce uniforme: cada gen se hereda de uno u otro padre con probabilidad p."""

    def __init__(self, pc: float = 0.85, p_swap: float = 0.5) -> None:
        super().__init__(pc=pc)
        self.p_swap = p_swap

    def cross(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> Tuple[Individual, Individual]:
        if random.random() > self.pc:
            return parent1.copy(), parent2.copy()

        genes1 = parent1.chromosome.genes
        genes2 = parent2.chromosome.genes
        length = min(len(genes1), len(genes2))

        child1_genes = []
        child2_genes = []

        for i in range(length):
            if random.random() < self.p_swap:
                child1_genes.append(genes1[i].copy())
                child2_genes.append(genes2[i].copy())
            else:
                child1_genes.append(genes2[i].copy())
                child2_genes.append(genes1[i].copy())

        return (
            Individual(chromosome=Chromosome(genes=child1_genes)),
            Individual(chromosome=Chromosome(genes=child2_genes))
        )


def get_crossover_operator(name: str, pc: float = 0.85, **kwargs) -> CrossoverMethod:
    """Factory para instanciar métodos de cruce."""
    normalized = name.strip().lower()
    if normalized in ["dos_puntos", "two_point", "two_points", "2_puntos"]:
        return TwoPointCrossover(pc=pc)
    elif normalized in ["uniforme", "uniform"]:
        p_swap = float(kwargs.get("p_swap", 0.5))
        return UniformCrossover(pc=pc, p_swap=p_swap)
    else:
        raise ValueError(
            f"Método de cruce no soportado: '{name}'. Opciones disponibles: 'dos_puntos', 'uniforme'."
        )
