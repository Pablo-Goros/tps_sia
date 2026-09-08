"""Operadores de mutación: Gen, Multigen Limitada, Multigen Uniforme y No Uniforme."""
from __future__ import annotations
import random
from abc import ABC, abstractmethod
from tp2.src.models.individual import Individual

class MutationMethod(ABC):
    def __init__(self, pm: float = 0.05) -> None:
        self.pm = pm

    @abstractmethod
    def mutate(
        self,
        individual: Individual,
        generation: int = 0,
        max_generations: int = 1000
    ) -> Individual:
        """Aplica mutación sobre el individuo."""
        pass


class SingleGeneMutation(MutationMethod):
    """Mutación de Gen: altera un solo gen con probabilidad Pm."""

    def mutate(
        self,
        individual: Individual,
        generation: int = 0,
        max_generations: int = 1000
    ) -> Individual:
        if random.random() <= self.pm:
            genes = individual.chromosome.genes
            if genes:
                idx = random.randrange(len(genes))
                # 80% perturbación delta, 20% reinicialización aleatoria
                if random.random() < 0.8:
                    genes[idx].mutate_delta(scale=1.0)
                else:
                    genes[idx].mutate_random()
                individual.fitness = None
                individual.mse = None
        return individual


class LimitedMultiGeneMutation(MutationMethod):
    """Mutación Multigen Limitada: con probabilidad Pm, se selecciona una cantidad
    azarosa de m en [1, M] genes a mutar."""

    def __init__(self, pm: float = 0.1, max_genes: int = 3) -> None:
        super().__init__(pm=pm)
        self.max_genes = max(1, max_genes)

    def mutate(
        self,
        individual: Individual,
        generation: int = 0,
        max_generations: int = 1000
    ) -> Individual:
        if random.random() <= self.pm:
            genes = individual.chromosome.genes
            if genes:
                m = min(len(genes), random.randint(1, self.max_genes))
                indices = random.sample(range(len(genes)), m)
                for idx in indices:
                    if random.random() < 0.8:
                        genes[idx].mutate_delta(scale=1.0)
                    else:
                        genes[idx].mutate_random()
                individual.fitness = None
                individual.mse = None
        return individual


class UniformMultiGeneMutation(MutationMethod):
    """Mutación Multigen Uniforme: cada gen tiene una probabilidad independiente
    Pm de ser mutado."""

    def mutate(
        self,
        individual: Individual,
        generation: int = 0,
        max_generations: int = 1000
    ) -> Individual:
        mutated = False
        genes = individual.chromosome.genes
        for tri in genes:
            if random.random() <= self.pm:
                if random.random() < 0.8:
                    tri.mutate_delta(scale=1.0)
                else:
                    tri.mutate_random()
                mutated = True

        if mutated:
            individual.fitness = None
            individual.mse = None
        return individual


class NonUniformMutation(MutationMethod):
    """Mutación No Uniforme: la magnitud del delta decrece con el avance de las
    generaciones t: scale = (1 - t / T_max)^b, favoreciendo exploración inicial
    y explotación/ajuste fino al final."""

    def __init__(self, pm: float = 0.1, b: float = 2.0) -> None:
        super().__init__(pm=pm)
        self.b = b

    def mutate(
        self,
        individual: Individual,
        generation: int = 0,
        max_generations: int = 1000
    ) -> Individual:
        # Factor de enfriamiento / reducción de escala de exploración
        t = min(generation, max_generations)
        t_max = max(1, max_generations)
        scale = max(0.01, (1.0 - (t / t_max)) ** self.b)

        mutated = False
        genes = individual.chromosome.genes
        for tri in genes:
            if random.random() <= self.pm:
                # La probabilidad de regeneración pura también decrece con scale
                if random.random() < (0.3 * scale):
                    tri.mutate_random()
                else:
                    tri.mutate_delta(scale=scale)
                mutated = True

        if mutated:
            individual.fitness = None
            individual.mse = None
        return individual


def get_mutation_operator(name: str, pm: float = 0.05, **kwargs) -> MutationMethod:
    """Factory para instanciar métodos de mutación."""
    normalized = name.strip().lower()
    if normalized in ["gen", "single_gene", "single"]:
        return SingleGeneMutation(pm=pm)
    elif normalized in ["multigen_limitada", "limited_multigene", "multigen_limited"]:
        max_m = int(kwargs.get("max_genes_to_mutate", 3))
        return LimitedMultiGeneMutation(pm=pm, max_genes=max_m)
    elif normalized in ["multigen_uniforme", "uniform_multigene", "uniform"]:
        return UniformMultiGeneMutation(pm=pm)
    elif normalized in ["no_uniforme", "non_uniform", "nouniforme"]:
        b = float(kwargs.get("non_uniform_b", 2.0))
        return NonUniformMutation(pm=pm, b=b)
    else:
        raise ValueError(
            f"Método de mutación desconocido: '{name}'. "
            f"Opciones: 'gen', 'multigen_limitada', 'multigen_uniforme', 'no_uniforme'."
        )
