"""Individuo que representa una solución en la población del AG."""
from __future__ import annotations
from dataclasses import dataclass, field
import uuid
from typing import Optional, Any
from tp2.src.models.chromosome import Chromosome

@dataclass
class Individual:
    chromosome: Chromosome
    fitness: Optional[float] = None
    mse: Optional[float] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @classmethod
    def random(cls, num_triangles: int) -> Individual:
        return cls(chromosome=Chromosome.random(num_triangles))

    def copy(self, keep_eval: bool = False) -> Individual:
        ind = Individual(chromosome=self.chromosome.copy())
        if keep_eval:
            ind.fitness = self.fitness
            ind.mse = self.mse
        return ind

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "fitness": self.fitness,
            "mse": self.mse,
            "chromosome": self.chromosome.to_dict()
        }
