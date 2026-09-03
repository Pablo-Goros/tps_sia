"""Cromosoma compuesto por una secuencia de triángulos (genes)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List
from tp2.src.models.triangle import Triangle

@dataclass
class Chromosome:
    genes: List[Triangle] = field(default_factory=list)

    @classmethod
    def random(cls, num_triangles: int) -> Chromosome:
        """Crea un cromosoma aleatorio con num_triangles genes."""
        return cls(genes=[Triangle.random() for _ in range(num_triangles)])

    def copy(self) -> Chromosome:
        """Copia profunda del cromosoma."""
        return Chromosome(genes=[g.copy() for g in self.genes])

    def __len__(self) -> int:
        return len(self.genes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_triangles": len(self.genes),
            "triangles": [g.to_dict() for g in self.genes]
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chromosome:
        return cls(genes=[Triangle.from_dict(t) for t in d["triangles"]])
