"""Modelo del gen: Triángulo translúcido."""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Any, Tuple

@dataclass
class Triangle:
    """Gen que representa un triángulo en un espacio normalizado [0, 1] x [0, 1]
    junto con su color uniforme translúcido RGBA."""
    x1: float
    y1: float
    x2: float
    y2: float
    x3: float
    y3: float
    r: int
    g: int
    b: int
    a: int  # Opacidad 0 (transparente) a 255 (opaco)

    @classmethod
    def random(cls, min_alpha: int = 30, max_alpha: int = 220) -> Triangle:
        """Crea un triángulo aleatorio en el espacio [0, 1] con color RGBA."""
        return cls(
            x1=random.random(),
            y1=random.random(),
            x2=random.random(),
            y2=random.random(),
            x3=random.random(),
            y3=random.random(),
            r=random.randint(0, 255),
            g=random.randint(0, 255),
            b=random.randint(0, 255),
            a=random.randint(min_alpha, max_alpha),
        )

    def copy(self) -> Triangle:
        return Triangle(
            x1=self.x1, y1=self.y1,
            x2=self.x2, y2=self.y2,
            x3=self.x3, y3=self.y3,
            r=self.r, g=self.g, b=self.b, a=self.a
        )

    def mutate_delta(
        self,
        scale: float = 1.0,
        sigma_pos: float = 0.1,
        sigma_color: float = 25.0,
        sigma_alpha: float = 20.0
    ) -> None:
        """Aplica una perturbación delta gaussiana a los alelos del triángulo.
        
        Args:
            scale: factor de escala para la mutación no uniforme (1.0 = máxima, -> 0.0 hacia el final).
            sigma_pos: desvío estándar de posición en coordenadas [0, 1].
            sigma_color: desvío estándar de canales RGB [0, 255].
            sigma_alpha: desvío estándar de canal Alpha [0, 255].
        """
        eff_pos = sigma_pos * scale
        eff_col = sigma_color * scale
        eff_alp = sigma_alpha * scale

        # Mutar coordenadas
        self.x1 = max(0.0, min(1.0, self.x1 + random.gauss(0.0, eff_pos)))
        self.y1 = max(0.0, min(1.0, self.y1 + random.gauss(0.0, eff_pos)))
        self.x2 = max(0.0, min(1.0, self.x2 + random.gauss(0.0, eff_pos)))
        self.y2 = max(0.0, min(1.0, self.y2 + random.gauss(0.0, eff_pos)))
        self.x3 = max(0.0, min(1.0, self.x3 + random.gauss(0.0, eff_pos)))
        self.y3 = max(0.0, min(1.0, self.y3 + random.gauss(0.0, eff_pos)))

        # Mutar color
        self.r = int(max(0, min(255, round(self.r + random.gauss(0.0, eff_col)))))
        self.g = int(max(0, min(255, round(self.g + random.gauss(0.0, eff_col)))))
        self.b = int(max(0, min(255, round(self.b + random.gauss(0.0, eff_col)))))
        self.a = int(max(10, min(255, round(self.a + random.gauss(0.0, eff_alp)))))

    def mutate_random(self, min_alpha: int = 30, max_alpha: int = 220) -> None:
        """Reemplaza alelos por valores completamente nuevos (exploración pura)."""
        self.x1 = random.random()
        self.y1 = random.random()
        self.x2 = random.random()
        self.y2 = random.random()
        self.x3 = random.random()
        self.y3 = random.random()
        self.r = random.randint(0, 255)
        self.g = random.randint(0, 255)
        self.b = random.randint(0, 255)
        self.a = random.randint(min_alpha, max_alpha)

    def to_pixel_points(self, width: int, height: int) -> list[Tuple[float, float]]:
        """Convierte vértices normalizados a coordenadas en píxeles."""
        return [
            (self.x1 * (width - 1), self.y1 * (height - 1)),
            (self.x2 * (width - 1), self.y2 * (height - 1)),
            (self.x3 * (width - 1), self.y3 * (height - 1)),
        ]

    @property
    def rgba(self) -> Tuple[int, int, int, int]:
        return (self.r, self.g, self.b, self.a)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vertices": [
                {"x": round(self.x1, 5), "y": round(self.y1, 5)},
                {"x": round(self.x2, 5), "y": round(self.y2, 5)},
                {"x": round(self.x3, 5), "y": round(self.y3, 5)},
            ],
            "color": {
                "r": self.r,
                "g": self.g,
                "b": self.b,
                "a": self.a
            }
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Triangle:
        v = d["vertices"]
        c = d["color"]
        return cls(
            x1=float(v[0]["x"]), y1=float(v[0]["y"]),
            x2=float(v[1]["x"]), y2=float(v[1]["y"]),
            x3=float(v[2]["x"]), y3=float(v[2]["y"]),
            r=int(c["r"]), g=int(c["g"]), b=int(c["b"]), a=int(c["a"])
        )
