"""Criterios de parada: generaciones, timeout, fitness aceptable, contenido y estructura."""
from __future__ import annotations
import time
from typing import List, Tuple, Optional
from tp2.src.models.individual import Individual

MAX_ALLOWED_TRIANGLES = 200  # Cota lógica superior para evitar colapsos de memoria/tiempo

def validate_triangle_count(num_triangles: int, max_limit: int = MAX_ALLOWED_TRIANGLES) -> None:
    if num_triangles < 1:
        raise ValueError(f"La cantidad de triángulos debe ser al menos 1 (recibido {num_triangles}).")
    if num_triangles > max_limit:
        raise ValueError(
            f"Tope lógico excedido: se solicitaron {num_triangles} triángulos, "
            f"pero el tope máximo seguro de ejecución es {max_limit}. "
            f"Ajuste el valor en la configuración."
        )

class StoppingCondition:
    def __init__(
        self,
        max_generations: int = 500,
        timeout_seconds: Optional[float] = 60.0,
        target_fitness: Optional[float] = None,
        target_mse: Optional[float] = None,
        content_window: Optional[int] = None,
        content_delta: float = 1e-5,
        structure_window: Optional[int] = None,
        structure_threshold: float = 1e-4
    ) -> None:
        self.max_generations = max_generations
        self.timeout_seconds = timeout_seconds
        self.target_fitness = target_fitness
        self.target_mse = target_mse
        self.content_window = content_window
        self.content_delta = content_delta
        self.structure_window = structure_window
        self.structure_threshold = structure_threshold

        self.start_time = time.time()
        self.best_fitness_history: List[float] = []
        self.diversity_history: List[float] = []

    def reset(self) -> None:
        self.start_time = time.time()
        self.best_fitness_history.clear()
        self.diversity_history.clear()

    def should_stop(
        self,
        generation: int,
        best_individual: Individual,
        diversity: float
    ) -> Tuple[bool, str]:
        """Evalúa si se cumple alguna de las condiciones de parada activas."""
        # 1. Timeout de tiempo
        elapsed = time.time() - self.start_time
        if self.timeout_seconds is not None and elapsed >= self.timeout_seconds:
            return True, f"Timeout alcanzado: {elapsed:.2f}s transcurridos (límite: {self.timeout_seconds}s)"

        # 2. Máxima cantidad de generaciones
        if generation >= self.max_generations:
            return True, f"Máximo de generaciones alcanzado ({self.max_generations})"

        # 3. Solución aceptable por fitness
        best_f = best_individual.fitness or 0.0
        if self.target_fitness is not None and best_f >= self.target_fitness:
            return True, f"Fitness objetivo alcanzado: {best_f:.6f} >= {self.target_fitness:.6f}"

        # 4. Solución aceptable por MSE
        best_mse = best_individual.mse if best_individual.mse is not None else float("inf")
        if self.target_mse is not None and best_mse <= self.target_mse:
            return True, f"MSE objetivo alcanzado: {best_mse:.2f} <= {self.target_mse:.2f}"

        # 5. Criterio de Contenido (estancamiento del mejor fitness)
        self.best_fitness_history.append(best_f)
        if self.content_window is not None and len(self.best_fitness_history) >= self.content_window:
            window = self.best_fitness_history[-self.content_window:]
            if (max(window) - min(window)) < self.content_delta:
                return True, (
                    f"Criterio de contenido: el mejor fitness no mejoró en más de "
                    f"{self.content_delta} durante {self.content_window} generaciones"
                )

        # 6. Criterio de Estructura (estancamiento por pérdida de diversidad)
        self.diversity_history.append(diversity)
        if self.structure_window is not None and len(self.diversity_history) >= self.structure_window:
            recent_div = self.diversity_history[-self.structure_window:]
            if all(d < self.structure_threshold for d in recent_div):
                return True, (
                    f"Criterio de estructura: diversidad poblacional inferior a "
                    f"{self.structure_threshold} durante {self.structure_window} generaciones"
                )

        return False, ""
