"""Evaluador de función de aptitud (Fitness) y error cuadrático medio (MSE)."""
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
from PIL import Image
from tp2.src.engine.renderer import Renderer
from tp2.src.models.individual import Individual

class FitnessEvaluator:
    """Evalúa individuos comparando su render contra la imagen objetivo."""

    def __init__(
        self,
        target_image_path: str,
        eval_size: Optional[Tuple[int, int]] = (64, 64),
        bg_color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    ) -> None:
        self.target_image_path = target_image_path
        self.raw_target = Image.open(target_image_path).convert("RGB")
        self.orig_width, self.orig_height = self.raw_target.size
        self.bg_color = bg_color

        if eval_size is not None:
            self.eval_width, self.eval_height = eval_size
            self.eval_target_img = self.raw_target.resize(
                (self.eval_width, self.eval_height), Image.Resampling.BILINEAR
            )
        else:
            self.eval_width, self.eval_height = self.orig_width, self.orig_height
            self.eval_target_img = self.raw_target

        self.target_array = np.asarray(self.eval_target_img, dtype=np.float32)
        self.renderer = Renderer(self.eval_width, self.eval_height, bg_color=self.bg_color)
        self.full_renderer = Renderer(self.orig_width, self.orig_height, bg_color=self.bg_color)

    def evaluate(self, individual: Individual) -> Tuple[float, float]:
        """Calcula (fitness, mse) para un individuo.
        
        Fitness: valor en (0, 1] donde 1 es réplica idéntica.
        MSE: Error cuadrático medio de intensidades RGB [0, 65025].
        """
        rendered_array = self.renderer.render_array(
            individual.chromosome.genes
        ).astype(np.float32)

        # Diferencia cuadrática media entre canales RGB
        diff = rendered_array - self.target_array
        mse = float(np.mean(diff ** 2))

        # Normalizamos MSE por 255^2 para acotar el error a [0, 1]
        norm_mse = mse / (255.0 ** 2)
        
        # Fitness estrictamente positivo en (0, 1]:
        fitness = 1.0 / (1.0 + norm_mse)

        individual.mse = mse
        individual.fitness = fitness
        return fitness, mse

    def render_full_resolution(self, individual: Individual) -> Image.Image:
        """Renderiza la mejor aproximación al tamaño original de la imagen objetivo."""
        return self.full_renderer.render_image(individual.chromosome.genes)
