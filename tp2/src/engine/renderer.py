"""Motor de renderizado de triángulos con soporte de mezcla alfa."""
from __future__ import annotations
from typing import Tuple, Sequence, Optional
import numpy as np
from PIL import Image, ImageDraw
from tp2.src.models.triangle import Triangle

class Renderer:
    def __init__(
        self,
        width: int,
        height: int,
        bg_color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    ) -> None:
        self.width = width
        self.height = height
        self.bg_color = bg_color

    def render_image(
        self,
        triangles: Sequence[Triangle],
        target_size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """Renderiza la secuencia de triángulos en una imagen RGBA."""
        w, h = target_size if target_size else (self.width, self.height)
        base = Image.new("RGBA", (w, h), self.bg_color)
        
        for tri in triangles:
            overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            draw.polygon(tri.to_pixel_points(w, h), fill=tri.rgba)
            base = Image.alpha_composite(base, overlay)
            
        return base

    def render_array(
        self,
        triangles: Sequence[Triangle],
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Renderiza y devuelve un array NumPy RGB (H, W, 3) con dtype uint8."""
        img = self.render_image(triangles, target_size=target_size)
        rgb_img = img.convert("RGB")
        return np.asarray(rgb_img, dtype=np.uint8)
