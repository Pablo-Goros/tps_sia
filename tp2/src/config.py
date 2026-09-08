"""Manejo de configuración para el TP2 de Algoritmos Genéticos."""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class Config:
    image_path: str = "ejemplos/japon.png"
    num_triangles: int = 30
    eval_resolution: List[int] = field(default_factory=lambda: [64, 64])
    background_color: List[int] = field(default_factory=lambda: [255, 255, 255, 255])
    population_size: int = 40
    num_offspring: int = 40

    # Selección
    parent_selection_method: str = "ruleta"
    survival_selection_method: str = "elite"
    survival_strategy: str = "aditiva"
    generational_gap: float = 0.8
    selection_params: Dict[str, Any] = field(default_factory=dict)

    # Cruce
    crossover_method: str = "uniforme"
    crossover_pc: float = 0.85
    crossover_params: Dict[str, Any] = field(default_factory=dict)

    # Mutación
    mutation_method: str = "multigen_uniforme"
    mutation_pm: float = 0.1
    mutation_params: Dict[str, Any] = field(default_factory=dict)

    # Criterios de corte
    max_generations: int = 250
    timeout_seconds: Optional[float] = 60.0
    target_fitness: Optional[float] = None
    target_mse: Optional[float] = None
    content_window: Optional[int] = None
    content_delta: float = 1e-5
    structure_window: Optional[int] = None
    structure_threshold: float = 1e-4

    # Salida
    output_dir: str = "salidas"
    snapshot_interval: int = 25

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Config:
        cfg = cls()
        if "image_path" in data:
            cfg.image_path = str(data["image_path"])
        if "num_triangles" in data:
            cfg.num_triangles = int(data["num_triangles"])
        if "eval_resolution" in data:
            cfg.eval_resolution = list(data["eval_resolution"])
        if "background_color" in data:
            cfg.background_color = list(data["background_color"])
        if "population_size" in data:
            cfg.population_size = int(data["population_size"])
        if "num_offspring" in data:
            cfg.num_offspring = int(data["num_offspring"])

        # Selección
        sel = data.get("selection", {})
        if "parent_method" in sel:
            cfg.parent_selection_method = str(sel["parent_method"])
        if "survival_method" in sel:
            cfg.survival_selection_method = str(sel["survival_method"])
        if "survival_strategy" in sel:
            cfg.survival_strategy = str(sel["survival_strategy"])
        if "generational_gap" in sel:
            cfg.generational_gap = float(sel["generational_gap"])
        if "params" in sel:
            cfg.selection_params = dict(sel["params"])

        # Cruce
        cross = data.get("crossover", {})
        if "method" in cross:
            cfg.crossover_method = str(cross["method"])
        if "pc" in cross:
            cfg.crossover_pc = float(cross["pc"])
        if "params" in cross:
            cfg.crossover_params = dict(cross["params"])

        # Mutación
        mut = data.get("mutation", {})
        if "method" in mut:
            cfg.mutation_method = str(mut["method"])
        if "pm" in mut:
            cfg.mutation_pm = float(mut["pm"])
        if "params" in mut:
            cfg.mutation_params = dict(mut["params"])

        # Corte
        stop = data.get("stopping", {})
        if "max_generations" in stop:
            cfg.max_generations = int(stop["max_generations"])
        if "timeout_seconds" in stop:
            cfg.timeout_seconds = float(stop["timeout_seconds"]) if stop["timeout_seconds"] is not None else None
        if "target_fitness" in stop:
            cfg.target_fitness = float(stop["target_fitness"]) if stop["target_fitness"] is not None else None
        if "target_mse" in stop:
            cfg.target_mse = float(stop["target_mse"]) if stop["target_mse"] is not None else None
        if "content_window" in stop:
            cfg.content_window = int(stop["content_window"]) if stop["content_window"] is not None else None
        if "content_delta" in stop:
            cfg.content_delta = float(stop["content_delta"])
        if "structure_window" in stop:
            cfg.structure_window = int(stop["structure_window"]) if stop["structure_window"] is not None else None
        if "structure_threshold" in stop:
            cfg.structure_threshold = float(stop["structure_threshold"])

        # Salida
        out = data.get("output", {})
        if "output_dir" in out:
            cfg.output_dir = str(out["output_dir"])
        if "snapshot_interval" in out:
            cfg.snapshot_interval = int(out["snapshot_interval"])

        return cfg

    @classmethod
    def load_json(cls, path: str) -> Config:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_json(self, path: str) -> None:
        data = {
            "image_path": self.image_path,
            "num_triangles": self.num_triangles,
            "eval_resolution": self.eval_resolution,
            "background_color": self.background_color,
            "population_size": self.population_size,
            "num_offspring": self.num_offspring,
            "selection": {
                "parent_method": self.parent_selection_method,
                "survival_method": self.survival_selection_method,
                "survival_strategy": self.survival_strategy,
                "generational_gap": self.generational_gap,
                "params": self.selection_params
            },
            "crossover": {
                "method": self.crossover_method,
                "pc": self.crossover_pc,
                "params": self.crossover_params
            },
            "mutation": {
                "method": self.mutation_method,
                "pm": self.mutation_pm,
                "params": self.mutation_params
            },
            "stopping": {
                "max_generations": self.max_generations,
                "timeout_seconds": self.timeout_seconds,
                "target_fitness": self.target_fitness,
                "target_mse": self.target_mse,
                "content_window": self.content_window,
                "content_delta": self.content_delta,
                "structure_window": self.structure_window,
                "structure_threshold": self.structure_threshold
            },
            "output": {
                "output_dir": self.output_dir,
                "snapshot_interval": self.snapshot_interval
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
