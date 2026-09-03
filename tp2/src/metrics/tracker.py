"""Métricas y seguimiento de evolución para análisis experimental."""
from __future__ import annotations
import csv
import os
import time
from dataclasses import dataclass
from typing import List
from PIL import Image
from tp2.src.models.individual import Individual

@dataclass
class GenerationRecord:
    generation: int
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    best_mse: float
    diversity: float
    elapsed_seconds: float

class MetricsTracker:
    def __init__(self, output_dir: str = "salidas") -> None:
        self.output_dir = output_dir
        self.snapshots_dir = os.path.join(output_dir, "snapshots")
        os.makedirs(self.snapshots_dir, exist_ok=True)
        self.records: List[GenerationRecord] = []
        self.start_time = time.time()

    def calculate_diversity(self, population: List[Individual]) -> float:
        """Calcula la diversidad genética como la varianza del fitness de la población."""
        if len(population) <= 1:
            return 0.0
        fitnesses = [ind.fitness or 0.0 for ind in population]
        mean_fit = sum(fitnesses) / len(fitnesses)
        variance = sum((f - mean_fit) ** 2 for f in fitnesses) / len(fitnesses)
        return float(variance)

    def record_generation(
        self,
        generation: int,
        population: List[Individual]
    ) -> GenerationRecord:
        fitnesses = [ind.fitness or 0.0 for ind in population]
        mses = [ind.mse if ind.mse is not None else float("inf") for ind in population]
        best_idx = int(max(range(len(fitnesses)), key=lambda i: fitnesses[i]))

        best_fit = fitnesses[best_idx]
        worst_fit = min(fitnesses)
        avg_fit = sum(fitnesses) / len(fitnesses)
        best_mse = mses[best_idx]
        diversity = self.calculate_diversity(population)
        elapsed = time.time() - self.start_time

        rec = GenerationRecord(
            generation=generation,
            best_fitness=best_fit,
            avg_fitness=avg_fit,
            worst_fitness=worst_fit,
            best_mse=best_mse,
            diversity=diversity,
            elapsed_seconds=elapsed
        )
        self.records.append(rec)
        return rec

    def save_snapshot(self, image: Image.Image, generation: int, prefix: str = "gen") -> str:
        path = os.path.join(self.snapshots_dir, f"{prefix}_{generation:05d}.png")
        image.save(path)
        return path

    def export_csv(self, filename: str = "metricas.csv") -> str:
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "generacion",
                "mejor_fitness",
                "promedio_fitness",
                "peor_fitness",
                "mejor_mse",
                "diversidad",
                "segundos_transcurridos"
            ])
            for r in self.records:
                writer.writerow([
                    r.generation,
                    f"{r.best_fitness:.6f}",
                    f"{r.avg_fitness:.6f}",
                    f"{r.worst_fitness:.6f}",
                    f"{r.best_mse:.2f}",
                    f"{r.diversity:.8f}",
                    f"{r.elapsed_seconds:.3f}"
                ])
        return filepath
