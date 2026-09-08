"""Motor principal de Algoritmos Genéticos para aproximación de imágenes."""
from __future__ import annotations
import json
import os
import random
import time
from typing import List, Optional, Callable, Dict, Any
from tp2.src.models.individual import Individual
from tp2.src.engine.fitness import FitnessEvaluator
from tp2.src.operators.selection import SelectionMethod
from tp2.src.operators.crossover import CrossoverMethod
from tp2.src.operators.mutation import MutationMethod
from tp2.src.operators.survival import SurvivalStrategy
from tp2.src.stopping.stopping import StoppingCondition, validate_triangle_count
from tp2.src.metrics.tracker import MetricsTracker

class GeneticAlgorithm:
    def __init__(
        self,
        evaluator: FitnessEvaluator,
        parent_selector: SelectionMethod,
        crossover_operator: CrossoverMethod,
        mutation_operator: MutationMethod,
        survival_strategy: SurvivalStrategy,
        stopping_condition: StoppingCondition,
        tracker: MetricsTracker,
        population_size: int = 50,
        num_offspring: int = 50,
        num_triangles: int = 50,
        snapshot_interval: int = 50,
        progress_callback: Optional[Callable[[int, float, float, float], None]] = None
    ) -> None:
        # Validación de cota lógica de triángulos
        validate_triangle_count(num_triangles)

        self.evaluator = evaluator
        self.parent_selector = parent_selector
        self.crossover_operator = crossover_operator
        self.mutation_operator = mutation_operator
        self.survival_strategy = survival_strategy
        self.stopping_condition = stopping_condition
        self.tracker = tracker
        self.population_size = population_size
        self.num_offspring = num_offspring
        self.num_triangles = num_triangles
        self.snapshot_interval = snapshot_interval
        self.progress_callback = progress_callback

        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None

    def initialize_population(self) -> None:
        """Crea la Generación 0 con N individuos aleatorios y los evalúa."""
        self.population = [
            Individual.random(self.num_triangles)
            for _ in range(self.population_size)
        ]
        for ind in self.population:
            self.evaluator.evaluate(ind)

        self.best_individual = max(
            self.population, key=lambda ind: (ind.fitness or 0.0)
        ).copy(keep_eval=True)

    def run(self) -> Dict[str, Any]:
        """Ejecuta el ciclo evolutivo completo respetando los criterios de parada."""
        self.stopping_condition.reset()
        t_start = time.time()

        self.initialize_population()

        # Registro de la Generación 0
        rec0 = self.tracker.record_generation(0, self.population)
        if self.snapshot_interval > 0 and self.best_individual:
            img = self.evaluator.render_full_resolution(self.best_individual)
            self.tracker.save_snapshot(img, 0)

        if self.progress_callback and self.best_individual:
            self.progress_callback(
                0, self.best_individual.fitness or 0.0,
                self.best_individual.mse or 0.0, 0.0
            )

        stop_reason = ""
        generation = 0

        while True:
            generation += 1
            current_diversity = self.tracker.calculate_diversity(self.population)

            # Verificar criterios de parada
            should_stop, reason = self.stopping_condition.should_stop(
                generation, self.best_individual, current_diversity
            )
            if should_stop:
                stop_reason = reason
                break

            # 1. Selección de K padres
            parents = self.parent_selector.select(
                self.population, k=self.num_offspring, generation=generation
            )
            random.shuffle(parents)

            # 2. Cruce: generar K descendientes
            offspring: List[Individual] = []
            for i in range(0, len(parents), 2):
                p1 = parents[i]
                p2 = parents[i + 1] if i + 1 < len(parents) else parents[0]
                c1, c2 = self.crossover_operator.cross(p1, p2)
                offspring.append(c1)
                if len(offspring) < self.num_offspring:
                    offspring.append(c2)

            # 3. Mutación sobre los K hijos
            max_gen = self.stopping_condition.max_generations
            for child in offspring:
                self.mutation_operator.mutate(
                    child, generation=generation, max_generations=max_gen
                )

            # 4. Evaluación de descendientes
            for child in offspring:
                self.evaluator.evaluate(child)

            # 5. Supervivencia y conformación de la nueva generación N
            self.population = self.survival_strategy.form_next_generation(
                current_pop=self.population,
                offspring=offspring,
                n=self.population_size,
                generation=generation
            )

            # 6. Actualizar el mejor individuo global
            current_gen_best = max(
                self.population, key=lambda ind: (ind.fitness or 0.0)
            )
            if (
                self.best_individual is None
                or (current_gen_best.fitness or 0.0) > (self.best_individual.fitness or 0.0)
            ):
                self.best_individual = current_gen_best.copy(keep_eval=True)

            # 7. Seguimiento de métricas
            elapsed = time.time() - t_start
            rec = self.tracker.record_generation(generation, self.population)

            # Snapshot periódico
            if (
                self.snapshot_interval > 0
                and generation % self.snapshot_interval == 0
                and self.best_individual
            ):
                img = self.evaluator.render_full_resolution(self.best_individual)
                self.tracker.save_snapshot(img, generation)

            if self.progress_callback and self.best_individual:
                self.progress_callback(
                    generation,
                    self.best_individual.fitness or 0.0,
                    self.best_individual.mse or 0.0,
                    elapsed
                )

        # Fin de ejecución: guardar imagen final, triángulos y métricas
        total_time = time.time() - t_start
        final_img = (
            self.evaluator.render_full_resolution(self.best_individual)
            if self.best_individual
            else None
        )
        
        best_img_path = os.path.join(self.tracker.output_dir, "mejor_imagen.png")
        if final_img:
            final_img.save(best_img_path)

        triangles_path = os.path.join(self.tracker.output_dir, "triangulos.json")
        if self.best_individual:
            with open(triangles_path, "w", encoding="utf-8") as f:
                json.dump(self.best_individual.to_dict(), f, indent=2)

        csv_path = self.tracker.export_csv("metricas.csv")

        return {
            "stop_reason": stop_reason,
            "generations_completed": generation,
            "total_time_seconds": total_time,
            "best_fitness": self.best_individual.fitness if self.best_individual else 0.0,
            "best_mse": self.best_individual.mse if self.best_individual else float("inf"),
            "best_image_path": best_img_path,
            "triangles_json_path": triangles_path,
            "metrics_csv_path": csv_path,
        }
