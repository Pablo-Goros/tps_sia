"""Interfaz de Línea de Comandos (CLI) para el TP2 de Algoritmos Genéticos."""
from __future__ import annotations
import argparse
import os
import sys
import time
from tp2.src.config import Config
from tp2.src.engine.fitness import FitnessEvaluator
from tp2.src.operators.selection import get_selection_operator
from tp2.src.operators.crossover import get_crossover_operator
from tp2.src.operators.mutation import get_mutation_operator
from tp2.src.operators.survival import get_survival_strategy
from tp2.src.stopping.stopping import StoppingCondition, validate_triangle_count, MAX_ALLOWED_TRIANGLES
from tp2.src.metrics.tracker import MetricsTracker
from tp2.src.engine.genetic_algorithm import GeneticAlgorithm

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TP2 SIA: Aproximador de Imágenes con Triángulos mediante Algoritmos Genéticos"
    )
    parser.add_argument(
        "-c", "--config", type=str, default="tp2/configuracion.json",
        help="Ruta al archivo de configuración JSON"
    )
    parser.add_argument("-i", "--image", type=str, help="Ruta a la imagen objetivo a procesar")
    parser.add_argument("-t", "--triangles", type=int, help=f"Cantidad de triángulos (tope lógico: {MAX_ALLOWED_TRIANGLES})")
    parser.add_argument("-g", "--generations", type=int, help="Máxima cantidad de generaciones")
    parser.add_argument("--timeout", type=float, help="Timeout máximo de ejecución en segundos")
    parser.add_argument("-n", "--pop-size", type=int, help="Tamaño de la población (N)")
    parser.add_argument("-k", "--offspring", type=int, help="Cantidad de hijos por generación (K)")
    parser.add_argument(
        "--parent-sel", type=str,
        choices=["elite", "ruleta", "universal", "ranking", "boltzmann", "torneo_deterministico", "torneo_probabilistico"],
        help="Método de selección de padres"
    )
    parser.add_argument(
        "--survival-sel", type=str,
        choices=["elite", "ruleta", "universal", "ranking", "boltzmann", "torneo_deterministico", "torneo_probabilistico"],
        help="Método de selección para supervivencia"
    )
    parser.add_argument(
        "--survival-strat", type=str,
        choices=["aditiva", "exclusiva", "brecha"],
        help="Estrategia de supervivencia"
    )
    parser.add_argument(
        "--crossover", type=str,
        choices=["dos_puntos", "uniforme"],
        help="Método de cruce"
    )
    parser.add_argument(
        "--mutation", type=str,
        choices=["gen", "multigen_limitada", "multigen_uniforme", "no_uniforme"],
        help="Método de mutación"
    )
    parser.add_argument("--pm", type=float, help="Probabilidad de mutación Pm")
    parser.add_argument("-o", "--output-dir", type=str, help="Directorio para guardar salidas")
    parser.add_argument("--plot", action="store_true", help="Generar gráficos de evolución al finalizar")
    return parser.parse_args()

def main() -> None:
    args = parse_arguments()

    # Cargar base de configuración
    config_path = args.config
    if not os.path.exists(config_path) and os.path.exists("configuracion.json"):
        config_path = "configuracion.json"

    if os.path.exists(config_path):
        cfg = Config.load_json(config_path)
    else:
        print(f"Aviso: no se encontró archivo de configuración en '{config_path}'. Usando configuración por defecto.")
        cfg = Config()

    # Sobrescribir con flags CLI
    if args.image:
        cfg.image_path = args.image
    if args.triangles is not None:
        cfg.num_triangles = args.triangles
    if args.generations is not None:
        cfg.max_generations = args.generations
    if args.timeout is not None:
        cfg.timeout_seconds = args.timeout
    if args.pop_size is not None:
        cfg.population_size = args.pop_size
    if args.offspring is not None:
        cfg.num_offspring = args.offspring
    if args.parent_sel:
        cfg.parent_selection_method = args.parent_sel
    if args.survival_sel:
        cfg.survival_selection_method = args.survival_sel
    if args.survival_strat:
        cfg.survival_strategy = args.survival_strat
    if args.crossover:
        cfg.crossover_method = args.crossover
    if args.mutation:
        cfg.mutation_method = args.mutation
    if args.pm is not None:
        cfg.mutation_pm = args.pm
    if args.output_dir:
        cfg.output_dir = args.output_dir

    # Validar tope lógico de triángulos
    validate_triangle_count(cfg.num_triangles)

    # Verificar existencia de la imagen objetivo
    if not os.path.exists(cfg.image_path):
        alt_path = os.path.join("tp2", cfg.image_path)
        if os.path.exists(alt_path):
            cfg.image_path = alt_path
        else:
            print(f"Error: La imagen objetivo no existe en '{cfg.image_path}'.", file=sys.stderr)
            sys.exit(1)

    print("=" * 70)
    print(" ITBA - SIA :: TP2 Compresor de Imágenes con Algoritmos Genéticos")
    print("=" * 70)
    print(f" • Imagen objetivo:      {cfg.image_path}")
    print(f" • Triángulos:           {cfg.num_triangles} (Tope máx permitido: {MAX_ALLOWED_TRIANGLES})")
    print(f" • Población N:          {cfg.population_size} | Hijos K: {cfg.num_offspring}")
    print(f" • Selección de padres:  {cfg.parent_selection_method}")
    print(f" • Cruce:                {cfg.crossover_method} (pc={cfg.crossover_pc})")
    print(f" • Mutación:             {cfg.mutation_method} (pm={cfg.mutation_pm})")
    print(f" • Supervivencia:        {cfg.survival_strategy} ({cfg.survival_selection_method})")
    print(f" • Criterios de parada:  Máx {cfg.max_generations} gens | Timeout: {cfg.timeout_seconds}s")
    print(f" • Directorio de salida: {cfg.output_dir}")
    print("-" * 70)

    # Instanciar componentes
    eval_res = tuple(cfg.eval_resolution) if cfg.eval_resolution else (64, 64)
    bg_col = tuple(cfg.background_color)
    evaluator = FitnessEvaluator(
        target_image_path=cfg.image_path,
        eval_size=eval_res,
        bg_color=bg_col
    )

    parent_selector = get_selection_operator(
        cfg.parent_selection_method, **cfg.selection_params
    )
    survival_selector = get_selection_operator(
        cfg.survival_selection_method, **cfg.selection_params
    )
    survival_strat = get_survival_strategy(
        cfg.survival_strategy, selector=survival_selector, gap=cfg.generational_gap
    )
    crossover_op = get_crossover_operator(
        cfg.crossover_method, pc=cfg.crossover_pc, **cfg.crossover_params
    )
    mutation_op = get_mutation_operator(
        cfg.mutation_method, pm=cfg.mutation_pm, **cfg.mutation_params
    )
    stopping = StoppingCondition(
        max_generations=cfg.max_generations,
        timeout_seconds=cfg.timeout_seconds,
        target_fitness=cfg.target_fitness,
        target_mse=cfg.target_mse,
        content_window=cfg.content_window,
        content_delta=cfg.content_delta,
        structure_window=cfg.structure_window,
        structure_threshold=cfg.structure_threshold
    )
    tracker = MetricsTracker(output_dir=cfg.output_dir)

    last_print_time = 0.0

    def progress(gen: int, fit: float, mse: float, elapsed: float) -> None:
        nonlocal last_print_time
        curr = time.time()
        if gen == 0 or gen % 10 == 0 or (curr - last_print_time) > 2.0:
            last_print_time = curr
            print(f" [Gen {gen:4d}/{cfg.max_generations}] Fitness: {fit:.6f} | MSE: {mse:7.2f} | Tiempo: {elapsed:5.1f}s")

    ga = GeneticAlgorithm(
        evaluator=evaluator,
        parent_selector=parent_selector,
        crossover_operator=crossover_op,
        mutation_operator=mutation_op,
        survival_strategy=survival_strat,
        stopping_condition=stopping,
        tracker=tracker,
        population_size=cfg.population_size,
        num_offspring=cfg.num_offspring,
        num_triangles=cfg.num_triangles,
        snapshot_interval=cfg.snapshot_interval,
        progress_callback=progress
    )

    results = ga.run()

    plot_path = None
    if args.plot:
        from tp2.src.visualization import plot_metrics
        plot_path = plot_metrics(results["metrics_csv_path"])

    print("-" * 70)
    print(" Finalización del Algoritmo Genético:")
    print(f"  • Razón de parada:        {results['stop_reason']}")
    print(f"  • Generaciones totales:   {results['generations_completed']}")
    print(f"  • Tiempo total:           {results['total_time_seconds']:.2f} s")
    print(f"  • Mejor Fitness:          {results['best_fitness']:.6f}")
    print(f"  • Mejor MSE:              {results['best_mse']:.2f}")
    print(f"  • Imagen generada:        {results['best_image_path']}")
    print(f"  • Triángulos guardados:   {results['triangles_json_path']}")
    print(f"  • Métricas CSV:           {results['metrics_csv_path']}")
    if plot_path:
        print(f"  • Gráfico de evolución:   {plot_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
