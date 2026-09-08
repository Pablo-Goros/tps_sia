"""Barrido experimental (grid search) sobre los operadores del AG del TP2.

Corre cada variante de configuración N veces (no hay semilla en el motor,
por lo que cada repetición ya difiere) y agrega los resultados en dos CSV,
al estilo de `resultados/ejecuciones.csv` y `resultados/resumen.csv` del TP1:

- ejecuciones.csv: una fila por corrida individual.
- resumen.csv: promedio y desvío estándar por variante.

Además guarda, para la mejor repetición de cada variante, la imagen final
y la curva de métricas completa (para graficar evolución de fitness/MSE).

Uso:
    python -m tp2.src.experiments --experiments-config tp2/experimentos.json
    python -m tp2.src.experiments --experiments-config tp2/experimentos.json --quick
    python -m tp2.src.experiments --experiments-config tp2/experimentos.json --list
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

from tp2.src.config import Config
from tp2.src.engine.fitness import FitnessEvaluator
from tp2.src.engine.genetic_algorithm import GeneticAlgorithm
from tp2.src.operators.selection import get_selection_operator
from tp2.src.operators.crossover import get_crossover_operator
from tp2.src.operators.mutation import get_mutation_operator
from tp2.src.operators.survival import get_survival_strategy
from tp2.src.stopping.stopping import StoppingCondition, validate_triangle_count
from tp2.src.metrics.tracker import MetricsTracker

# Columnas de configuración que se registran en ejecuciones.csv/resumen.csv
# para poder identificar qué se barrió en cada variante.
CONFIG_COLUMNS = [
    "num_triangles",
    "population_size",
    "num_offspring",
    "parent_selection_method",
    "survival_selection_method",
    "survival_strategy",
    "generational_gap",
    "crossover_method",
    "crossover_pc",
    "mutation_method",
    "mutation_pm",
]


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Combina dos diccionarios anidados; las claves de `override` tienen prioridad."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_experiments_spec(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_variant_config(
    base_dict: Dict[str, Any],
    common_overrides: Dict[str, Any],
    variant_overrides: Dict[str, Any],
) -> Config:
    merged = deep_merge(base_dict, common_overrides)
    merged = deep_merge(merged, variant_overrides)
    return Config.from_dict(merged)


def build_ga(cfg: Config, run_output_dir: str, image_path: str) -> GeneticAlgorithm:
    validate_triangle_count(cfg.num_triangles)

    eval_res = tuple(cfg.eval_resolution) if cfg.eval_resolution else (64, 64)
    bg_col = tuple(cfg.background_color)
    evaluator = FitnessEvaluator(target_image_path=image_path, eval_size=eval_res, bg_color=bg_col)

    parent_selector = get_selection_operator(cfg.parent_selection_method, **cfg.selection_params)
    survival_selector = get_selection_operator(cfg.survival_selection_method, **cfg.selection_params)
    survival_strat = get_survival_strategy(
        cfg.survival_strategy, selector=survival_selector, gap=cfg.generational_gap
    )
    crossover_op = get_crossover_operator(cfg.crossover_method, pc=cfg.crossover_pc, **cfg.crossover_params)
    mutation_op = get_mutation_operator(cfg.mutation_method, pm=cfg.mutation_pm, **cfg.mutation_params)
    stopping = StoppingCondition(
        max_generations=cfg.max_generations,
        timeout_seconds=cfg.timeout_seconds,
        target_fitness=cfg.target_fitness,
        target_mse=cfg.target_mse,
        content_window=cfg.content_window,
        content_delta=cfg.content_delta,
        structure_window=cfg.structure_window,
        structure_threshold=cfg.structure_threshold,
    )
    tracker = MetricsTracker(output_dir=run_output_dir)

    return GeneticAlgorithm(
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
        snapshot_interval=0,  # sin snapshots intermedios durante el barrido
    )


def run_single(cfg: Config, image_path: str, scratch_root: str, run_id: str) -> Dict[str, Any]:
    """Corre una única repetición y devuelve sus resultados; limpia sus archivos temporales."""
    run_dir = os.path.join(scratch_root, run_id)
    ga = build_ga(cfg, run_dir, image_path)
    results = ga.run()

    with open(results["metrics_csv_path"], "r", encoding="utf-8") as f:
        curve_rows = list(csv.DictReader(f))

    image_data: Optional[bytes] = None
    if os.path.exists(results["best_image_path"]):
        with open(results["best_image_path"], "rb") as img_f:
            image_data = img_f.read()

    shutil.rmtree(run_dir, ignore_errors=True)

    results["curve"] = curve_rows
    results["image_data"] = image_data
    return results


def config_row(cfg: Config) -> Dict[str, Any]:
    return {col: getattr(cfg, col) for col in CONFIG_COLUMNS}


def run_grid(
    spec_path: str,
    repetitions_override: Optional[int],
    quick: bool,
    only_variants: Optional[List[str]],
    image_override: Optional[str],
    output_dir_override: Optional[str],
    list_only: bool,
) -> None:
    spec = load_experiments_spec(spec_path)
    spec_dir = os.path.dirname(os.path.abspath(spec_path))

    base_config_path = spec.get("base_config", "tp2/configuracion.json")
    if not os.path.isabs(base_config_path):
        base_config_path = os.path.join(spec_dir, "..", base_config_path) if not os.path.exists(
            base_config_path
        ) else base_config_path
    if not os.path.exists(base_config_path):
        base_config_path = spec["base_config"]
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_dict = json.load(f)

    common_overrides = spec.get("overrides_common", {})
    variants = spec.get("variants", [])
    if only_variants:
        variants = [v for v in variants if v["name"] in only_variants]
        if not variants:
            print(f"Ningún variant coincide con {only_variants}", file=sys.stderr)
            sys.exit(2)

    if quick:
        common_overrides = deep_merge(
            common_overrides,
            {"stopping": {"max_generations": 15, "timeout_seconds": 20}},
        )

    repetitions = repetitions_override if repetitions_override is not None else int(spec.get("repetitions", 5))
    output_dir = output_dir_override or spec.get("output_dir", "tp2/resultados/experimentos")
    image_path = image_override or base_dict.get("image_path")

    if list_only:
        print(f"Config base:     {base_config_path}")
        print(f"Imagen objetivo: {image_path}")
        print(f"Repeticiones:    {repetitions}")
        print(f"Variantes ({len(variants)}):")
        for v in variants:
            print(f"  - {v['name']}: {json.dumps({k: val for k, val in v.items() if k != 'name'}, ensure_ascii=False)}")
        return

    if not os.path.exists(image_path):
        print(f"Error: no se encontró la imagen objetivo '{image_path}'.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "imagenes")
    curves_dir = os.path.join(output_dir, "curvas")
    scratch_root = os.path.join(output_dir, "_scratch")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(curves_dir, exist_ok=True)
    os.makedirs(scratch_root, exist_ok=True)

    runs_csv_path = os.path.join(output_dir, "ejecuciones.csv")
    summary_csv_path = os.path.join(output_dir, "resumen.csv")

    all_rows: List[Dict[str, Any]] = []
    total_runs = len(variants) * repetitions
    run_counter = 0
    t_grid_start = time.time()

    print("=" * 70)
    print(f" Barrido experimental TP2 :: {len(variants)} variantes x {repetitions} repeticiones = {total_runs} corridas")
    print(f" Imagen objetivo: {image_path}")
    print(f" Salida:          {output_dir}")
    print("=" * 70)

    for variant in variants:
        name = variant["name"]
        group = variant.get("group", name)
        overrides = {k: v for k, v in variant.items() if k not in ("name", "group")}
        cfg = resolve_variant_config(base_dict, common_overrides, overrides)
        row_base = {"variante": name, "grupo": group, **config_row(cfg)}

        variant_curves_dir = os.path.join(curves_dir, name)
        os.makedirs(variant_curves_dir, exist_ok=True)

        best_fitness_for_variant = -1.0
        best_image_data: Optional[bytes] = None

        for rep in range(repetitions):
            run_counter += 1
            t0 = time.time()
            run_id = f"{name}__rep{rep}"
            result = run_single(cfg, image_path, scratch_root, run_id)
            elapsed = time.time() - t0

            row = dict(row_base)
            row.update({
                "repeticion": rep,
                "stop_reason": result["stop_reason"],
                "generaciones": result["generations_completed"],
                "tiempo_total_s": round(result["total_time_seconds"], 3),
                "mejor_fitness": round(result["best_fitness"], 6),
                "mejor_mse": round(result["best_mse"], 4),
            })
            all_rows.append(row)

            # Guardamos la curva completa de ESTA repetición (una por rep, no sólo
            # la mejor) para que el módulo de reporting pueda promediar entre
            # repeticiones y graficar el error por generación.
            curve_rows = result["curve"]
            if curve_rows:
                rep_curve_path = os.path.join(variant_curves_dir, f"rep_{rep:02d}.csv")
                with open(rep_curve_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(curve_rows)

            if result["best_fitness"] > best_fitness_for_variant:
                best_fitness_for_variant = result["best_fitness"]
                best_image_data = result["image_data"]

            print(
                f" [{run_counter:4d}/{total_runs}] {name:28s} rep {rep+1}/{repetitions} "
                f"-> fitness={result['best_fitness']:.6f} mse={result['best_mse']:7.2f} "
                f"gens={result['generations_completed']:4d} t={elapsed:5.1f}s"
            )

        # Persistimos sólo la imagen de la mejor repetición (las curvas de todas
        # las repeticiones ya se guardaron arriba, una por archivo).
        if best_image_data is not None:
            img_path = os.path.join(images_dir, f"{name}.png")
            with open(img_path, "wb") as f:
                f.write(best_image_data)

    shutil.rmtree(scratch_root, ignore_errors=True)

    # --- ejecuciones.csv: una fila por corrida ---
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with open(runs_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # --- resumen.csv: agregación por variante (media y desvío estándar) ---
    by_variant: Dict[str, List[Dict[str, Any]]] = {}
    for row in all_rows:
        by_variant.setdefault(row["variante"], []).append(row)

    summary_rows = []
    for variant in variants:
        name = variant["name"]
        group = variant.get("group", name)
        rows = by_variant.get(name, [])
        if not rows:
            continue
        fits = [r["mejor_fitness"] for r in rows]
        mses = [r["mejor_mse"] for r in rows]
        gens = [r["generaciones"] for r in rows]
        times = [r["tiempo_total_s"] for r in rows]
        # Tiempo por generación calculado por corrida (no como cociente de
        # promedios) para poder reportar también su desvío entre repeticiones.
        times_per_gen = [r["tiempo_total_s"] / r["generaciones"] for r in rows if r["generaciones"] > 0]
        overrides = {k: v for k, v in variant.items() if k not in ("name", "group")}

        summary_rows.append({
            "variante": name,
            "grupo": group,
            "descripcion": json.dumps(overrides, ensure_ascii=False),
            "repeticiones": len(rows),
            "fitness_promedio": round(statistics.mean(fits), 6),
            "fitness_desvio": round(statistics.pstdev(fits), 6) if len(fits) > 1 else 0.0,
            "fitness_max": round(max(fits), 6),
            "mse_promedio": round(statistics.mean(mses), 4),
            "mse_desvio": round(statistics.pstdev(mses), 4) if len(mses) > 1 else 0.0,
            "generaciones_promedio": round(statistics.mean(gens), 2),
            "tiempo_por_generacion_promedio": round(statistics.mean(times_per_gen), 6) if times_per_gen else 0.0,
            "tiempo_por_generacion_desvio": round(statistics.pstdev(times_per_gen), 6) if len(times_per_gen) > 1 else 0.0,
            "tiempo_promedio_s": round(statistics.mean(times), 3),
            "tiempo_desvio_s": round(statistics.pstdev(times), 3) if len(times) > 1 else 0.0,
        })

    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        writer.writeheader()
        writer.writerows(summary_rows)

    total_elapsed = time.time() - t_grid_start
    print("-" * 70)
    print(f" Barrido finalizado en {total_elapsed/60:.1f} min ({total_elapsed:.1f}s)")
    print(f"  • Corridas individuales: {runs_csv_path}")
    print(f"  • Resumen por variante:  {summary_csv_path}")
    print(f"  • Imágenes (mejor rep):  {images_dir}")
    print(f"  • Curvas (mejor rep):    {curves_dir}")
    print("=" * 70)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Barrido experimental sobre los operadores del AG del TP2"
    )
    parser.add_argument(
        "--experiments-config", "-e", type=str, default="tp2/experimentos.json",
        help="Ruta al JSON que define la grilla de variantes a correr"
    )
    parser.add_argument(
        "--repetitions", "-r", type=int, default=None,
        help="Repeticiones por variante (sobreescribe el valor del JSON de experimentos)"
    )
    parser.add_argument(
        "--variants", nargs="+", default=None,
        help="Subconjunto de nombres de variantes a correr (por defecto, todas)"
    )
    parser.add_argument(
        "--image", "-i", type=str, default=None,
        help="Imagen objetivo (sobreescribe la de la config base para todas las variantes)"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=None,
        help="Directorio de salida (sobreescribe el del JSON de experimentos)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Corrida rápida de humo: fuerza pocas generaciones y timeout corto en todas las variantes"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Sólo lista las variantes resueltas sin ejecutar nada"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    run_grid(
        spec_path=args.experiments_config,
        repetitions_override=args.repetitions,
        quick=args.quick,
        only_variants=args.variants,
        image_override=args.image,
        output_dir_override=args.output_dir,
        list_only=args.list,
    )


if __name__ == "__main__":
    main()
