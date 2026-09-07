"""Módulo de reporting: genera gráficos comparativos a partir de los resultados
ya calculados por `tp2.src.experiments` (desacoplado del cómputo del AG).

Lee `ejecuciones.csv`, `resumen.csv` y las curvas por repetición
(`curvas/<variante>/rep_XX.csv`) que deja `tp2.src.experiments`, agrupa las
variantes por el campo `grupo` y genera, por cada grupo:

- `<grupo>_barras.png`: fitness promedio (± desvío entre repeticiones) de
  cada variante del grupo, ordenado de mejor a peor.
- `<grupo>_evolucion.png`: evolución del fitness promedio por generación,
  una curva por variante (promedio entre repeticiones) con una banda
  sombreada de ± 1 desvío estándar entre repeticiones.
- `<grupo>_tiempo_por_generacion.png`: segundos por generación (± desvío),
  ordenado de más rápido a más lento. Aísla el costo real de cada operador
  del tiempo total de la corrida (que también depende de cuántas
  generaciones corrió antes de cortar).
- `<grupo>_tradeoff.png`: scatter de fitness final vs. tiempo total promedio
  por corrida, un punto por variante del grupo (sin barras de error, sólo el
  promedio), para discutir si el fitness extra de un método justifica su
  costo en tiempo dentro de esa misma categoría de operador.

Uso:
    python -m tp2.src.reporting --experiments-dir tp2/resultados/experimentos
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_variants(rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Devuelve {grupo: [variantes...]} preservando el orden de aparición."""
    groups: Dict[str, List[str]] = {}
    for row in rows:
        group = row.get("grupo") or row["variante"]
        bucket = groups.setdefault(group, [])
        if row["variante"] not in bucket:
            bucket.append(row["variante"])
    return groups


def sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _bar_chart(
    labels: List[str],
    values: List[float],
    errors: List[float],
    ylabel: str,
    title: str,
    output_path: str,
    value_fmt: str = "{:.4f}",
    clip_to_unit: bool = False,
) -> None:
    """Barra genérica con barras de error, ordenada de mejor a peor (ya viene
    ordenada por el llamador) y con el valor numérico sobre cada barra."""
    colors = ["#2ca02c"] + ["#1f77b4"] * (len(labels) - 1)

    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(labels)), 5.5))
    bars = ax.bar(labels, values, yerr=errors, capsize=4, color=colors)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            value_fmt.format(val),
            ha="center", va="bottom", fontsize=8,
        )

    lower = min(v - e for v, e in zip(values, errors))
    upper = max(v + e for v, e in zip(values, errors))
    margin = max((upper - lower) * 0.3, upper * 0.02, 1e-9)
    ymin = lower - margin
    ymax = upper + margin
    if clip_to_unit:
        ymin = max(0.0, ymin)
        ymax = min(1.0, ymax) + 0.002
    else:
        ymin = max(0.0, ymin)
    ax.set_ylim(ymin, ymax)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_bar_group(
    group_name: str,
    variant_names: List[str],
    summary_by_name: Dict[str, Dict[str, Any]],
    output_path: str,
) -> bool:
    """Gráfico de barras del fitness promedio (± desvío) de un grupo, ordenado
    de mejor (mayor fitness) a peor."""
    entries = []
    for name in variant_names:
        row = summary_by_name.get(name)
        if row is None:
            continue
        entries.append((name, float(row["fitness_promedio"]), float(row["fitness_desvio"])))

    if not entries:
        return False

    entries.sort(key=lambda e: e[1], reverse=True)
    labels = [e[0] for e in entries]
    fits = [e[1] for e in entries]
    errs = [e[2] for e in entries]

    _bar_chart(
        labels, fits, errs,
        ylabel="Fitness promedio (± desvío entre repeticiones)",
        title=f"Comparación de fitness final — {group_name}",
        output_path=output_path,
        value_fmt="{:.4f}",
        clip_to_unit=True,
    )
    return True


def plot_time_bar_group(
    group_name: str,
    variant_names: List[str],
    summary_by_name: Dict[str, Dict[str, Any]],
    output_path: str,
) -> bool:
    """Gráfico de barras del tiempo por generación (± desvío) de un grupo,
    ordenado de más rápido a más lento. Aísla el costo real del operador del
    tiempo total (que también depende de cuántas generaciones corrió)."""
    entries = []
    for name in variant_names:
        row = summary_by_name.get(name)
        if row is None or "tiempo_por_generacion_promedio" not in row:
            continue
        entries.append((
            name,
            float(row["tiempo_por_generacion_promedio"]),
            float(row["tiempo_por_generacion_desvio"]),
        ))

    if not entries:
        return False

    entries.sort(key=lambda e: e[1])  # más rápido primero
    labels = [e[0] for e in entries]
    times = [e[1] for e in entries]
    errs = [e[2] for e in entries]

    _bar_chart(
        labels, times, errs,
        ylabel="Segundos por generación (± desvío entre repeticiones)",
        title=f"Costo por generación — {group_name}",
        output_path=output_path,
        value_fmt="{:.4f}",
        clip_to_unit=False,
    )
    return True


def load_variant_rep_curves(curves_dir: str, variant_name: str) -> List[List[Dict[str, Any]]]:
    """Carga todas las curvas por repetición de una variante."""
    variant_dir = os.path.join(curves_dir, variant_name)
    curves: List[List[Dict[str, Any]]] = []
    if not os.path.isdir(variant_dir):
        return curves
    for fname in sorted(os.listdir(variant_dir)):
        if fname.endswith(".csv"):
            curves.append(load_csv(os.path.join(variant_dir, fname)))
    return curves


def aggregate_curve(
    curves: List[List[Dict[str, Any]]], field: str = "promedio_fitness"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Media y desvío estándar de `field` entre repeticiones, generación a
    generación. Si las corridas terminaron en distinta cantidad de
    generaciones, trunca a la más corta común."""
    if not curves:
        return np.array([]), np.array([]), np.array([])

    min_len = min(len(c) for c in curves)
    if min_len == 0:
        return np.array([]), np.array([]), np.array([])

    gens = np.array([int(curves[0][i]["generacion"]) for i in range(min_len)])
    values = np.array([[float(c[i][field]) for i in range(min_len)] for c in curves])
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    return gens, means, stds


def plot_line_group(
    group_name: str,
    variant_names: List[str],
    curves_dir: str,
    output_path: str,
) -> bool:
    """Curva de evolución del fitness promedio (media entre repeticiones) con
    banda sombreada de ± 1 desvío, una serie por variante del grupo."""
    fig, ax = plt.subplots(figsize=(9, 6))
    any_data = False

    for name in variant_names:
        curves = load_variant_rep_curves(curves_dir, name)
        if not curves:
            print(f"  Aviso: sin curvas por repetición para '{name}' (grupo '{group_name}'), se omite.", file=sys.stderr)
            continue

        gens, means, stds = aggregate_curve(curves, field="promedio_fitness")
        if gens.size == 0:
            continue

        any_data = True
        line, = ax.plot(gens, means, label=f"{name} (n={len(curves)})", linewidth=1.8)
        ax.fill_between(gens, means - stds, means + stds, color=line.get_color(), alpha=0.15)

    if not any_data:
        plt.close(fig)
        return False

    ax.set_xlabel("Generación")
    ax.set_ylabel("Fitness promedio (media entre repeticiones ± desvío)")
    ax.set_title(f"Evolución del fitness promedio — {group_name}")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return True


GROUP_COLORS = {
    "seleccion_padres": "#1f77b4",
    "cruce": "#ff7f0e",
    "mutacion": "#2ca02c",
    "supervivencia": "#d62728",
}
_FALLBACK_COLORS = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def _color_for_group(group_name: str, seen_order: List[str]) -> str:
    if group_name in GROUP_COLORS:
        return GROUP_COLORS[group_name]
    idx = seen_order.index(group_name) % len(_FALLBACK_COLORS)
    return _FALLBACK_COLORS[idx]


def plot_tradeoff_group(
    group_name: str,
    variant_names: List[str],
    summary_by_name: Dict[str, Dict[str, Any]],
    output_path: str,
    seen_groups: List[str],
) -> bool:
    """Scatter de fitness final vs. tiempo total de corrida, un punto por
    variante del grupo, sin barras de error (sólo la posición del promedio)."""
    xs, ys, labels = [], [], []
    for name in variant_names:
        row = summary_by_name.get(name)
        if row is None:
            continue
        xs.append(float(row["tiempo_promedio_s"]))
        ys.append(float(row["fitness_promedio"]))
        labels.append(name)

    if not xs:
        return False

    color = _color_for_group(group_name, seen_groups)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xs, ys, s=70, color=color, alpha=0.85, edgecolors="white", linewidths=0.8)
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(
            label, (x, y), textcoords="offset points", xytext=(7, 5),
            fontsize=8.5, color=color,
        )

    ax.set_xlabel("Tiempo total promedio por corrida (s)")
    ax.set_ylabel("Fitness promedio final")
    ax.set_title(f"Trade-off fitness vs. tiempo — {group_name}")
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return True


def generate_reports(experiments_dir: str, output_dir: Optional[str] = None) -> List[str]:
    summary_path = os.path.join(experiments_dir, "resumen.csv")
    runs_path = os.path.join(experiments_dir, "ejecuciones.csv")
    curves_dir = os.path.join(experiments_dir, "curvas")

    if not os.path.exists(summary_path) or not os.path.exists(runs_path):
        raise FileNotFoundError(
            f"No se encontraron 'ejecuciones.csv'/'resumen.csv' en '{experiments_dir}'. "
            f"Corré primero 'python -m tp2.src.experiments'."
        )

    summary_rows = load_csv(summary_path)
    run_rows = load_csv(runs_path)
    summary_by_name = {r["variante"]: r for r in summary_rows}

    groups = group_variants(run_rows)

    final_output_dir = output_dir or os.path.join(experiments_dir, "graficos")
    os.makedirs(final_output_dir, exist_ok=True)

    print(f"Grupos detectados: {list(groups.keys())}")

    group_names = list(groups.keys())
    generated: List[str] = []
    for group_name, variant_names in groups.items():
        safe_name = sanitize_filename(group_name)

        bar_path = os.path.join(final_output_dir, f"{safe_name}_barras.png")
        if plot_bar_group(group_name, variant_names, summary_by_name, bar_path):
            generated.append(bar_path)

        line_path = os.path.join(final_output_dir, f"{safe_name}_evolucion.png")
        if plot_line_group(group_name, variant_names, curves_dir, line_path):
            generated.append(line_path)

        time_bar_path = os.path.join(final_output_dir, f"{safe_name}_tiempo_por_generacion.png")
        if plot_time_bar_group(group_name, variant_names, summary_by_name, time_bar_path):
            generated.append(time_bar_path)

        tradeoff_path = os.path.join(final_output_dir, f"{safe_name}_tradeoff.png")
        if plot_tradeoff_group(group_name, variant_names, summary_by_name, tradeoff_path, group_names):
            generated.append(tradeoff_path)

    return generated


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera gráficos comparativos a partir de los resultados de tp2.src.experiments"
    )
    parser.add_argument(
        "--experiments-dir", "-d", type=str, default="tp2/resultados/experimentos",
        help="Directorio con ejecuciones.csv, resumen.csv y curvas/ generado por tp2.src.experiments"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=None,
        help="Directorio de salida para los gráficos (por defecto: <experiments-dir>/graficos)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    try:
        generated = generate_reports(args.experiments_dir, args.output_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("-" * 70)
    print(f"Gráficos generados ({len(generated)}):")
    for path in generated:
        print(f"  - {path}")
    print("-" * 70)


if __name__ == "__main__":
    main()
