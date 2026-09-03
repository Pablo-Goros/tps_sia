"""Generación de gráficos para análisis experimental de los resultados del AG."""
from __future__ import annotations
import csv
import os
from typing import Optional
import matplotlib.pyplot as plt

def plot_metrics(csv_path: str, output_image_path: Optional[str] = None) -> str:
    """Lee el archivo metricas.csv y genera un gráfico con curvas de evolución."""
    generations = []
    best_fits = []
    avg_fits = []
    worst_fits = []
    mses = []
    diversities = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            generations.append(int(row["generacion"]))
            best_fits.append(float(row["mejor_fitness"]))
            avg_fits.append(float(row["promedio_fitness"]))
            worst_fits.append(float(row["peor_fitness"]))
            mses.append(float(row["mejor_mse"]))
            diversities.append(float(row["diversidad"]))

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    # 1. Curva de Fitness
    axes[0].plot(generations, best_fits, label="Mejor Fitness", color="#1f77b4", linewidth=2)
    axes[0].plot(generations, avg_fits, label="Fitness Promedio", color="#ff7f0e", linestyle="--")
    axes[0].plot(generations, worst_fits, label="Peor Fitness", color="#d62728", alpha=0.5)
    axes[0].set_ylabel("Fitness (0 a 1)")
    axes[0].set_title("Evolución de Fitness por Generación")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower right")

    # 2. Curva de Error MSE
    axes[1].plot(generations, mses, label="Mejor MSE", color="#2ca02c", linewidth=2)
    axes[1].set_ylabel("MSE (RGB)")
    axes[1].set_title("Reducción de Error Cuadrático Medio")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right")

    # 3. Curva de Diversidad Genética
    axes[2].plot(generations, diversities, label="Diversidad (Varianza Fitness)", color="#9467bd", linewidth=1.5)
    axes[2].set_xlabel("Generación")
    axes[2].set_ylabel("Diversidad")
    axes[2].set_title("Diversidad de la Población")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(loc="upper right")

    plt.tight_layout()

    if output_image_path is None:
        base_dir = os.path.dirname(csv_path)
        output_image_path = os.path.join(base_dir, "graficos_evolucion.png")

    fig.savefig(output_image_path, dpi=150)
    plt.close(fig)
    return output_image_path
