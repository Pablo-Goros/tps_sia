"""Genera gráficos desde resultados guardados, sin volver a entrenar."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from .datos_digitos import EJERCICIO


RESULTS = EJERCICIO / "results" / "baseline"


def generate(results_path: Path = RESULTS / "results.json") -> None:
    results_path = Path(results_path)
    report = json.loads(results_path.read_text(encoding="utf-8"))
    history = report["history"]
    epochs = np.arange(1, history["epocas_corridas"] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, train_key, val_key, label in (
        (axes[0], "costo", "costo_validacion", "Loss (cross-entropy)"),
        (axes[1], "accuracy", "accuracy_validacion", "Accuracy"),
    ):
        ax.plot(epochs, history[train_key], label="Train", color="#2a78d6")
        ax.plot(epochs, history[val_key], label="Validación", color="#eb6834")
        ax.set(xlabel="Época", ylabel=label, title=label)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[1].set_ylim(0, 1)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle(f"Baseline {report['config']['architecture']} · SGD")
    fig.savefig(results_path.parent / "learning_curves.png", dpi=160)
    plt.close(fig)

    matrix = np.asarray(report["validation_confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    heatmap = ax.imshow(matrix, cmap="Blues", vmin=0)
    for row in range(10):
        for col in range(10):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > matrix.max() / 2 else "black", fontsize=9)
    absent = np.flatnonzero(matrix.sum(axis=1) == 0)
    labels = [f"{d} (sin muestras)" if d in absent else str(d) for d in range(10)]
    ax.set(xticks=range(10), yticks=range(10), yticklabels=labels,
           xlabel="Dígito predicho", ylabel="Dígito real",
           title=f"Validación · accuracy {report['final']['validation_accuracy']:.2%}")
    fig.colorbar(heatmap, ax=ax, label="Cantidad de muestras")
    fig.savefig(results_path.parent / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    print(f"Gráficos: {results_path.parent.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS / "results.json")
    generate(parser.parse_args().results)


if __name__ == "__main__":
    main()
