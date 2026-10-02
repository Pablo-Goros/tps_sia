"""Entrena un único baseline y guarda resultados reproducibles para su análisis."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from .datos_digitos import DIGITS, EJERCICIO, N_CLASES, cargar, particionar
from .mlp import MLP
from .optimizadores import SGD


CONFIG = EJERCICIO / "baseline.json"
RESULTS = EJERCICIO / "results" / "baseline"


def confusion_matrix(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Filas = dígito real; columnas = predicción; conserva las diez clases."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if (actual.ndim != 1 or actual.shape != predicted.shape or actual.size == 0
            or actual.dtype.kind not in "iu" or predicted.dtype.kind not in "iu"
            or np.any((actual < 0) | (actual >= N_CLASES))
            or np.any((predicted < 0) | (predicted >= N_CLASES))):
        raise ValueError("Se requieren etiquetas enteras entre 0 y 9 de igual longitud.")
    matrix = np.zeros((N_CLASES, N_CLASES), dtype=np.int64)
    np.add.at(matrix, (actual, predicted), 1)
    return matrix


def run(config_path: Path = CONFIG, output_dir: Path = RESULTS) -> dict:
    config_path, output_dir = Path(config_path), Path(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["optimizer"] != "sgd":
        raise ValueError("El baseline usa SGD.")
    if config["architecture"][0] != 784 or config["architecture"][-1] != N_CLASES:
        raise ValueError("El modelo de dígitos requiere 784 entradas y diez salidas.")
    if config["output"] != "softmax":
        raise ValueError("El baseline de clasificación usa softmax y cross-entropy.")
    if not 0 <= config["sanity_accuracy"] <= 1:
        raise ValueError("sanity_accuracy debe estar entre 0 y 1.")
    X, y = cargar()
    X_train, y_train, X_val, y_val = particionar(X, y, semilla=config["split_seed"])
    model = MLP(
        config["architecture"], activacion=config["activation"],
        salida=config["output"], optimizador=SGD(config["learning_rate"]),
        tamano_lote=config["batch_size"], inicializacion=config["initialization"],
        semilla=config["model_seed"],
    )
    print(f"Baseline {config['architecture']} / SGD; train={len(y_train)}, val={len(y_val)}", flush=True)
    history = model.entrenar(
        X_train, y_train, epocas=config["epochs"],
        X_val=X_val, y_val=y_val, verbose=1,
    )
    matrix = confusion_matrix(y_val.argmax(axis=1), model.predecir_clases(X_val))
    val_accuracy = float(matrix.trace() / matrix.sum())
    digest = hashlib.sha256()
    with DIGITS.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    report = {
        "schema_version": 1,
        "config": config,
        "dataset": {
            "source": "tp3/data/digits.csv", "sha256": digest.hexdigest(),
            "preprocessing": "Original pixels, no rescaling",
            "split": "Stratified 80/20, rounded per class",
            "train_samples": len(y_train), "validation_samples": len(y_val),
            "train_class_counts": y_train.sum(axis=0).astype(int).tolist(),
            "validation_class_counts": y_val.sum(axis=0).astype(int).tolist(),
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "history": history.to_dict(),
        "final": {
            "train_loss": history.costo[-1], "train_accuracy": history.accuracy[-1],
            "validation_loss": history.costo_validacion[-1],
            "validation_accuracy": val_accuracy,
            "sanity_passed": val_accuracy >= config["sanity_accuracy"],
        },
        "validation_confusion_matrix": matrix.tolist(),
        "confusion_axes": {"rows": "actual", "columns": "predicted"},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model.guardar(output_dir / "model.npz")
    (output_dir / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
    )
    with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "validation_loss", "train_accuracy", "validation_accuracy"])
        writer.writerows(zip(range(1, history.epocas_corridas + 1), history.costo,
                              history.costo_validacion, history.accuracy, history.accuracy_validacion))
    with (output_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *range(N_CLASES)])
        writer.writerows((digit, *row) for digit, row in enumerate(matrix))
    status = "OK" if report["final"]["sanity_passed"] else "REVISAR"
    print(f"Train: {history.accuracy[-1]:.2%}; val: {val_accuracy:.2%}; "
          f"sanity check >= {config['sanity_accuracy']:.0%}: {status}")
    print(f"Resultados: {output_dir.resolve()}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    run(args.config, args.output_dir)
    from .plots import generate
    generate(args.output_dir / "results.json")


if __name__ == "__main__":
    main()
