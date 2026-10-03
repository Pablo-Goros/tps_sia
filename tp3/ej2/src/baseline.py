"""Entrena un único baseline y guarda resultados reproducibles para su análisis."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform

import numpy as np

from .datos_digitos import CACHE, DIGITS, EJERCICIO
from tps_sia.tp3.shared.digit_dataset import N_CLASES, cargar, particionar
from tps_sia.tp3.shared.mlp import MLP
from tps_sia.tp3.shared.optimizers import SGD
from tps_sia.tp3.shared.metrics import confusion_matrix
from tps_sia.tp3.shared.experiments import DEFAULTS, build_model, sha256_file, write_history


CONFIG = EJERCICIO / "baseline.json"
RESULTS = EJERCICIO / "results" / "baseline"


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
    X, y = cargar(DIGITS, CACHE)
    X_train, y_train, X_val, y_val = particionar(X, y, semilla=config["split_seed"])
    model = build_model({
        **DEFAULTS, 'architecture': config['architecture'], 'activation': config['activation'],
        'output': config['output'], 'optimizer': {'name': 'sgd', 'learning_rate': config['learning_rate']},
        'batch_size': config['batch_size'], 'initialization': config['initialization'],
        'model_seed': config['model_seed'],
    })
    print(f"Baseline {config['architecture']} / SGD; train={len(y_train)}, val={len(y_val)}", flush=True)
    history = model.entrenar(
        X_train, y_train, epocas=config["epochs"],
        X_val=X_val, y_val=y_val, verbose=1,
    )
    matrix = confusion_matrix(y_val.argmax(axis=1), model.predecir_clases(X_val))
    val_accuracy = float(matrix.trace() / matrix.sum())
    dataset_digest = sha256_file(DIGITS)
    report = {
        "schema_version": 1,
        "config": config,
        "dataset": {
            "source": "tp3/data/digits.csv", "sha256": dataset_digest,
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
    write_history(output_dir / "history.csv", history.to_dict())
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
