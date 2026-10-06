"""Generate evidence charts used in the ej3 presentation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def generate(report_path: Path, output_dir: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    counts = {
        key: [report[key]["classes"]["counts"][str(digit)] for digit in range(10)]
        for key in ("digits", "more_digits")
    }
    for key, values in counts.items():
        if sum(values) != report[key]["rows"]:
            raise ValueError(f"Class counts do not match dataset size: {key}")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 17,
                         "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(16, 8.5), facecolor="white")
    fig.subplots_adjust(left=0.095, right=0.975, bottom=0.18, top=0.73)
    fig.text(0.095, 0.93, "Ejemplos por dígito en ambos datasets",
             fontsize=29, fontweight="bold", color="#172635")
    fig.text(0.095, 0.865, "El nuevo conjunto incorpora el 8 y duplica los ejemplos del 5",
             fontsize=20, color="#4B5967")

    for digit in (5, 8):
        ax.axvspan(digit - 0.47, digit + 0.47, color="#FFF0CB", zorder=0)
    for key, offset, color, label in (
        ("digits", -0.2, "#526F89", "digits.csv · 12.449 imágenes"),
        ("more_digits", 0.2, "#159D95", "more_digits.csv · 15.741 imágenes"),
    ):
        bars = ax.bar([digit + offset for digit in range(10)], counts[key],
                      width=0.36, color=color, label=label, zorder=3)
        for digit, bar in enumerate(bars):
            if digit in (5, 8):
                bar.set_edgecolor("#9F6900")
                bar.set_linewidth(2)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 37,
                    f"{counts[key][digit]:,}".replace(",", "."),
                    ha="center", va="bottom", fontsize=14,
                    fontweight="bold" if digit in (5, 8) else "normal",
                    color="#172635")

    ax.set_xticks(range(10))
    ax.set_xlabel("Dígito", labelpad=13)
    ax.set_ylabel("Cantidad de imágenes", labelpad=12)
    ax.set_ylim(0, 2350)
    ax.set_xlim(-0.65, 9.65)
    ax.set_yticks(range(0, 2001, 500))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}".replace(",", ".")))
    ax.grid(axis="y", color="#DFE5EA", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#BCC6CF")
    for tick in ax.get_xticklabels():
        if tick.get_text() in ("5", "8"):
            tick.set_fontweight("bold")
            tick.set_color("#9F6900")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.025), ncol=2,
              frameon=False, fontsize=17, borderaxespad=0)
    fig.text(0.095, 0.045,
             "Conteos de los archivos completos, antes de la partición train/validación. Fuente: data_report.json",
             fontsize=12, color="#637180")
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"dataset_class_counts.{extension}", dpi=180,
                    facecolor="white")
    plt.close(fig)
    with (output_dir / "dataset_class_counts.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["digit", "digits", "more_digits"])
        writer.writerows((digit, counts["digits"][digit], counts["more_digits"][digit])
                         for digit in range(10))


def generate_controls(stage_path: Path, output_dir: Path) -> None:
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    rows = {row["label"]: row for row in stage["runs"]}
    selected = [rows["baseline"], rows["ej2-reference"]]
    if any(row["status"] != "completed" for row in selected):
        raise ValueError("Both controls must be completed.")
    values = [100 * row["accuracy"] for row in selected]
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 18,
                         "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(16, 8.5), facecolor="white")
    fig.subplots_adjust(left=0.255, right=0.945, bottom=0.26, top=0.71)
    fig.text(0.08, 0.93, "Punto de partida con los nuevos datos",
             fontsize=29, fontweight="bold", color="#172635")
    fig.text(0.08, 0.865, "Controles entrenados desde cero sobre more_digits.csv",
             fontsize=20, color="#4B5967")
    fig.text(0.08, 0.79, "12.594 imágenes de entrenamiento · 3.147 de validación · Semilla 42",
             fontsize=17, color="#4B5967")
    ax.barh([1, 0], values, height=0.42, color=["#526F89", "#159D95"], zorder=3)
    ax.set_yticks([1, 0], ["Baseline", "Configuración\nseleccionada en ej2"])
    for position, value in zip([1, 0], values):
        ax.text(value - 1.4, position, f"{value:.2f} %".replace(".", ","),
                ha="right", va="center", color="white", fontsize=25, fontweight="bold")
    ax.axvline(98, color="#9F6900", linewidth=2, linestyle=(0, (5, 4)), zorder=4)
    ax.text(98, 1.48, "Objetivo: 98 %", ha="right", va="bottom", fontsize=17, color="#9F6900")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, 1.7)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Accuracy de validación (%)", labelpad=14)
    ax.grid(axis="x", color="#DFE5EA", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=12)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#BCC6CF")
    difference = f"{values[1] - values[0]:.2f}".replace(".", ",")
    fig.text(0.08, 0.125, f"La configuración de ej2 obtiene {difference} puntos porcentuales más de accuracy",
             fontsize=21, fontweight="bold", color="#172635")
    fig.text(0.08, 0.055, "Resultados iniciales de validación. El cumplimiento del objetivo se evaluará sobre test al finalizar.",
             fontsize=13, color="#637180")
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"controls_accuracy.{extension}", dpi=180, facecolor="white")
    plt.close(fig)
    with (output_dir / "controls_accuracy.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label", "validation_accuracy", "validation_loss", "chosen_epoch", "run_id"])
        writer.writerows((row["label"], row["accuracy"], row["loss"], row["chosen_epoch"], row["run_id"])
                         for row in selected)


def generate_search(search_dir: Path, output_dir: Path) -> None:
    """Plot completed exploratory stages, checking the underlying run reports."""
    stages = []
    for name in ("rates", "architectures", "batches"):
        stage = json.loads((search_dir / f"stage_{name}.json").read_text(encoding="utf-8"))
        for row in stage["runs"]:
            report = json.loads((search_dir / "runs" / row["run_id"] / "results.json").read_text(encoding="utf-8"))
            if row["status"] != "completed" or report["status"] != "completed":
                raise ValueError(f"Incomplete run: {row['run_id']}")
            if (row["accuracy"] != report["metrics"]["validation"]["accuracy"]
                    or row["chosen_epoch"] != report["chosen_epoch"]
                    or row["seed"] != 42
                    or report["metadata"] != stage["identity"]):
                raise ValueError(f"Run evidence differs: {row['run_id']}")
        if stages and stage["identity"] != stages[0]["identity"]:
            raise ValueError("Stages belong to different protocols.")
        stages.append(stage)

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 18,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), facecolor="white")
    fig.subplots_adjust(left=0.075, right=0.965, bottom=0.22, top=0.73, wspace=0.75)
    fig.text(0.055, 0.94, "Búsqueda de mejoras", fontsize=30,
             fontweight="bold", color="#172635")
    fig.text(0.055, 0.86, "Accuracy de validación con semilla 42", fontsize=22, color="#4B5967")
    titles = ("Tasa de aprendizaje", "Arquitectura", "Tamaño de lote")
    settings = ("256 neuronas · lote 32", "Tasa 0,1 · lote 32", "128 neuronas · tasa 0,1")
    for ax, stage, title, setting in zip(axes, stages, titles, settings):
        rows = stage["runs"]
        if stage["stage"] == "rates":
            labels = [f"{r['learning_rate']:g}".replace(".", ",") for r in rows]
        elif stage["stage"] == "architectures":
            labels = ["–".join(map(str, r["architecture"])) for r in rows]
        else:
            labels = [str(r["batch_size"]) for r in rows]
        for index, row in enumerate(rows):
            value = row["accuracy"] * 100
            color = "#159D95" if row["run_id"] == stage["winner"]["run_id"] else "#526F89"
            ax.scatter(value, index, color=color, s=120, zorder=3)
            ax.text(value - 0.45, index - 0.19, f"{value:.2f} %".replace(".", ","),
                    ha="right", va="bottom", fontsize=18, color=color, fontweight="bold")
        ax.set_yticks(range(len(rows)), labels)
        ax.set_ylim(len(rows) - 0.5, -0.7)
        ax.set_xlim(80, 100)
        ax.set_xticks([80, 90, 98])
        ax.set_xlabel("Accuracy (%)", labelpad=13)
        ax.set_title(f"{title}\n{setting}", fontsize=19, pad=20, color="#172635")
        ax.axvline(98, color="#9F6900", linestyle=(0, (5, 4)), linewidth=1.5)
        ax.grid(axis="x", color="#DFE5EA")
        ax.tick_params(length=0, pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(0.055, 0.10, "Verde: ganador de cada etapa. Línea punteada: referencia del 98 %.",
             fontsize=17, color="#4B5967")
    fig.text(0.055, 0.045,
             "Misma partición de desarrollo. Checkpoints de menor pérdida de validación. Resultados exploratorios, sin test.",
             fontsize=15, color="#637180")
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"search_accuracy.{extension}", dpi=180, facecolor="white")
    plt.close(fig)
    with (output_dir / "search_accuracy.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stage", "label", "validation_accuracy", "validation_loss", "chosen_epoch",
                         "seed", "architecture", "learning_rate", "batch_size", "run_id", "stage_winner"])
        for stage in stages:
            for row in stage["runs"]:
                writer.writerow([stage["stage"], row["label"], row["accuracy"], row["loss"], row["chosen_epoch"],
                                 row["seed"], json.dumps(row["architecture"]), row["learning_rate"],
                                 row["batch_size"], row["run_id"], row["run_id"] == stage["winner"]["run_id"]])


def generate_search_table(search_dir: Path, output_dir: Path) -> None:
    """Render a concise presentation table from stages and completed extensions."""
    stages = [json.loads((search_dir / f"stage_{name}.json").read_text(encoding="utf-8"))
              for name in ("rates", "architectures", "batches")]
    batch_rows = list(stages[2]["runs"])
    for path in sorted((search_dir.parent / "batch_extension/runs").glob("*/results.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["status"] != "completed":
            continue
        reference = stages[2]["winner"]["config"]
        expected = {**reference, "batch_size": report["config"]["batch_size"], "model_seed": 42}
        if report["config"] != expected or report["metadata"] != stages[2]["identity"]:
            raise ValueError(f"Batch extension is not comparable: {path}")
        batch_rows.append({"batch_size": report["config"]["batch_size"],
                           "accuracy": report["metrics"]["validation"]["accuracy"],
                           "loss": report["metrics"]["validation"]["loss"],
                           "chosen_epoch": report["chosen_epoch"], "run_id": report["run_id"]})
    winners = [stages[0]["winner"], stages[1]["winner"],
               sorted(batch_rows, key=lambda r: (-r["accuracy"], r["loss"]))[0]]
    values = [f"{100 * row['accuracy']:.2f} %".replace(".", ",") for row in winners]
    confirmation = json.loads((search_dir / "stage_confirmation.json").read_text(encoding="utf-8"))
    confirmed = confirmation["winner"]
    selected = json.loads((search_dir / "selection.json").read_text(encoding="utf-8"))
    if confirmed["config_id"] != selected["config_id"] or confirmed["status"] != "completed":
        raise ValueError("Selection does not match completed confirmation.")
    confirmed_accuracy = f"{100 * confirmed['accuracy_mean']:.2f} %".replace(".", ",")
    hidden = "[" + ", ".join(map(str, selected["config"]["architecture"][1:-1])) + "]"
    table_rows = [
        ["Tasa de aprendizaje", "6 tasas entre\n0,00001 y 0,1", str(winners[0]["learning_rate"]).replace(".", ","), values[0]],
        ["Arquitectura", "Capas ocultas:\n[128], [256], [256, 128]",
         hidden, confirmed_accuracy],
        ["Tamaño de lote", ", ".join(map(str, sorted({row["batch_size"] for row in batch_rows}))) + "\nCon 128 neuronas",
         str(winners[2]["batch_size"]), values[2]],
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(16, 5.5), facecolor="white")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_axis_off()
    fig.text(0.045, 0.88, "Búsqueda de mejoras", fontsize=32,
             fontweight="bold", color="#172635")
    table = ax.table(cellText=table_rows,
                     colLabels=["Etapa", "Comparación", "Opción elegida", "Accuracy de\nvalidación"],
                     colWidths=[0.255, 0.325, 0.205, 0.215],
                     cellLoc="left", colLoc="left", bbox=[0.045, 0.075, 0.91, 0.68])
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    for (row, col), cell in table.get_celld().items():
        cell.PAD = 0.075
        cell.set_edgecolor("white")
        cell.set_linewidth(3)
        if row == 0:
            cell.set_facecolor("#172635")
            cell.set_text_props(color="white", weight="bold", fontsize=20)
        else:
            cell.set_facecolor("#F0F4F6" if row % 2 else "#FAFBFC")
            cell.set_text_props(color="#172635")
            if col >= 2:
                cell.set_text_props(color="#087E78", weight="bold")
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"search_summary_table.{extension}", dpi=200, facecolor="white")
    plt.close(fig)
    with (output_dir / "search_summary_table.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stage", "variants", "best_option", "validation_accuracy_percent"])
        writer.writerows([[cell.replace("\n", " ") for cell in row] for row in table_rows])


def generate_configuration_table(search_dir: Path, output_dir: Path) -> None:
    selected = json.loads((search_dir / "selection.json").read_text(encoding="utf-8"))
    exploration = json.loads((search_dir / "stage_batches.json").read_text(encoding="utf-8"))["winner"]["config"]
    tp3 = Path(__file__).resolve().parents[2]
    reference = json.loads((tp3 / selected["identity"]["reference_selection"]).read_text(encoding="utf-8"))["config"]
    configs = [reference, exploration, selected["config"]]
    def number(value):
        return f"{value:g}".replace(".", ",")
    rows = [
        ["Datos", *[Path(c["dataset"]).name for c in configs]],
        ["Arquitectura", *["[" + ", ".join(map(str, c["architecture"])) + "]" for c in configs]],
        ["Activación", *[c["activation"] for c in configs]],
        ["Optimizador", *["Momentum " + number(c["optimizer"]["momentum"]) for c in configs]],
        ["Tasa", *[number(c["optimizer"]["learning_rate"]) for c in configs]],
        ["Tamaño de lote", *[str(c["batch_size"]) for c in configs]],
        ["Salida", *[c["output"].capitalize() for c in configs]],
        ["Pérdida", *["Entropía cruzada" if c["loss"] == "cross_entropy" else c["loss"] for c in configs]],
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_axis_off()
    fig.text(0.04, 0.925, "Configuraciones de ej2 y ej3", fontsize=31,
             fontweight="bold", color="#172635")
    table = ax.table(cellText=rows,
                     colLabels=["Aspecto", "Ej2\nSeleccionada", "Ej3\nGanadora inicial", "Ej3\nSeleccionada"],
                     colWidths=[0.225, 0.258, 0.258, 0.259],
                     cellLoc="left", colLoc="left", bbox=[0.04, 0.195, 0.92, 0.65])
    table.auto_set_font_size(False)
    table.set_fontsize(19)
    for (row, col), cell in table.get_celld().items():
        cell.PAD = 0.065
        cell.set_edgecolor("white")
        cell.set_linewidth(2)
        if row == 0:
            cell.set_height(cell.get_height() * 1.45)
            cell.set_facecolor("#172635")
            cell.set_text_props(color="white", weight="bold", fontsize=20)
        else:
            cell.set_facecolor("#F0F4F6" if row % 2 else "#FAFBFC")
            cell.set_text_props(color="#172635")
            if col == 3 or (row == 2 and col == 2):
                cell.set_text_props(color="#087E78", weight="bold")
    fig.text(0.04, 0.12, "Ganadora inicial: búsqueda con semilla 42. Seleccionada: confirmación con semillas 42, 0 y 1.",
             fontsize=17, color="#596977")
    fig.text(0.04, 0.055, "En ej3 entrenamos desde cero con los datos nuevos. Reutilizamos la configuración de ej2.",
             fontsize=17, color="#596977")
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"configuration_comparison.{extension}", dpi=200, facecolor="white")
    plt.close(fig)
    with (output_dir / "configuration_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["aspect", "ej2_selected", "ej3_exploratory", "ej3_selected"])
        writer.writerows(rows)


def generate_augmentation_chart(search_dir: Path, output_dir: Path) -> None:
    stage_path = search_dir.parent / "search_translation_widths/stage_confirmation.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    groups = []
    for augmentation in (None, {"name": "translation", "max_shift": 1}):
        matches = [row for row in stage["ranking"]
                   if row["config"]["architecture"] == [784, 256, 10]
                   and row["config"]["optimizer"]["learning_rate"] == 0.01
                   and row["config"]["batch_size"] == 32
                   and row["config"].get("augmentation") == augmentation]
        if len(matches) != 1 or matches[0]["status"] != "completed":
            raise ValueError("Expected one completed group per augmentation condition.")
        groups.append(matches[0])
    config_without_aug = [{k: v for k, v in group["config"].items() if k != "augmentation"}
                          for group in groups]
    if config_without_aug[0] != config_without_aug[1]:
        raise ValueError("Comparison changes more than augmentation.")
    runs = []
    for group in groups:
        rows = sorted([row for row in stage["runs"] if row["config_id"] == group["config_id"]],
                      key=lambda row: row["seed"])
        for row in rows:
            report_path = stage_path.parent / "runs" / row["run_id"] / "results.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (report["status"] != "completed" or report["metadata"] != stage["identity"]
                    or report["metrics"]["validation"]["accuracy"] != row["accuracy"]):
                raise ValueError(f"Run evidence differs: {report_path}")
        runs.append(rows)
    if [row["seed"] for row in runs[0]] != [row["seed"] for row in runs[1]]:
        raise ValueError("Comparison needs paired seeds.")
    means = [group["accuracy_mean"] * 100 for group in groups]
    delta = means[1] - means[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(16, 8.5), facecolor="white")
    fig.subplots_adjust(left=0.13, right=0.69, bottom=0.20, top=0.73)
    fig.text(0.055, 0.92, "Aumento de datos", fontsize=34, fontweight="bold", color="#172635")
    fig.text(0.055, 0.82, "Red de 256 neuronas · Tasa 0,01 · Lote 32", fontsize=23, color="#4B5967")
    ax.bar([0, 1], means, width=0.55, color=["#526F89", "#159D95"], zorder=3)
    for position, value, color in zip([0, 1], means, ["#526F89", "#159D95"]):
        ax.text(position, value - 8, f"{value:.2f} %".replace(".", ","),
                ha="center", va="center", color="white", fontsize=29, fontweight="bold", zorder=4)
    ax.axhline(98, color="#9F6900", linewidth=1.5, linestyle=(0, (5, 4)), zorder=4)
    ax.text(1.55, 98, "98 %", ha="right", va="bottom", fontsize=16, color="#9F6900")
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xticks([0, 1], ["Sin aumento\nde datos", "Con aumento\nde datos"])
    ax.set_ylabel("Accuracy de validación (%)", labelpad=16)
    ax.grid(axis="y", color="#E2E8ED", zorder=0)
    ax.tick_params(length=0, pad=13)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.75, 0.53, f"+{delta:.2f}".replace(".", ","), fontsize=42,
             fontweight="bold", color="#087E78")
    fig.text(0.75, 0.47, "puntos porcentuales", fontsize=20, color="#172635")
    fig.text(0.75, 0.35, "Accuracy media\nde 3 entrenamientos", fontsize=20, color="#4B5967")
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"augmentation_accuracy.{extension}", dpi=200, facecolor="white")
    plt.close(fig)
    with (output_dir / "augmentation_accuracy.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["augmentation", "seed", "validation_accuracy", "run_id", "chosen_epoch"])
        for condition, rows in zip(["none", "translation_max_shift_1"], runs):
            writer.writerows([condition, row["seed"], row["accuracy"], row["run_id"], row["chosen_epoch"]]
                             for row in rows)


def generate_l2_table(search_dir: Path, output_dir: Path) -> None:
    stage_path = search_dir.parent / "search_translation_l2/stage_joint.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    evidence = {}
    configs = []
    for row in stage["runs"]:
        config = row["config"]
        condition = bool(config.get("augmentation"))
        strength = config.get("l2", 0)
        report_path = stage_path.parent / "runs" / row["run_id"] / "results.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (row["status"] != "completed" or report["status"] != "completed"
                or row["seed"] != 42 or report["metadata"] != stage["identity"]
                or report["metrics"]["validation"]["accuracy"] != row["accuracy"]):
            raise ValueError(f"Run evidence differs: {report_path}")
        configs.append({k: v for k, v in config.items() if k not in ("l2", "augmentation")})
        evidence[(strength, condition)] = row
    if any(config != configs[0] for config in configs):
        raise ValueError("L2 comparison changes additional settings.")
    strengths = sorted({strength for strength, _ in evidence})
    rows = []
    for strength in strengths:
        label = "0 (sin L2)" if strength == 0 else f"{strength:.5f}".rstrip("0").replace(".", ",")
        rows.append([label, *[f"{100 * evidence[(strength, condition)]['accuracy']:.2f} %".replace(".", ",")
                              for condition in (False, True)]])
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(16, 7), facecolor="white")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_axis_off()
    fig.text(0.05, 0.91, "Búsqueda de mejoras con L2", fontsize=32,
             fontweight="bold", color="#172635")
    fig.text(0.05, 0.825, "Red de 256 neuronas · Tasa 0,01 · Lote 32", fontsize=22, color="#4B5967")
    table = ax.table(cellText=rows,
                     colLabels=["Penalización L2", "Sin aumento de datos", "Con aumento de datos"],
                     colWidths=[0.30, 0.35, 0.35], cellLoc="left", colLoc="left",
                     bbox=[0.05, 0.245, 0.90, 0.50])
    table.auto_set_font_size(False)
    table.set_fontsize(23)
    for (row, col), cell in table.get_celld().items():
        cell.PAD = 0.08
        cell.set_edgecolor("white")
        cell.set_linewidth(3)
        if row == 0:
            cell.set_facecolor("#172635")
            cell.set_text_props(color="white", weight="bold", fontsize=22)
        else:
            cell.set_facecolor("#F0F4F6" if row % 2 else "#FAFBFC")
            cell.set_text_props(color="#172635")
            if col == 2:
                cell.set_text_props(color="#087E78", weight="bold")
    fig.text(0.05, 0.155, "Con aumento de datos, L2 no aportó mayor accuracy", fontsize=24,
             fontweight="bold", color="#172635")
    fig.text(0.05, 0.065, "Accuracy de validación. Comparación exploratoria con semilla 42.",
             fontsize=18, color="#596977")
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"l2_accuracy_table.{extension}", dpi=200, facecolor="white")
    plt.close(fig)
    with (output_dir / "l2_accuracy_table.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["l2", "augmentation", "validation_accuracy", "seed", "run_id", "chosen_epoch"])
        for (strength, condition), row in sorted(evidence.items()):
            writer.writerow([strength, "translation_max_shift_1" if condition else "none",
                             row["accuracy"], row["seed"], row["run_id"], row["chosen_epoch"]])


def generate_l2_chart(search_dir: Path, output_dir: Path) -> None:
    stage_path = search_dir.parent / "search_translation_l2/stage_joint.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    evidence = {}
    common_config = None
    for row in stage["runs"]:
        config = row["config"]
        settings = {k: v for k, v in config.items() if k not in ("l2", "augmentation")}
        if common_config is None:
            common_config = settings
        if settings != common_config:
            raise ValueError("L2 comparison changes additional settings.")
        report = json.loads((stage_path.parent / "runs" / row["run_id"] / "results.json").read_text(encoding="utf-8"))
        if (row["status"] != "completed" or report["status"] != "completed"
                or row["seed"] != 42 or report["metadata"] != stage["identity"]
                or report["metrics"]["validation"]["accuracy"] != row["accuracy"]):
            raise ValueError(f"Run evidence differs: {row['run_id']}")
        evidence[(config.get("l2", 0), bool(config.get("augmentation")))] = row
    strengths = sorted({strength for strength, _ in evidence})
    labels = [f"{strength:.5f}".rstrip("0").replace(".", ",") for strength in strengths if strength]
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True, facecolor="white")
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.22, top=0.65, wspace=0.25)
    fig.text(0.5, 0.92, "Efecto de la regularización L2", ha="center", fontsize=32, fontweight="bold", color="#172635")
    fig.text(0.5, 0.84, "Red de 256 neuronas · Tasa 0,01 · Lote 32", ha="center", fontsize=22, color="#4B5967")
    csv_rows = []
    for ax, condition, color, title in zip(axes, (False, True), ("#526F89", "#159D95"),
                                          ("Sin aumento de datos", "Con aumento de datos")):
        baseline = evidence[(0, condition)]["accuracy"] * 100
        deltas = [evidence[(strength, condition)]["accuracy"] * 100 - baseline for strength in strengths if strength]
        ax.bar(range(len(labels)), deltas, width=0.50, color=color, zorder=3)
        ax.axhline(0, color="#172635", linewidth=1.4, zorder=4)
        for index, delta in enumerate(deltas):
            value = f"{delta:+.2f}".replace(".", ",") if abs(delta) > 1e-10 else "0,00"
            ax.text(index, delta + (0.025 if delta >= 0 else -0.025), value,
                    ha="center", va="bottom" if delta >= 0 else "top", fontsize=23,
                    fontweight="bold", color=color)
            accuracy = evidence[(strengths[index + 1], condition)]["accuracy"] * 100
            csv_rows.append([strengths[index + 1], condition, baseline / 100, accuracy / 100, delta,
                             evidence[(strengths[index + 1], condition)]["run_id"]])
        ax.set_title(title,
                     fontsize=22, fontweight="bold", color="#172635", pad=18)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_xlabel("Penalización L2", labelpad=15)
        ax.set_ylim(-0.5, 0.6)
        ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}".replace(".", ",")))
        ax.grid(axis="y", color="#E2E8ED", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("Cambio en accuracy (pp)", labelpad=12)
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"l2_accuracy_effects.{extension}", dpi=200, facecolor="white")
    plt.close(fig)
    with (output_dir / "l2_accuracy_effects.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["l2", "augmentation", "baseline_accuracy", "validation_accuracy", "delta_pp", "run_id"])
        writer.writerows(csv_rows)


def generate_ensemble_chart(search_dir: Path, output_dir: Path) -> None:
    root = search_dir.parent / "ensemble_rate015_seed42"
    validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
    final = json.loads((root / "final/final_evaluation.json").read_text(encoding="utf-8"))
    if validation["validation"] != final["validation"]:
        raise ValueError("Final evaluation refers to different validation measurements.")
    members = final["provenance"]["members"]
    if len(members) != 10 or any(abs(member["weight"] - 0.1) > 1e-12 for member in members):
        raise ValueError("Expected the equal-weight ten-model ensemble.")
    values = [validation["validation"]["accuracy"] * 100, final["results"]["accuracy"] * 100]
    counts = [(validation["correct"], validation["validation_samples"]),
              (final["correct"], final["test"]["samples"])]
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(16, 8), facecolor="white")
    fig.subplots_adjust(left=0.12, right=0.95, bottom=0.17, top=0.80)
    fig.text(0.5, 0.92, "Ensamble de 10 modelos", ha="center", fontsize=34, fontweight="bold", color="#172635")
    ax.bar([0, 1], values, width=0.54, color=["#526F89", "#159D95"], zorder=3)
    for position, value in enumerate(values):
        ax.text(position, value - 9, f"{value:.2f} %".replace(".", ","), ha="center", va="center",
                fontsize=29, fontweight="bold", color="white", zorder=4)
        correct, total = counts[position]
        ax.text(position, value - 23, f"{correct:,} / {total:,}\naciertos".replace(",", "."),
                ha="center", va="center", fontsize=22, color="white", zorder=4)
    ax.axhline(98, color="#9F6900", linewidth=1.5, linestyle=(0, (5, 4)), zorder=4)
    ax.text(1.5, 98.5, "98 %", ha="right", va="bottom", fontsize=17, color="#9F6900")
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xticks([0, 1], ["Validación", "Test"])
    ax.set_ylabel("Accuracy (%)", labelpad=14)
    ax.grid(axis="y", color="#E2E8ED", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=13)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"ensemble_accuracy.{extension}", dpi=200, facecolor="white")
    plt.close(fig)
    with (output_dir / "ensemble_accuracy.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "evaluation_set", "accuracy", "correct", "total"])
        writer.writerow(["ensemble_10_models", "validation", validation["validation"]["accuracy"], validation["correct"], validation["validation_samples"]])
        writer.writerow(["ensemble_10_models", "test", final["results"]["accuracy"], final["correct"], final["test"]["samples"]])
    with (output_dir / "ensemble_members.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["config_id", "architecture", "learning_rate", "seed", "chosen_epoch", "validation_accuracy", "weight", "model"])
        for member in members:
            writer.writerow([member["config_id"], json.dumps(member["config"]["architecture"]),
                             member["config"]["optimizer"]["learning_rate"], member["seed"], member["chosen_epoch"],
                             member["validation_accuracy"], member["weight"], member["model"]])


def generate_ensemble_table(search_dir: Path, output_dir: Path) -> None:
    root = search_dir.parent / "ensemble_rate015_seed42"
    validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
    frozen = json.loads((root / "final/frozen_ensemble.json").read_text(encoding="utf-8"))
    members = frozen["members"]
    if len(members) != 10 or len(validation["members"]) != 10:
        raise ValueError("Expected ten ensemble members.")
    rows = []
    source_rows = []
    for index, (member, original) in enumerate(zip(members, validation["members"]), 1):
        for key in ("config_id", "seed", "model", "model_sha256", "chosen_epoch", "validation_accuracy", "weight"):
            if member[key] != original[key]:
                raise ValueError(f"Frozen member {index} differs from validation: {key}")
        if abs(member["weight"] - 0.1) > 1e-12:
            raise ValueError("Expected equal weights of 1/10.")
        config = member["config"]
        width = config["architecture"][1]
        rate = config["optimizer"]["learning_rate"]
        rows.append([str(index), str(width), str(rate).replace(".", ","),
                     f"{member['validation_accuracy'] * 100:.2f} %".replace(".", ",")])
        source_rows.append([index, member["config_id"], json.dumps(config["architecture"]), rate,
                            member["seed"], member["chosen_epoch"], member["validation_accuracy"],
                            member["weight"], member["model"], member["model_sha256"]])
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
    fig.subplots_adjust(left=0.035, right=0.965, bottom=0.035, top=0.79)
    ax.axis("off")
    fig.text(0.5, 0.935, "Ensamble de 10 modelos", ha="center", fontsize=34,
             fontweight="bold", color="#172635")
    table = ax.table(cellText=rows,
                     colLabels=["Modelo", "Neuronas\nocultas", "Tasa", "Accuracy de\nvalidación"],
                     colWidths=[0.18, 0.27, 0.23, 0.32],
                     cellLoc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(23)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(2)
        if row == 0:
            cell.set_facecolor("#172635")
            cell.set_text_props(color="white", weight="bold", fontsize=21)
            cell.set_height(cell.get_height() * 1.4)
        else:
            cell.set_facecolor("#E8F4F2" if row == 10 else ("#EDF2F6" if row % 2 else "#F8FAFC"))
            cell.set_text_props(color="#172635", weight="normal")
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"ensemble_10_models.{extension}", dpi=200, facecolor="white")
    plt.close(fig)
    with (output_dir / "ensemble_10_models.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model_number", "config_id", "architecture", "learning_rate", "seed", "chosen_epoch",
                         "validation_accuracy", "weight", "model", "model_sha256"])
        writer.writerows(source_rows)


def generate_learning_rate_chart(search_dir: Path, output_dir: Path) -> None:
    """Compare matched historical runs; use validation checkpoints, never test."""
    import copy

    groups = []
    for folder, config_id, rate in (
        ("search_translation_widths", "37f123dfbe70e5ba", 0.01),
        ("search_translation_rates_fine", "2b0b7c9e299b6527", 0.015),
    ):
        rows = []
        for seed in (42, 0, 1):
            run_id = f"{config_id}-seed-{seed}"
            path = search_dir.parent / folder / "runs" / run_id
            report = json.loads((path / "results.json").read_text(encoding="utf-8"))
            if (report["status"] != "completed" or report["config"]["model_seed"] != seed
                    or report["config"]["optimizer"]["learning_rate"] != rate):
                raise ValueError(f"Unexpected run: {path}")
            with (path / "history.csv").open(encoding="utf-8", newline="") as stream:
                history = list(csv.DictReader(stream))
            rows.append((run_id, report, history))
        groups.append(rows)
    for left, right in zip(*groups):
        configs = [copy.deepcopy(row[1]["config"]) for row in (left, right)]
        for config in configs:
            config["optimizer"].pop("learning_rate")
            config.setdefault("l2", 0)
        if configs[0] != configs[1]:
            raise ValueError("Comparison changes more than the learning rate.")
        if left[1]["metadata"]["split_sha256"] != right[1]["metadata"]["split_sha256"]:
            raise ValueError("Comparison needs the same development split.")

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 19, "svg.fonttype": "none"})
    colors = ["#526F89", "#159D95"]
    labels = ["LR = 0,01", "LR = 0,015"]
    values = [[100 * row[1]["metrics"]["validation"]["accuracy"] for row in group] for group in groups]
    means = [sum(group) / len(group) for group in values]
    decimal = lambda value: f"{value:.2f}".replace(".", ",")
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), facecolor="white")
    fig.subplots_adjust(left=0.09, right=0.95, bottom=0.12, top=0.70, wspace=0.42)
    fig.text(0.07, 0.93, "Ajuste de la tasa de aprendizaje", fontsize=33, fontweight="bold", color="#172635")
    fig.text(0.07, 0.85, "256 neuronas · Momentum · Lote 32 · Traslaciones de hasta 1 píxel", fontsize=21, color="#4B5967")
    ax = axes[0]
    ax.bar([0, 1], [max(group) for group in values], color=colors, width=0.55, zorder=3)
    for i, group in enumerate(values):
        ax.text(i, max(group) - 9, decimal(max(group)) + " %", ha="center", color="white", fontsize=28, fontweight="bold")
    ax.axhline(98, color="#9F6900", linestyle="--", linewidth=1.4)
    ax.set(xticks=[0, 1], xticklabels=labels, ylim=(0, 105), ylabel="Accuracy de validación (%)")
    ax.set_title("Mejor resultado entre 3 semillas", pad=18, fontsize=21, fontweight="bold")
    ax = axes[1]
    for i, seed in enumerate((42, 0, 1)):
        ax.plot([0, 1], [group[i] for group in values], color="#BCC6CF", linewidth=2, zorder=1)
        for j, group in enumerate(values):
            ax.scatter(j, group[i], color=colors[j], s=90, zorder=3)
            ax.annotate(f"{decimal(group[i])} % · s{seed}", (j, group[i]),
                        xytext=(-12 if j == 0 else 12, 0), textcoords="offset points",
                        ha="right" if j == 0 else "left", va="center", fontsize=15, color=colors[j])
    ax.axhline(98, color="#9F6900", linestyle="--", linewidth=1.4)
    ax.set(xticks=[0, 1], xticklabels=labels, xlim=(-0.85, 1.85), ylim=(97.3, 98.35))
    ax.set_title("Comparación con 3 semillas", pad=18, fontsize=23, fontweight="bold")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}".replace(".", ",")))
    for ax in axes:
        ax.grid(axis="y", color="#E2E8ED")
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=12)
        for spine in ax.spines.values():
            spine.set_visible(False)
    mean_text = (
        f"Accuracy media de 3 semillas: LR 0,01 = {means[0]:.4f} % · "
        f"LR 0,015 = {means[1]:.4f} % · Incremento: {means[1] - means[0]:.4f} puntos porcentuales"
    ).replace(".", ",")
    fig.text(0.5, 0.035, mean_text, ha="center", fontsize=13, color="#637180")
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"learning_rate_accuracy.{extension}", dpi=200, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.12, top=0.73)
    fig.text(0.08, 0.93, "Evolución del error durante el entrenamiento", fontsize=31, fontweight="bold", color="#172635")
    fig.text(0.08, 0.85, "Semilla 42 · Misma red y partición · Curvas completas hasta early stopping", fontsize=21, color="#4B5967")
    for group, label, color in zip(groups, labels, colors):
        report, history = group[0][1:]
        epochs = [int(row["epoch"]) for row in history]
        for metric, style, suffix in (("validation_loss", "-", "validación"), ("train_loss", "--", "train")):
            ax.plot(epochs, [float(row[metric]) for row in history], style, color=color, linewidth=2.2, label=f"{label} · {suffix}")
        ax.scatter(report["chosen_epoch"], report["metrics"]["validation"]["loss"], color=color, s=120, marker="*", zorder=5)
    ax.set(xlabel="Época", ylabel="Entropía cruzada", ylim=(0, None))
    ax.grid(color="#E2E8ED")
    ax.legend(frameon=False, fontsize=17)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"learning_rate_loss.{extension}", dpi=200, facecolor="white")
    plt.close(fig)
    with (output_dir / "learning_rate_accuracy.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["learning_rate", "seed", "validation_accuracy", "validation_loss", "chosen_epoch", "source"])
        for group in groups:
            for run_id, report, _ in group:
                writer.writerow([report["config"]["optimizer"]["learning_rate"], report["config"]["model_seed"],
                                 report["metrics"]["validation"]["accuracy"], report["metrics"]["validation"]["loss"],
                                 report["chosen_epoch"], run_id])


def main() -> None:
    exercise = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chart", choices=("datasets", "controls", "search", "search-table", "config-table", "augmentation", "l2-table", "l2-effects", "ensemble", "ensemble-table", "learning-rate"), default="datasets")
    parser.add_argument("--report", type=Path, default=exercise / "results/data_report.json")
    parser.add_argument("--controls-stage", type=Path, default=exercise / "results/search/stage_controls.json")
    parser.add_argument("--search-dir", type=Path, default=exercise / "results/search")
    parser.add_argument("--output-dir", type=Path, default=exercise / "results/presentation")
    args = parser.parse_args()
    if args.chart == "learning-rate":
        generate_learning_rate_chart(args.search_dir, args.output_dir)
    elif args.chart == "ensemble-table":
        generate_ensemble_table(args.search_dir, args.output_dir)
    elif args.chart == "ensemble":
        generate_ensemble_chart(args.search_dir, args.output_dir)
    elif args.chart == "l2-effects":
        generate_l2_chart(args.search_dir, args.output_dir)
    elif args.chart == "l2-table":
        generate_l2_table(args.search_dir, args.output_dir)
    elif args.chart == "augmentation":
        generate_augmentation_chart(args.search_dir, args.output_dir)
    elif args.chart == "config-table":
        generate_configuration_table(args.search_dir, args.output_dir)
    elif args.chart == "search-table":
        generate_search_table(args.search_dir, args.output_dir)
    elif args.chart == "search":
        generate_search(args.search_dir, args.output_dir)
    elif args.chart == "controls":
        generate_controls(args.controls_stage, args.output_dir)
    else:
        generate(args.report, args.output_dir)
    print(f"Saved PNG, SVG and source counts in {args.output_dir}")


if __name__ == "__main__":
    main()
