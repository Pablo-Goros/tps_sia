"""Gráficos y tabla del opcional ReLU (TP3 - SIA).

Lee ej1/salidas/relu/resultados.json (producido por experimentos_relu.py).

    python -m tps_sia.tp3.ej1.src.graficos_relu
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Importarlo aplica también el estilo (rcParams) de los gráficos de aprendizaje.
from .graficos_aprendizaje import C_LINEAL, C_LOGIS, TINTA
from .graficos_generalizacion import clasificar_diapositiva_30

DIR = Path(__file__).resolve().parents[1] / "salidas" / "relu"
C_RELU = "#1baf7a"   # slot 3 de la paleta categórica (aqua)
MODELOS = (("lineal", C_LINEAL, "Lineal"), ("logistica", C_LOGIS, "Logística"),
           ("relu", C_RELU, "ReLU"))
METRICAS = (("recall", "Recall"), ("precision", "Precision"), ("f1", "F1"))


def _guardar(fig, nombre):
    ruta = os.path.join(DIR, nombre)
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  ->", ruta)


# --------------------------------------------------------------------------- #
def g_curvas(R):
    M = R["aprendizaje"]["modelos"]
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for act, color, nombre in MODELOS:
        m = np.array(M[act]["mse_medio_por_epoca"])
        s = np.array(M[act]["mse_desvio_por_epoca"])
        x = np.arange(1, len(m) + 1)
        ax.plot(x, m, color=color, lw=2,
                label=f"{nombre} (η={M[act]['config']['eta']:g}): MSE {m[-1]:.4f}")
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Época (escala log)"); ax.set_ylabel("MSE sobre las 7500 muestras")
    ax.legend(fontsize=9)
    ax.set_title("ReLU se comporta casi igual que el lineal y también satura en 5 épocas",
                 loc="left", fontsize=11.5)
    _guardar(fig, "01_relu_curvas.png")


def g_barrido_eta(R):
    """Fig. 2: MSE y % de muestras inactivas según η: con η grande la ReLU "muere"."""
    A = R["aprendizaje"]
    B = A["barrido_eta"]
    etas = [b["eta"] for b in B]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    fig.subplots_adjust(wspace=0.28)
    ax = axes[0]
    ax.plot(etas, [b["mse_medio"] for b in B], "o-", color=C_RELU, lw=2, ms=8, label="ReLU")
    for act, color, nombre in MODELOS[:2]:
        ax.axhline(A["modelos"][act]["metricas_medias"]["mse"], color=color, ls="--", lw=1.5,
                   label=f"{nombre} (su mejor η)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Tasa de aprendizaje η"); ax.set_ylabel("MSE tras 60 épocas")
    ax.legend(fontsize=9)
    ax.set_title("Con η chico, ReLU iguala al lineal", loc="left", fontsize=11)
    ax = axes[1]
    ax.plot(etas, [100 * b["muertas_final"] for b in B], "o-", color=C_RELU, lw=2, ms=8)
    for b in B:
        ax.annotate(f"{100 * b['muertas_final']:.0f} %", (b["eta"], 100 * b["muertas_final"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5,
                    color=TINTA)
    ax.set_xscale("log"); ax.set_ylim(0, 105)
    ax.set_xlabel("Tasa de aprendizaje η")
    ax.set_ylabel("% de muestras con h ≤ 0 (sin gradiente)")
    ax.set_title("Con η grande la ReLU \"muere\": 94 % sin gradiente", loc="left", fontsize=11)
    fig.suptitle("Barrido de η para ReLU (7500 muestras, 3 semillas)",
                 x=0.012, ha="left", fontsize=12.5, color=TINTA, y=1.04)
    _guardar(fig, "02_relu_barrido_eta.png")


def g_predicho_vs_real(R):
    M = R["aprendizaje"]["modelos"]
    y = np.array(R["y"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, (act, color, nombre) in zip(axes, MODELOS):
        o = np.array(M[act]["predicciones"])
        ax.axhspan(-0.6, 0, color="#e34948", alpha=0.08, lw=0)
        ax.axhspan(1, 1.8, color="#e34948", alpha=0.08, lw=0)
        ax.scatter(y, o, s=5, color=color, alpha=0.25, lw=0, rasterized=True)
        ax.plot([0, 1], [0, 1], color=TINTA, lw=1.2)
        m = M[act]
        ax.set_title(f"{nombre}: R²={m['metricas_medias']['r2']:.3f}  "
                     f"fuera de [0,1]: {m['metricas_medias']['fuera_de_rango']:.1%}",
                     loc="left", fontsize=10.5)
        ax.set_xlabel("BigModel (ζ)")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.5, 1.7)
    axes[0].set_ylabel("Salida del perceptrón")
    fig.suptitle("ReLU corta las salidas negativas en 0, pero sigue pasándose de 1 "
                 "(zona sombreada = fuera de [0,1])",
                 x=0.012, ha="left", fontsize=12.5, color=TINTA, y=1.04)
    _guardar(fig, "03_relu_predicho_vs_real.png")


def g_sin_gradiente(R):
    M = R["aprendizaje"]["modelos"]
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    for act, color, nombre, txt in (
            ("relu", C_RELU, "ReLU", "h ≤ 0 (inactivas: θ′ = 0)"),
            ("logistica", C_LOGIS, "Logística", "sigmoide saturada (θ′ normalizada < 0.05)")):
        v = 100 * np.array(M[act]["sin_gradiente_por_epoca"])
        ax.plot(np.arange(1, len(v) + 1), v, color=color, lw=2, label=f"{nombre}: {txt}")
    ax.set_xscale("log")
    ax.set_xlabel("Época (escala log)")
    ax.set_ylabel("% de muestras sin gradiente")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.set_title("Con η chico, sólo ~2 % de muestras inactivas: la ReLU opera casi siempre "
                 "como el lineal", loc="left", fontsize=11)
    _guardar(fig, "04_relu_sin_gradiente.png")


def g_residuos(R):
    M = R["aprendizaje"]["modelos"]
    y = np.array(R["y"])
    bins = np.linspace(0, 1, 11)
    centros = (bins[:-1] + bins[1:]) / 2
    idx = np.clip(np.digitize(y, bins) - 1, 0, 9)
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    for act, color, nombre in MODELOS:
        r = np.array(M[act]["predicciones"]) - y
        ax.plot(centros, [r[idx == b].mean() for b in range(10)], "o-", color=color, lw=2,
                ms=7, label=nombre)
    ax.axhline(0, color=TINTA, lw=1.1)
    ax.set_xlabel("Decil del valor deseado ζ")
    ax.set_ylabel("Residuo medio (O − ζ)")
    ax.legend(fontsize=9)
    ax.set_title("ReLU deja el mismo patrón de residuos que el lineal",
                 loc="left", fontsize=11.5)
    _guardar(fig, "05_relu_residuos.png")


def g_generalizacion(R):
    E = R["generalizacion"]["evaluacion"]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    rng = np.random.default_rng(0)
    for j, (act, color, nombre) in enumerate(MODELOS):
        for i, (met, _) in enumerate(METRICAS):
            v = [f["prueba"][met] for f in E[act]["folds"]]
            x0 = i * 4 + j
            ax.scatter(x0 + rng.uniform(-0.2, 0.2, len(v)), v, s=20, color=color, alpha=0.7,
                       lw=0, zorder=3, label=nombre if i == 0 else None)
            ax.hlines(np.mean(v), x0 - 0.34, x0 + 0.34, color=TINTA, lw=2, zorder=4)
            ax.text(x0, max(v) + 0.01, f"{np.mean(v):.3f}", ha="center", va="bottom",
                    fontsize=8.5, color=TINTA)
    ax.set_xticks([i * 4 + 1 for i in range(3)], [n for _, n in METRICAS])
    ax.set_ylabel("Valor en el fold de prueba")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title("Generalización (k-fold 5 × 3, umbral con recall ≥ 95 % elegido en "
                 "cada fold de entrenamiento)", loc="left", fontsize=11)
    _guardar(fig, "06_relu_generalizacion.png")


# --------------------------------------------------------------------------- #
def _clasif(R):
    E = R["generalizacion"]["evaluacion"]
    return clasificar_diapositiva_30([
        (nombre, [f["entrenamiento"]["f1"] for f in E[act]["folds"]],
         [f["prueba"]["f1"] for f in E[act]["folds"]]) for act, _, nombre in MODELOS])


def tabla(R):
    A, G = R["aprendizaje"], R["generalizacion"]
    M = A["modelos"]
    t = ["## Opcional ReLU - resumen", "", "### Aprendizaje (7500 muestras, 5 semillas)", "",
         "| Modelo | η | MSE | R² | Fuera de [0,1] | Salidas = 0 | Salidas > 1 | 99 % de la mejora | "
         "Sin gradiente al final |", "|---|---|---|---|---|---|---|---|---|"]
    for act, _, nombre in MODELOS:
        m, mm = M[act], M[act]["metricas_medias"]
        t.append(f"| {nombre} | {m['config']['eta']:g} | {mm['mse']:.5f} | {mm['r2']:.3f} | "
                 f"{mm['fuera_de_rango']:.2%} | {m['salidas_en_cero']:.2%} | "
                 f"{m['salidas_mayores_a_1']:.2%} | {m['epocas_99pct_mejora']} épocas | "
                 f"{m['sin_gradiente_por_epoca'][-1]:.1%} |")
    t += ["", "| Modelo | " + " | ".join(M["lineal"]["error_por_zona"]) + " |",
          "|---|" + "---|" * len(M["lineal"]["error_por_zona"])]
    for act, _, nombre in MODELOS:
        t.append(f"| {nombre} | " + " | ".join(
            f"MSE {v['mse']:.4f}, sesgo {v['sesgo']:+.3f}" for v in M[act]["error_por_zona"].values()) + " |")
    t += ["", "Barrido de η (ReLU, 60 épocas):", "", "| η | MSE | Inactivas al final | Divergentes |",
          "|---|---|---|---|"]
    for b in A["barrido_eta"]:
        mse = f"{b['mse_medio']:.5f}" if b["mse_medio"] is not None else "diverge"
        t.append(f"| {b['eta']:g} | {mse} | {b['muertas_final']:.1%} | {b['corridas_divergentes']}/3 |")

    t += ["", "### Generalización (k-fold 5 × 3, recall ≥ 95 % con máxima precision)", "",
          "Grilla ReLU (60 épocas):", "", "| Features | η | Recall | Precision | F1 | MSE val |",
          "|---|---|---|---|---|---|"]
    for g in G["grilla"]:
        t.append(f"| {g['features']} | {g['eta']:g} | {g['recall']:.3f} | {g['precision']:.3f} | "
                 f"{g['f1']:.3f} | {g['mse_validacion']:.5f} |")
    t += ["", "Épocas (ReLU elegida):", "", "| Épocas | Recall | Precision | F1 |", "|---|---|---|---|"]
    for e in G["epocas"]:
        marca = " ★" if e["epocas"] == G["epocas_relu"] else ""
        t.append(f"| {e['epocas']}{marca} | {e['recall']:.3f} | {e['precision']:.3f} | {e['f1']:.3f} |")
    t += ["", "Estimación final (mismos folds para los tres modelos):", "",
          "| Modelo | Configuración | Umbral | Recall | Precision | F1 prueba | F1 entrenamiento | "
          "Diapositiva 30 |", "|---|---|---|---|---|---|---|---|"]
    clas = {c["nombre"]: c for c in _clasif(R)}
    E = G["evaluacion"]
    for act, _, nombre in MODELOS:
        F, cfg = E[act]["folds"], E[act]["config"]
        beta = f", β={cfg['beta']:g}" if act == "logistica" else ""
        t.append(f"| {nombre} | {cfg['features']}, η={cfg['eta']:g}{beta}, {cfg['epocas']} ép. | "
                 f"{np.mean([f['umbral'] for f in F]):.3f} | " +
                 " | ".join(f"{np.mean([f['prueba'][k] for f in F]):.3f} ± "
                            f"{np.std([f['prueba'][k] for f in F], ddof=1):.3f}"
                            for k in ("recall", "precision")) +
                 f" | {clas[nombre]['f1_prueba']:.3f} | {clas[nombre]['f1_entrenamiento']:.3f} | "
                 f"{clas[nombre]['resultado']} |")
    md = "\n".join(t) + "\n"
    with open(os.path.join(DIR, "tabla_relu.md"), "w") as fh:
        fh.write(md)
    print(md)


def main():
    with open(os.path.join(DIR, "resultados.json")) as fh:
        R = json.load(fh)
    print("Generando figuras...")
    g_curvas(R); g_barrido_eta(R); g_predicho_vs_real(R); g_sin_gradiente(R)
    g_residuos(R); g_generalizacion(R)
    tabla(R)


if __name__ == "__main__":
    main()
