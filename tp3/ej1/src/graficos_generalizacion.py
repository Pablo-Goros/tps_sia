"""Gráficos y tablas del estudio de generalización (TP3 - SIA).

Lee salidas/generalizacion/resultados.json (producido por
experimentos_generalizacion.py) y genera las figuras 11-17 y la tabla resumen.
Métricas de evaluación: precision, recall y F1 (recall como prioridad).

    python -m tps_sia.tp3.ej1.src.graficos_generalizacion
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
from .graficos_aprendizaje import C_LINEAL, C_LOGIS, C_REF, TINTA, TINTA_2

DIR = Path(__file__).resolve().parents[1] / "salidas" / "generalizacion"
C_ESTRAT = "#1baf7a"      # slot 3 (aqua): partición estratificada / recall
C_PRECISION = "#4a3aa7"   # slot 7 (violeta): precision / transacciones con fraude
C_LEGITIMA = "#a3a29c"    # gris neutro: transacciones legítimas
METRICAS = (("recall", "Recall"), ("precision", "Precision"), ("f1", "F1"))


def _guardar(fig, nombre):
    ruta = os.path.join(DIR, nombre)
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  ->", ruta)


def _elegida(R):
    cv = R["g2_validacion_cruzada"]
    e = cv["elegida"]
    return next(c for c in cv["configuraciones"]
                if (c["features"], c["eta"], c["beta"]) == (e["features"], e["eta"], e["beta"]))


def _pct(x):
    return f"{100 * x:.1f} %"


def clasificar_diapositiva_30(filas):
    """Clasificación de la diapositiva 30 a partir del F1 por fold en entrenamiento y prueba.

    `filas`: lista de (nombre, f1_entrenamiento_por_fold, f1_prueba_por_fold).
    Error = 1 - F1. La diapositiva no define "alto", así que se toma como referencia el
    mejor resultado del mismo experimento: el error es ALTO si la media de F1 queda más de
    2 errores estándar (σ/√n, n = folds) por debajo de la mejor media de ese conjunto.

        alto en entrenamiento | alto en prueba | resultado
        sí                    | sí             | UNDERFITTING
        no                    | sí             | OVERFITTING
        no                    | no             | BUEN MODELO
    """
    ref_ent = max(np.mean(e) for _, e, _ in filas)
    ref_pru = max(np.mean(p) for _, _, p in filas)
    salida = []
    for nombre, e, p in filas:
        e, p = np.asarray(e), np.asarray(p)
        tol_e = 2 * np.std(e, ddof=1) / np.sqrt(len(e))
        tol_p = 2 * np.std(p, ddof=1) / np.sqrt(len(p))
        alto_e, alto_p = bool(e.mean() < ref_ent - tol_e), bool(p.mean() < ref_pru - tol_p)
        if alto_e and alto_p:
            resultado = "UNDERFITTING"
        elif alto_p:
            resultado = "OVERFITTING"
        elif alto_e:
            resultado = "revisar (alto sólo en entrenamiento)"
        else:
            resultado = "BUEN MODELO"
        salida.append({"nombre": nombre, "f1_entrenamiento": float(e.mean()),
                       "f1_prueba": float(p.mean()), "brecha": float(e.mean() - p.mean()),
                       "error_alto_entrenamiento": alto_e, "error_alto_prueba": alto_p,
                       "resultado": resultado})
    return salida


def _diapositiva_30_epocas(R):
    F = R["g2b_epocas"]["filas"]
    return clasificar_diapositiva_30([
        (f["epocas"], [x["entrenamiento"]["f1"] for x in f["folds"]],
         [x["prueba"]["f1"] for x in f["folds"]]) for f in F])


def _diapositiva_30_modelos(R):
    E = R["g4_evaluacion"]
    return clasificar_diapositiva_30([
        (nombre, [f["clasificacion_entrenamiento"]["f1"] for f in E[m]["folds"]],
         [f["clasificacion"]["f1"] for f in E[m]["folds"]])
        for m, nombre in (("logistica", "Logística (elegido)"), ("lineal", "Lineal (referencia)"))])


# --------------------------------------------------------------------------- #
def g_estrategia(R):
    """Fig. 11: recall y precision de cada fold, según k y forma de partir."""
    G = R["g1_estrategia"]["configuraciones"]
    ks = sorted({g["k"] for g in G})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    rng = np.random.default_rng(0)
    for ax, (met, nombre) in zip(axes, METRICAS[:2]):
        for i, k in enumerate(ks):
            for j, (estrategia, color) in enumerate((("aleatoria", C_REF), ("estratificada", C_ESTRAT))):
                g = next(x for x in G if x["k"] == k and x["estrategia"] == estrategia)
                v = [f[met] for f in g["folds"]]
                x0 = i * 3 + j
                ax.scatter(x0 + rng.uniform(-0.18, 0.18, len(v)), v, s=22, color=color,
                           alpha=0.7, lw=0, zorder=3,
                           label=f"Partición {estrategia}" if i == 0 else None)
                ax.hlines(np.mean(v), x0 - 0.32, x0 + 0.32, color=TINTA, lw=2, zorder=4)
                ax.text(x0, min(v) - 0.006, f"±{np.std(v, ddof=1):.3f}", ha="center",
                        va="top", fontsize=8, color=TINTA_2)
        ax.set_xticks([i * 3 + 0.5 for i in range(len(ks))], [f"k = {k}" for k in ks])
        ax.set_ylabel(f"{nombre} en el fold de validación")
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.12)
    axes[0].set_title("Recall de cada fold (el umbral apunta a ≥ 95 %)", loc="left", fontsize=11)
    axes[1].set_title("Precision de cada fold", loc="left", fontsize=11)
    axes[0].legend(fontsize=9, loc="lower left")
    fig.suptitle("¿Qué k y cómo partir? Cada punto es un fold (3 repeticiones); "
                 "la línea es la media que reporta la validación cruzada",
                 x=0.012, ha="left", fontsize=12.5, color=TINTA, y=1.04)
    _guardar(fig, "11_estrategia_kfold.png")


def g_tamano(R):
    T = R["g1_tamano"]
    n = [f["n_entrenamiento"] for f in T]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    for clave, color, etiqueta, ls in (("mse_entrenamiento", C_REF, "Entrenamiento", "--"),
                                       ("mse_validacion", C_LOGIS, "Validación", "-")):
        m = np.array([f[clave] for f in T])
        s = np.array([f[clave + "_desvio"] for f in T])
        ax.plot(n, m, marker="o", ms=7, color=color, lw=2, ls=ls, label=etiqueta)
        ax.fill_between(n, m - s, m + s, color=color, alpha=0.15, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.set_xlabel("Muestras de entrenamiento (escala log)")
    ax.set_ylabel("Costo MSE (media ± desvío, 5 folds)")
    ax.set_title("Con pocos datos memoriza; desde ~1500 muestras la brecha se cierra",
                 loc="left", fontsize=11.5)
    _guardar(fig, "12_tamano_entrenamiento.png")


def g_grilla(R):
    """Fig. 13: precision con recall ≥ 95 % para cada configuración (más oscuro = mejor)."""
    cv = R["g2_validacion_cruzada"]
    C = cv["configuraciones"]
    conjuntos = list(dict.fromkeys(c["features"] for c in C))
    etas = sorted({c["eta"] for c in C})
    betas = sorted({c["beta"] for c in C})
    todos = sorted(c["precision_criterio"] for c in C)
    vmin, vmax = todos[len(todos) // 2], todos[-1]
    e = cv["elegida"]
    fig, axes = plt.subplots(1, len(conjuntos), figsize=(5.6 * len(conjuntos), 3.9))
    fig.subplots_adjust(wspace=0.35)
    for ax, fs in zip(np.atleast_1d(axes), conjuntos):
        M = np.array([[next(c["precision_criterio"] for c in C
                            if c["features"] == fs and c["eta"] == eta and c["beta"] == b)
                       for b in betas] for eta in etas])
        im = ax.imshow(M, cmap="Oranges", vmin=vmin, vmax=vmax, aspect="auto")
        for i, eta in enumerate(etas):
            for j, b in enumerate(betas):
                es = (fs, eta, b) == (e["features"], e["eta"], e["beta"])
                ax.text(j, i, f"{M[i, j]:.3f}" + ("\n★ elegida" if es else ""),
                        ha="center", va="center", fontsize=9.5,
                        color="white" if M[i, j] > (vmin + vmax) / 2 else TINTA,
                        fontweight="bold" if es else "normal")
        ax.set_xticks(range(len(betas)), [f"β={b:g}" for b in betas])
        ax.set_yticks(range(len(etas)), [f"η={x:g}" for x in etas])
        ax.set_title(f"Features: {fs}", loc="left", fontsize=11)
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.85, label="Precision con recall ≥ 95 % (más oscuro = mejor)")
    fig.suptitle(f"Selección por validación cruzada ({cv['k']}-fold estratificado × "
                 f"{cv['repeticiones']}): precision alcanzable sin bajar del 95 % de recall",
                 x=0.012, ha="left", fontsize=12.5, color=TINTA, y=1.04)
    _guardar(fig, "13_grilla_cv.png")


def g_curvas(R):
    """Fig. 14: costo por época (sobreajuste) y épocas como hiperparámetro (G2b)."""
    c = _elegida(R)
    ent, val = np.array(c["curva_entrenamiento"]), np.array(c["curva_validacion"])
    s = np.array(c["curva_validacion_desvio"])
    x = np.arange(1, len(val) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    fig.subplots_adjust(wspace=0.3)
    ax = axes[0]
    ax.plot(x, ent, color=C_REF, lw=2, ls="--", label="Entrenamiento")
    ax.plot(x, val, color=C_LOGIS, lw=2, label="Validación (± desvío entre folds)")
    ax.fill_between(x, val - s, val + s, color=C_LOGIS, alpha=0.15, lw=0)
    ax.legend(fontsize=9)
    ax.set_yscale("log"); ax.set_xlim(1, x[-1])
    ax.set_xlabel("Época"); ax.set_ylabel("Costo MSE")
    ax.set_title(f"Sin sobreajuste (brecha val/ent {100 * (val[-1] / ent[-1] - 1):.1f} %)",
                 loc="left", fontsize=11)

    ax = axes[1]
    F = R["g2b_epocas"]["filas"]
    ep = [f["epocas"] for f in F]
    for clave, color, etiqueta in (("precision", C_PRECISION, "Precision (recall ≥ 95 %)"),
                                   ("f1", C_LOGIS, "F1")):
        ax.plot(ep, [f[clave] for f in F], marker="o", ms=7, color=color, lw=2, label=etiqueta)
    elegidas = R["g2b_epocas"]["epocas_elegidas"]
    ax.axvline(elegidas, color=TINTA, lw=1.2, ls="--")
    ax.text(elegidas, ax.get_ylim()[0], f" elegidas: {elegidas}", fontsize=8.5, color=TINTA,
            va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("Épocas de entrenamiento (escala log)")
    ax.set_ylabel("Métrica (predicciones fuera de fold)")
    ax.set_title("Entrenar de más baja la precision", loc="left", fontsize=11)
    ax.legend(fontsize=9, loc="center right")
    e = R["g2_validacion_cruzada"]["elegida"]
    fig.suptitle(f"Configuración elegida ({e['features']}, η={e['eta']:g}, β={e['beta']:g})",
                 x=0.012, ha="left", fontsize=12.5, color=TINTA, y=1.04)
    _guardar(fig, "14_epocas.png")


def g_entrenamiento_vs_prueba(R):
    """Fig. 17: precision, recall y F1 en entrenamiento y prueba según las épocas (diap. 17 y 30)."""
    F = R["g2b_epocas"]["filas"]
    ep = np.array([f["epocas"] for f in F])
    clas = {c["nombre"]: c["resultado"] for c in _diapositiva_30_epocas(R)}
    abrev = {"UNDERFITTING": "under", "OVERFITTING": "over", "BUEN MODELO": "bueno"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharex=True)
    fig.subplots_adjust(wspace=0.28)
    for ax, (met, nombre) in zip(axes, METRICAS):
        for conjunto, color, etiqueta, ls in (("entrenamiento", C_REF, "Entrenamiento", "--"),
                                              ("prueba", C_LOGIS, "Prueba", "-")):
            v = np.array([[x[conjunto][met] for x in f["folds"]] for f in F])
            m, sd = v.mean(axis=1), v.std(axis=1, ddof=1)
            ax.plot(ep, m, marker="o", ms=6, color=color, lw=2, ls=ls, label=etiqueta)
            ax.fill_between(ep, m - sd, m + sd, color=color, alpha=0.13, lw=0)
        ax.axvline(R["g2b_epocas"]["epocas_elegidas"], color=TINTA, lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("Épocas (escala log)")
        ax.set_title(nombre, loc="left", fontsize=11)
        ax.margins(y=0.15)
    axes[0].set_ylabel("Media ± desvío entre folds")
    axes[0].legend(fontsize=9, loc="lower right")
    ax = axes[2]
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.32 * (hi - lo))
    for i, e in enumerate(ep):   # alturas alternadas para que no se pisen
        ax.text(e, hi + (0.06 if i % 2 else 0.15) * (hi - lo), abrev.get(clas[e], "?"),
                ha="center", va="center", fontsize=8.5, color=TINTA_2)
    ax.text(ep[0], hi + 0.26 * (hi - lo), "diapositiva 30:", ha="left", va="center",
            fontsize=8.5, color=TINTA_2)
    fig.suptitle("Entrenamiento vs. prueba en cada fold según las épocas: nunca se separan "
                 "(sin overfitting); con pocas o muchas épocas ambos empeoran",
                 x=0.012, ha="left", fontsize=12.5, color=TINTA, y=1.04)
    _guardar(fig, "17_entrenamiento_vs_prueba.png")


def g_umbral(R):
    U = R["g3_umbral"]
    B = U["barrido"]
    u = [f["umbral"] for f in B]
    rec = U["umbral_recomendado"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    for clave, color, etiqueta in (("precision", C_PRECISION, "Precision"),
                                   ("recall", C_ESTRAT, "Recall"), ("f1", C_LOGIS, "F1")):
        v = [f[clave] for f in B]
        ax.plot(u, v, color=color, lw=2)
        ax.text(u[-1] + 0.004, v[-1], etiqueta, color=TINTA_2, fontsize=9.5, va="center")
    ax.axhline(R["recall_minimo"], color=C_ESTRAT, lw=1, ls=":")
    marcas = [(rec, f"recomendado: recall ≥ 95 % ({rec:.3f})", TINTA, "--"),
              (R["umbral_bigmodel"], "0.85 (umbral de BigModel)", TINTA_2, ":"),
              (U["umbral_f1"], f"F1 máximo ({U['umbral_f1']:.3f})", TINTA_2, ":")]
    for x, txt, col, ls in marcas:
        ax.axvline(x, color=col, lw=1.2, ls=ls)
        ax.text(x, 0.02, " " + txt, rotation=90, fontsize=8, color=col, va="bottom")
    ax.set_xlim(u[0], u[-1] + 0.05); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Umbral sobre la salida de TinyModel")
    ax.set_ylabel("Métrica (predicciones fuera de fold)")
    ax.set_title("Precision, recall y F1 según el umbral", loc="left", fontsize=11)

    ax = axes[1]
    y, clase = np.array(R["y"]), np.array(R["clase"])
    oof = np.array(_elegida(R)["oof"]).mean(axis=0)
    zona = y > 0.55
    for es_fraude, color, etiqueta in ((0, C_LEGITIMA, "Legítima"), (1, C_PRECISION, "Fraude")):
        mk = zona & (clase == es_fraude)
        ax.scatter(y[mk], oof[mk], s=7, color=color, alpha=0.5, lw=0, label=etiqueta,
                   rasterized=True)
    ax.plot([0.55, 1], [0.55, 1], color=TINTA, lw=1)
    ax.axvline(R["umbral_bigmodel"], color=TINTA_2, lw=1, ls=":")
    ax.axhline(rec, color=TINTA, lw=1.2, ls="--")
    ax.text(0.555, rec, f" umbral recomendado {rec:.3f}", fontsize=8.5, color=TINTA, va="bottom")
    ax.set_xlabel("BigModel (0.85 define flagged_fraud)")
    ax.set_ylabel("TinyModel (fuera de fold)")
    ax.set_title("Bajo la línea: fraudes que TinyModel subestima", loc="left", fontsize=11)
    ax.legend(fontsize=9, loc="upper left", markerscale=2.5)
    fig.suptitle("Elección del umbral: el más alto (menos falsas alarmas) que mantiene "
                 "recall ≥ 95 %", x=0.012, ha="left", fontsize=12.5, color=TINTA, y=1.04)
    _guardar(fig, "15_umbral.png")


def g_evaluacion(R):
    """Fig. 16: generalización estimada por k-fold (umbral elegido en cada fold de entrenamiento)."""
    E = R["g4_evaluacion"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    rng = np.random.default_rng(1)
    for j, (modelo, color, etiqueta) in enumerate((("logistica", C_LOGIS, "Logística (elegido)"),
                                                   ("lineal", C_LINEAL, "Lineal (referencia)"))):
        for i, (met, nombre) in enumerate(METRICAS):
            v = [f["clasificacion"][met] for f in E[modelo]["folds"]]
            x0 = i * 3 + j
            ax.scatter(x0 + rng.uniform(-0.18, 0.18, len(v)), v, s=22, color=color, alpha=0.7,
                       lw=0, zorder=3, label=etiqueta if i == 0 else None)
            ax.hlines(np.mean(v), x0 - 0.32, x0 + 0.32, color=TINTA, lw=2, zorder=4)
            ax.text(x0, max(v) + 0.008, f"{np.mean(v):.3f}", ha="center", va="bottom",
                    fontsize=8.5, color=TINTA)
    ax.set_xticks([i * 3 + 0.5 for i in range(3)], [n for _, n in METRICAS])
    ax.set_ylabel("Valor en el fold de validación")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title(f"{len(E['logistica']['folds'])} folds por modelo "
                 f"({E['k']}-fold × {E['repeticiones']})", loc="left", fontsize=11)

    # Matriz de confusión de la repetición 0: cada transacción aparece una sola vez.
    ax = axes[1]
    fs = [f["clasificacion"] for f in E["logistica"]["folds"] if f["repeticion"] == 0]
    # Orientación de la diapositiva 10: filas = clase real, columnas = predicción,
    # positivo (fraude) primero -> [[TP, FN], [FP, TN]].
    M = np.array([[sum(f["vp"] for f in fs), sum(f["fn"] for f in fs)],
                  [sum(f["fp"] for f in fs), sum(f["vn"] for f in fs)]])
    ax.imshow(M, cmap="Greys", vmin=0, vmax=M.max() * 1.6)
    nombres = [["TP\nfraudes detectados", "FN\nfraudes no detectados"],
               ["FP\nfalsas alarmas", "TN\nlegítimas bien"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i, j]}\n{nombres[i][j]}", ha="center", va="center",
                    fontsize=10, color=TINTA)
    ax.set_xticks([0, 1], ["Predice fraude", "Predice legítima"])
    ax.set_yticks([0, 1], ["Real: fraude", "Real: legítima"])
    ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
    ax.grid(False)
    ax.set_title("Logística: las 7500 transacciones, cada una\npredicha por el modelo que no la vio",
                 loc="left", fontsize=10.5, pad=26)
    fig.suptitle("Generalización estimada por k-fold: el umbral se elige con el fold de "
                 "entrenamiento y se mide en el de validación",
                 x=0.012, ha="left", fontsize=12.5, color=TINTA, y=1.12)
    _guardar(fig, "16_evaluacion_kfold.png")


# --------------------------------------------------------------------------- #
def _ms(vals, fmt="{:.3f}"):
    return f"{fmt.format(np.mean(vals))} ± {fmt.format(np.std(vals, ddof=1))}"


def tabla(R):
    cv, U, E = R["g2_validacion_cruzada"], R["g3_umbral"], R["g4_evaluacion"]
    t = ["## Estudio de generalización - resumen", "",
         f"{R['n']} transacciones ({R['fraude']:.2%} fraude). Validación cruzada k-fold; "
         f"métricas: recall (prioridad), precision y F1. Criterio: recall ≥ "
         f"{R['recall_minimo']:.0%} y máxima precision.", ""]

    t += ["### G1 - ¿Qué k y cómo partir? (media ± desvío entre folds)", "",
          "| k | Partición | Recall | Precision | F1 | Fraudes por fold | Desvío del recall entre repeticiones |",
          "|---|---|---|---|---|---|---|"]
    for g in R["g1_estrategia"]["configuraciones"]:
        t.append(f"| {g['k']} | {g['estrategia']} | " + " | ".join(
            f"{g[m]['media']:.3f} ± {g[m]['desvio_entre_folds']:.3f}" for m in ("recall", "precision", "f1"))
            + f" | {g['fraudes_por_fold']['min']}–{g['fraudes_por_fold']['max']} | "
            f"{g['recall']['desvio_entre_repeticiones']:.4f} |")

    t += ["", "### G1 - Tamaño del conjunto de entrenamiento (costo MSE)", "",
          "| n | MSE entrenamiento | MSE validación | brecha |", "|---|---|---|---|"]
    for f in R["g1_tamano"]:
        t.append(f"| {f['n_entrenamiento']} | {f['mse_entrenamiento']:.5f} | "
                 f"{f['mse_validacion']:.5f} ± {f['mse_validacion_desvio']:.5f} | "
                 f"{100 * (f['mse_validacion'] / f['mse_entrenamiento'] - 1):+.1f} % |")

    t += ["", f"### G2 - Selección por validación cruzada ({cv['k']}-fold × {cv['repeticiones']})", "",
          "| Features | η | β | Umbral | Recall | Precision | F1 | Costo MSE val |",
          "|---|---|---|---|---|---|---|---|"]
    for c in sorted(cv["configuraciones"], key=lambda c: (-c["precision_criterio"], -c["f1_criterio"])):
        t.append(f"| {c['features']} | {c['eta']:g} | {c['beta']:g} | {c['umbral_criterio']:.3f} | "
                 f"{c['recall_criterio']:.3f} | {c['precision_criterio']:.3f} ± "
                 f"{c['precision_criterio_desvio']:.3f} | {c['f1_criterio']:.3f} | "
                 f"{c['mse_validacion']:.6f} |")
    t.append(f"\nElegida: **{cv['elegida']['features']}, η={cv['elegida']['eta']:g}, "
             f"β={cv['elegida']['beta']:g}** (con {cv['epocas']} épocas; las épocas se ajustan en G2b).")

    t += ["", "### G2b - Épocas como hiperparámetro (configuración elegida)", "",
          "| Épocas | Umbral | Recall | Precision | F1 | Costo MSE val |", "|---|---|---|---|---|---|"]
    for f in R["g2b_epocas"]["filas"]:
        marca = " ★" if f["epocas"] == R["g2b_epocas"]["epocas_elegidas"] else ""
        t.append(f"| {f['epocas']}{marca} | {f['umbral']:.3f} | {f['recall']:.3f} | "
                 f"{f['precision']:.3f} ± {f['precision_desvio']:.3f} | {f['f1']:.3f} | "
                 f"{f['mse_validacion']:.6f} |")
    t += ["", "### Entrenamiento vs. prueba por fold y clasificación de la diapositiva 30", "",
          "Umbral elegido en cada fold de entrenamiento (recall ≥ 95 %) y aplicado a ambos "
          "conjuntos. Media de los 15 folds. Error = 1 − F1; \"alto\" = media de F1 más de 2 "
          "errores estándar por debajo de la mejor del experimento.", "",
          "| Épocas | Precision ent / prueba | Recall ent / prueba | F1 ent / prueba | Brecha F1 | "
          "Error alto ent. | Error alto prueba | Diapositiva 30 |",
          "|---|---|---|---|---|---|---|---|"]
    clas = {c["nombre"]: c for c in _diapositiva_30_epocas(R)}
    for f in R["g2b_epocas"]["filas"]:
        c = clas[f["epocas"]]
        m = {(k, cj): np.mean([x[cj][k] for x in f["folds"]])
             for k in ("precision", "recall", "f1") for cj in ("entrenamiento", "prueba")}
        marca = " ★" if f["epocas"] == R["g2b_epocas"]["epocas_elegidas"] else ""
        t.append(f"| {f['epocas']}{marca} | " + " | ".join(
            f"{m[(k, 'entrenamiento')]:.3f} / {m[(k, 'prueba')]:.3f}" for k in ("precision", "recall", "f1"))
            + f" | {c['brecha']:+.3f} | {'sí' if c['error_alto_entrenamiento'] else 'no'} | "
            f"{'sí' if c['error_alto_prueba'] else 'no'} | **{c['resultado']}** |")
    t += ["", "### G3 - Umbral (predicciones fuera de fold)", "",
          "| Criterio | Umbral | Recall | Precision | F1 | Fraudes no detectados | Falsas alarmas |",
          "|---|---|---|---|---|---|---|"]
    B = {round(f["umbral"], 3): f for f in U["barrido"]}
    for nombre, u in (("Recall ≥ 95 % (recomendado)", U["umbral_recomendado"]),
                      ("F1 máximo", U["umbral_f1"]), ("Umbral de BigModel", 0.85)):
        f = B[round(u, 3)]
        t.append(f"| {nombre} | {u:.3f} | {f['recall']:.3f} | {f['precision']:.3f} | "
                 f"{f['f1']:.3f} | {f['fn']:.0f} | {f['fp']:.0f} |")
    t += ["", "| Costo de un fraude no detectado / falsa alarma | Umbral óptimo | Recall | Precision |",
          "|---|---|---|---|"]
    for c in U["por_costo"]:
        t.append(f"| {c['razon_costo_fn_fp']}× | {c['umbral']:.3f} | {c['recall']:.3f} | "
                 f"{c['precision']:.3f} |")

    t += ["", f"### G4 - Generalización estimada por k-fold ({E['k']}-fold × {E['repeticiones']}; "
          "umbral elegido en cada fold de entrenamiento)", "",
          "| Modelo | Umbral por fold | Recall | Precision | F1 | Costo MSE val |",
          "|---|---|---|---|---|---|"]
    for modelo, etiqueta in (("logistica", "Logística (elegido)"), ("lineal", "Lineal (referencia)")):
        F = E[modelo]["folds"]
        us = [f["umbral_fold"] for f in F]
        t.append(f"| {etiqueta} | {np.mean(us):.3f} [{min(us):.3f}–{max(us):.3f}] | " +
                 " | ".join(_ms([f["clasificacion"][m] for f in F]) for m in ("recall", "precision", "f1"))
                 + f" | {_ms([f['regresion']['mse'] for f in F], '{:.5f}')} |")
    t += ["", "| Modelo | F1 entrenamiento | F1 prueba | Brecha | Error alto ent. | "
          "Error alto prueba | Diapositiva 30 |", "|---|---|---|---|---|---|---|"]
    for c in _diapositiva_30_modelos(R):
        t.append(f"| {c['nombre']} | {c['f1_entrenamiento']:.3f} | {c['f1_prueba']:.3f} | "
                 f"{c['brecha']:+.3f} | {'sí' if c['error_alto_entrenamiento'] else 'no'} | "
                 f"{'sí' if c['error_alto_prueba'] else 'no'} | **{c['resultado']}** |")
    fs = [f["clasificacion"] for f in E["logistica"]["folds"] if f["repeticion"] == 0]
    tot = {k: sum(f[k] for f in fs) for k in ("vp", "fp", "fn", "vn")}
    t.append(f"\nLogística, repetición 1 (cada transacción una vez): {tot['vp']} fraudes detectados, "
             f"{tot['fn']} no detectados, {tot['fp']} falsas alarmas, {tot['vn']} legítimas bien.")
    mf = R["modelo_final"]
    t.append(f"\nModelo final (7500 muestras, {R['epocas_modelo_final']} épocas, {len(mf)} semillas): costo MSE "
             f"{_ms([m['mse_entrenamiento'] for m in mf], '{:.6f}')}; umbral entregado "
             f"{U['umbral_recomendado']:.3f}.")
    md = "\n".join(t) + "\n"
    with open(os.path.join(DIR, "tabla_generalizacion.md"), "w") as fh:
        fh.write(md)
    print(md)


def main():
    with open(os.path.join(DIR, "resultados.json")) as fh:
        R = json.load(fh)
    print("Generando figuras...")
    g_estrategia(R); g_tamano(R); g_grilla(R); g_curvas(R); g_umbral(R); g_evaluacion(R)
    g_entrenamiento_vs_prueba(R)
    tabla(R)


if __name__ == "__main__":
    main()
