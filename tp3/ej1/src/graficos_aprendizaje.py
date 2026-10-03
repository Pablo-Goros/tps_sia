"""Gráficos del estudio de aprendizaje (TP3 - SIA).

Lee salidas/aprendizaje/resultados.json (producido por experimentos_aprendizaje.py)
y genera las figuras. El análisis está separado de la ejecución de experimentos,
como recomienda el enunciado.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIR = Path(__file__).resolve().parents[1] / "salidas" / "aprendizaje"

# Paleta categórica validada (slots 1-3) + tintas de texto.
C_LINEAL = "#2a78d6"
C_LOGIS = "#eb6834"
C_REF = "#52514e"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
SUPERFICIE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE,
    "savefig.facecolor": SUPERFICIE, "font.size": 10,
    "axes.edgecolor": "#d8d7d2", "axes.labelcolor": TINTA_2,
    "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "axes.titlecolor": TINTA, "axes.grid": True,
    "grid.color": "#e7e6e1", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _guardar(fig, nombre):
    ruta = os.path.join(DIR, nombre)
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  ->", ruta)


# --------------------------------------------------------------------------- #
def g_curvas(R):
    cur, piso = R["curvas"], R["piso_analitico_lineal"]["mse"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    # Panel izquierdo: MSE por época (media ± desvío sobre 5 semillas).
    for nombre, color, etiqueta in (("lineal", C_LINEAL, "Perceptrón lineal"),
                                    ("logistica", C_LOGIS, "Perceptrón no lineal (logística)")):
        m = np.array(cur[nombre]["mse_medio_por_epoca"])
        s_ = np.array(cur[nombre]["mse_desvio_por_epoca"])
        x = np.arange(1, len(m) + 1)
        axes[0].plot(x, m, color=color, lw=2, label=etiqueta)
        axes[0].fill_between(x, m - s_, m + s_, color=color, alpha=0.18, lw=0)
    axes[0].axhline(piso, color=C_REF, ls="--", lw=1.5,
                    label=f"Óptimo analítico del lineal (MSE={piso:.4f})")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("Época (escala log)"); axes[0].set_ylabel("MSE sobre las 7500 muestras")
    axes[0].set_title("El lineal se detiene exactamente sobre su óptimo teórico",
                      loc="left", fontsize=11)
    axes[0].legend(fontsize=9, loc="lower left")

    # Panel derecho: cuánto baja el MSE en cada época -> se apaga = capacidad agotada.
    for nombre, color, etiqueta in (("lineal", C_LINEAL, "Perceptrón lineal"),
                                    ("logistica", C_LOGIS, "Perceptrón no lineal (logística)")):
        m = np.array(cur[nombre]["mse_medio_por_epoca"])
        delta = np.abs(np.diff(m))
        delta[delta == 0] = np.nan
        axes[1].plot(np.arange(2, len(m) + 1), delta, color=color, lw=1.6, label=etiqueta)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("Época (escala log)")
    axes[1].set_ylabel("|ΔMSE| de una época a la siguiente")
    axes[1].set_title("La mejora por época cae ~6 órdenes: no queda nada por aprender",
                      loc="left", fontsize=11)
    axes[1].legend(fontsize=9)

    fig.suptitle("Curvas de aprendizaje sobre las 7500 muestras (media ± desvío, 5 semillas)",
                 x=0.012, ha="left", fontsize=12.5, color=TINTA, y=1.04)
    _guardar(fig, "01_curvas_aprendizaje.png")


def g_barrido_eta(R):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    piso = R["piso_analitico_lineal"]["mse"]
    for nombre, color, etiqueta in (("lineal", C_LINEAL, "Perceptrón lineal"),
                                    ("logistica", C_LOGIS, "Perceptrón no lineal (logística)")):
        fs = [f for f in R["barrido_eta"] if f["activacion"] == nombre]
        xs = [f["eta"] for f in fs if f["mse_medio"] is not None]
        ys = [f["mse_medio"] for f in fs if f["mse_medio"] is not None]
        ax.plot(xs, ys, "o-", color=color, lw=2, ms=8, label=etiqueta)
        div = [f["eta"] for f in fs if f["mse_medio"] is None]
        for d in div:
            ax.axvline(d, color=color, ls=":", lw=1, alpha=0.5)
    ax.axhline(piso, color=C_REF, ls="--", lw=1.5, label="Óptimo analítico del lineal")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Tasa de aprendizaje η")
    ax.set_ylabel(f"MSE tras {R['configuracion']['epocas_barrido']} épocas")
    ax.set_title("Ningún η rescata al lineal: el techo no es de optimización, es de capacidad",
                 loc="left", fontsize=11.5)
    ax.legend(fontsize=9)
    _guardar(fig, "02_barrido_eta.png")


def g_predicho_vs_real(R):
    y = np.array(R["y_real"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (nombre, color, etiqueta) in zip(axes, (
            ("lineal", C_LINEAL, "Perceptrón lineal"),
            ("logistica", C_LOGIS, "Perceptrón no lineal (logística)"))):
        o = np.array(R["curvas"][nombre]["predicciones_mejor"])
        ax.axhspan(-0.6, 0, color="#e34948", alpha=0.10, lw=0)
        ax.axhspan(1, 1.6, color="#e34948", alpha=0.10, lw=0)
        ax.scatter(y, o, s=5, color=color, alpha=0.22, lw=0, rasterized=True)
        ax.plot([0, 1], [0, 1], color=TINTA, lw=1.4, ls="-")
        met = R["curvas"][nombre]["metricas_por_semilla"][
            R["configuracion"]["semillas"].index(R["curvas"][nombre]["mejor_semilla"])]
        ax.set_xlabel("BigModel (valor deseado ζ)")
        ax.set_title(f"{etiqueta}\nR²={met['r2']:.3f}   MSE={met['mse']:.4f}   "
                     f"fuera de [0,1]: {met['fuera_de_rango']:.1%}",
                     loc="left", fontsize=10.5)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.55, 1.55)
    axes[0].set_ylabel("Salida del perceptrón (O)")
    fig.suptitle("La nube del lineal se curva alrededor de la identidad y se sale de [0,1]",
                 x=0.012, ha="left", fontsize=12.5, color=TINTA, y=1.02)
    _guardar(fig, "03_predicho_vs_real.png")


def g_residuos(R):
    y = np.array(R["y_real"])
    bins = np.linspace(0, 1, 11)
    centros = (bins[:-1] + bins[1:]) / 2
    idx = np.digitize(y, bins) - 1
    idx = np.clip(idx, 0, 9)
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    for nombre, color, etiqueta in (("lineal", C_LINEAL, "Perceptrón lineal"),
                                    ("logistica", C_LOGIS, "Perceptrón no lineal (logística)")):
        o = np.array(R["curvas"][nombre]["predicciones_mejor"])
        r = o - y
        medias = np.array([r[idx == b].mean() for b in range(10)])
        desv = np.array([r[idx == b].std() for b in range(10)])
        ax.plot(centros, medias, "o-", color=color, lw=2, ms=7, label=etiqueta)
        ax.fill_between(centros, medias - desv, medias + desv, color=color, alpha=0.13, lw=0)
    ax.axhline(0, color=TINTA, lw=1.2)
    ax.set_xlabel("Decil del valor deseado ζ (probabilidad de BigModel)")
    ax.set_ylabel("Residuo medio  (O − ζ)")
    ax.set_title("Firma de underfitting: el residuo del lineal es estructura, no ruido",
                 loc="left", fontsize=11.5)
    ax.legend(fontsize=9)
    _guardar(fig, "04_residuos_por_decil.png")


def g_beta(R):
    fs = R["barrido_beta"]
    b = [f["beta"] for f in fs]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(b, [f["mse_medio"] for f in fs], "o-", color=C_LOGIS, lw=2, ms=8)
    axes[0].set_ylabel("MSE final"); axes[0].set_yscale("log")
    axes[0].set_title("Error alcanzado", loc="left", fontsize=11)
    axes[1].plot(b, [100 * f["fraccion_saturada"] for f in fs], "o-", color=C_REF, lw=2, ms=8)
    axes[1].set_ylabel("% de muestras saturadas  (θ'(h) normalizada < 0.05)")
    axes[1].set_title("Saturación de la sigmoidea", loc="left", fontsize=11)
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("β de la función logística")
    fig.suptitle("β controla el compromiso entre pendiente útil y saturación del gradiente",
                 x=0.012, ha="left", fontsize=12.5, color=TINTA, y=1.03)
    _guardar(fig, "05_barrido_beta.png")


def g_capacidad(R):
    cap = R["capacidad"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for nombre, color, etiqueta in (("lineal", C_LINEAL, "Perceptrón lineal"),
                                    ("logistica", C_LOGIS, "Perceptrón no lineal (logística)")):
        cortes = cap[nombre]["cortes"]
        mse = np.array(cap[nombre]["mse_en_corte"])
        piso_propio = float(np.min(cap[nombre]["curva_completa"]))
        axes[0].plot(cortes, mse, "o-", color=color, lw=2, ms=7, label=etiqueta)
        axes[1].plot(cortes, np.maximum(mse - piso_propio, 1e-12), "o-",
                     color=color, lw=2, ms=7, label=etiqueta)
    axes[0].axhline(R["piso_analitico_lineal"]["mse"], color=C_REF, ls="--", lw=1.5,
                    label="Óptimo analítico del lineal")
    axes[0].set_ylabel("MSE alcanzado"); axes[0].set_yscale("log")
    axes[0].set_title("Error en función del presupuesto de épocas", loc="left", fontsize=11)
    axes[1].set_ylabel("Distancia al piso propio de cada modelo")
    axes[1].set_yscale("log")
    axes[1].set_title("Cada modelo toca su techo antes de la época 25", loc="left", fontsize=11)
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("Épocas (escala log)"); ax.legend(fontsize=9)
    fig.suptitle("Saturación de capacidades: más épocas ya no compran error",
                 x=0.012, ha="left", fontsize=12.5, color=TINTA, y=1.03)
    _guardar(fig, "06_saturacion_capacidad.png")


def g_distribucion(R):
    y = np.array(R["y_real"])
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    bins = np.linspace(-0.5, 1.5, 81)
    ax.hist(y, bins=bins, color=TINTA_2, alpha=0.30, label="BigModel (ζ, valor deseado)")
    for nombre, color, etiqueta in (("lineal", C_LINEAL, "Perceptrón lineal"),
                                    ("logistica", C_LOGIS, "Perceptrón no lineal (logística)")):
        o = np.array(R["curvas"][nombre]["predicciones_mejor"])
        ax.hist(o, bins=bins, histtype="step", lw=2, color=color, label=etiqueta)
    ax.axvline(0, color="#e34948", lw=1.2, ls=":")
    ax.axvline(1, color="#e34948", lw=1.2, ls=":")
    ax.set_xlabel("Probabilidad estimada"); ax.set_ylabel("Cantidad de transacciones")
    ax.set_title("Solo la logística respeta el dominio [0,1] de una probabilidad",
                 loc="left", fontsize=11.5)
    ax.legend(fontsize=9)
    _guardar(fig, "07_distribucion_salidas.png")


def g_exploracion(R):
    corr = R["exploracion"]["correlacion_pearson_con_objetivo"]
    corr.pop("flagged_fraud", None)
    nombres = list(corr.keys()); vals = [corr[n] for n in nombres]
    orden = np.argsort(vals)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    colores = [C_LOGIS if vals[i] > 0 else C_LINEAL for i in orden]
    axes[0].barh([nombres[i] for i in orden], [vals[i] for i in orden], color=colores)
    axes[0].axvline(0, color=TINTA, lw=1)
    axes[0].set_xlabel("Correlación de Pearson con la salida de BigModel")
    axes[0].set_title("Qué features informan al objetivo", loc="left", fontsize=11)
    axes[0].grid(axis="y", visible=False)
    y = np.array(R["y_real"])
    axes[1].hist(y, bins=50, color=TINTA_2, alpha=0.55)
    axes[1].set_xlabel("big_model_fraud_probability")
    axes[1].set_ylabel("Cantidad de transacciones")
    axes[1].set_title(f"Distribución del objetivo (media {y.mean():.3f}, "
                      f"{(y > 0.99).sum()} casos > 0.99)", loc="left", fontsize=11)
    _guardar(fig, "08_exploracion_datos.png")


def tabla_resumen(R):
    """Tabla comparativa en Markdown para pegar en la presentación."""
    filas = []
    for nombre, etiqueta in (("lineal", "Lineal (identidad)"),
                             ("logistica", "No lineal (logística)")):
        m, d = R["curvas"][nombre]["metricas_medias"], R["curvas"][nombre]["metricas_desvio"]
        cfg = R["curvas"][nombre]["config"]
        filas.append((etiqueta, cfg, m, d))
    piso = R["piso_analitico_lineal"]
    txt = ["| Modelo | η | β | MSE | RMSE | MAE | R² | Error máx. | Salidas fuera de [0,1] |",
           "|---|---|---|---|---|---|---|---|---|"]
    for etiqueta, cfg, m, d in filas:
        beta = cfg["beta"] if "logística" in etiqueta else "—"
        txt.append(f"| {etiqueta} | {cfg['eta']} | {beta} "
                   f"| {m['mse']:.5f} ± {d['mse']:.5f} | {m['rmse']:.4f} | {m['mae']:.4f} "
                   f"| {m['r2']:.4f} | {m['error_max']:.3f} | {m['fuera_de_rango']:.2%} |")
    txt.append(f"| *Óptimo analítico del lineal (mín. cuadrados)* | — | — | {piso['mse']:.5f} | "
               f"{piso['rmse']:.4f} | {piso['mae']:.4f} | {piso['r2']:.4f} | "
               f"{piso['error_max']:.3f} | {piso['fuera_de_rango']:.2%} |")
    ruta = os.path.join(DIR, "tabla_resumen.md")
    with open(ruta, "w") as fh:
        fh.write("\n".join(txt) + "\n")
    print("  ->", ruta)
    print("\n".join(txt))


def main():
    with open(os.path.join(DIR, "resultados.json")) as fh:
        R = json.load(fh)
    print("Generando figuras...")
    g_curvas(R); g_barrido_eta(R); g_predicho_vs_real(R); g_residuos(R)
    g_beta(R); g_capacidad(R); g_distribucion(R); g_exploracion(R)
    tabla_resumen(R)


if __name__ == "__main__":
    main()
