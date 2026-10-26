"""Ejercicio 1.2 - Figuras de análisis de hiperparámetros de Kohonen.

Usa el barrido ya corrido por ej1_kohonen.py (resultados/kohonen/busqueda_raw.csv) y entrena
unos pocos modelos extra para las curvas. Salidas en resultados/kohonen/hp_*.png:

  hp_k.png                 efecto del tamaño de grilla: QE, TE, unidades muertas, creadores/neurona
  hp_mapas_k.png           los mapas de hits reales para k = 3, 4, 5, 6 (config final)
  hp_ablacion.png          un hiperparámetro a la vez alrededor de la config final (media ± desvío, 3 semillas)
  hp_curvas.png            QE por época para los esquemas de eta y de radio
  hp_tabla.csv             tabla resumen de la ablación
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.kohonen import Kohonen, KohonenConfig
from src.plots import save, heatmap
from src.preprocessing import Standardizer

OUT = Path("resultados/kohonen")
df = pd.read_csv("datos/creadores.csv")
sel = json.load(open("resultados/eda/variables_seleccionadas.json"))["selected"]
tema = pd.read_csv("resultados/eda/tema_dominante.csv")["tema_dominante"].values
X = Standardizer().fit_transform(df[sel].values)
P, n = X.shape
final = json.load(open("configs/kohonen.json"))
raw = pd.read_csv(OUT / "busqueda_raw.csv")
raw["radius0"] = raw["radius0"].astype(str)
SEEDS = sorted(raw.seed.unique())

def is_final(r, skip):
    """fila del barrido que coincide con la config final salvo en `skip`"""
    cond = {"k": r.k == final["k"], "radius0": r.radius0 == "k",
            "eta0": r.eta0 == final["eta0"], "eta_schedule": r.eta_schedule == final["eta_schedule"],
            "epochs": r.epochs == final["epochs"], "init": r.init == final["init"]}
    return all(v for key, v in cond.items() if key != skip)

# ================================================================ 1. efecto de k
sub = raw[[is_final(r, "k") for r in raw.itertuples()]]
g = sub.groupby("k").agg(QE=("QE", "mean"), QE_sd=("QE", "std"), TE=("TE", "mean"), TE_sd=("TE", "std"),
                         dead=("dead", "mean"), dead_sd=("dead", "std"))
g["neuronas"] = g.index ** 2
g["creadores_por_neurona_activa"] = P / (g.neuronas - g.dead)
g["muertas_pct"] = 100 * g.dead / g.neuronas
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
ks = [str(k) for k in g.index]
axes[0].bar(ks, g.QE, yerr=g.QE_sd, capsize=4, color="C0"); axes[0].set_title("Error de cuantización (QE)")
axes[0].set_ylim(g.QE.min() * 0.85, g.QE.max() * 1.05)
axes[1].bar(ks, g.TE, yerr=g.TE_sd, capsize=4, color="C3"); axes[1].set_title("Error topográfico (TE)"); axes[1].set_ylim(bottom=0)
axes[2].bar(ks, g.muertas_pct, yerr=100 * g.dead_sd / g.neuronas, capsize=4, color="gray")
axes[2].set_title("Unidades muertas (% de la grilla)"); axes[2].set_ylim(bottom=0)
for i, (k, row) in enumerate(g.iterrows()):
    axes[2].text(i, row.muertas_pct + 1, f"{row.dead:.1f} de {row.neuronas}", ha="center", fontsize=8)
axes[3].bar(ks, g.creadores_por_neurona_activa, color="C2"); axes[3].set_title("Creadores por neurona activa")
axes[3].axhline(3, color="k", ls="--", lw=0.8)
for ax in axes:
    ax.set_xlabel("k (grilla k×k)")
fig.suptitle(f"Tamaño de la grilla con el resto de la configuración final (R(0)=k→1, η {final['eta0']}→{final['eta_final']} lineal, "
             f"{final['epochs']} épocas, init muestras; media ± desvío sobre {len(SEEDS)} semillas)")
save(fig, OUT / "hp_k.png")
g.round(3).to_csv(OUT / "hp_k.csv")
print(g.round(3).to_string())

# mapas reales para cada k
fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
for ax, k in zip(axes, [3, 4, 5, 6]):
    cfg = KohonenConfig(**{**final, "k": k, "radius0": float(k)})
    som = Kohonen(n, cfg).train(X)
    hits = som.hits(X)
    heatmap(hits, title=f"k={k}: QE={som.quantization_error(X):.2f}  TE={som.topographic_error(X):.3f}  "
                        f"muertas={som.dead_units(X)}/{k * k}", annot=True, fmt="{:.0f}", cmap="YlOrBr",
            vmin=0, vmax=max(hits.max(), 1), ax=ax)
    for (r, c), v in np.ndenumerate(hits):          # unidades muertas marcadas con una cruz
        if v == 0:
            ax.text(c, r, "✕", ha="center", va="center", fontsize=18, color="C3")
fig.suptitle("Creadores por neurona según el tamaño de la grilla (config final, semilla fija)")
save(fig, OUT / "hp_mapas_k.png")

# ================================================= 2. ablación: un hiperparámetro a la vez
hps = {"radius0": ("R(0)", {"1.0": "1 (const.)", "2.0": "2 → 1", "k": "k → 1"}),
       "eta0": ("η(0)", {0.1: "0.1", 0.5: "0.5", 0.9: "0.9"}),
       "eta_schedule": ("esquema de η", {"inverse": "η₀/(1+ép.)", "linear": "lineal"}),
       "epochs": ("épocas", {100: "100", 500: "500"}),
       "init": ("inicialización", {"samples": "muestras", "uniform": "uniforme"})}
rows = []
fig, axes = plt.subplots(3, len(hps), figsize=(3.2 * len(hps), 8.5), sharey="row")
for j, (hp, (label, names)) in enumerate(hps.items()):
    sub = raw[[is_final(r, hp) for r in raw.itertuples()]]
    g = sub.groupby(hp).agg(QE=("QE", "mean"), QE_sd=("QE", "std"), TE=("TE", "mean"), TE_sd=("TE", "std"),
                            dead=("dead", "mean"), dead_sd=("dead", "std"))
    g = g.loc[[v for v in names if v in g.index]]
    xs = [names[v] for v in g.index]
    finalval = {"radius0": "k", "eta0": final["eta0"], "eta_schedule": final["eta_schedule"],
                "epochs": final["epochs"], "init": final["init"]}[hp]
    cols = ["C0" if v == finalval else "lightgray" for v in g.index]
    for i, (m, c) in enumerate([("QE", "C0"), ("TE", "C3"), ("dead", "gray")]):
        ax = axes[i, j]
        ax.bar(xs, g[m], yerr=g[f"{m}_sd"], capsize=4, color=[c if v == finalval else "lightgray" for v in g.index],
               edgecolor="k", lw=0.5)
        for xi, val in enumerate(g[m]):
            ax.text(xi, val, f"{val:.3f}" if m != "dead" else f"{val:.1f}", ha="center", va="bottom", fontsize=7)
        if i == 0:
            ax.set_title(label)
        if j == 0:
            ax.set_ylabel({"QE": "QE", "TE": "TE", "dead": "unidades muertas"}[m])
        ax.tick_params(axis="x", labelsize=8)
    for v in g.index:
        rows.append(dict(hiperparametro=hp, valor=v, es_final=(v == finalval), **g.loc[v].round(4).to_dict()))
axes[0, 0].set_ylim(2.9, 3.4)
for ax in list(axes[1]) + list(axes[2]):
    ax.set_ylim(bottom=0)
fig.suptitle(f"Ablación: se cambia UN hiperparámetro de la configuración final (en color) y se mantienen los demás\n"
             f"k={final['k']}; media ± desvío sobre {len(SEEDS)} semillas")
save(fig, OUT / "hp_ablacion.png")
tab = pd.DataFrame(rows)
tab.to_csv(OUT / "hp_tabla.csv", index=False)
print(tab.to_string(index=False))

# ======================================== 2b. épocas con 10 semillas (3 no alcanzan)
rows = []
for ep in [50, 100, 200, 300, 500, 1000]:
    for sd in range(10):
        som = Kohonen(n, KohonenConfig(**{**final, "epochs": ep, "seed": sd})).train(X)
        rows.append(dict(epochs=ep, seed=sd, QE=som.quantization_error(X), TE=som.topographic_error(X), dead=som.dead_units(X)))
ep_df = pd.DataFrame(rows)
ge = ep_df.groupby("epochs").agg(QE=("QE", "mean"), QE_sd=("QE", "std"), TE=("TE", "mean"), TE_sd=("TE", "std"),
                                 dead=("dead", "mean"), semillas_con_muertas=("dead", lambda x: int((x > 0).sum())))
ge.round(4).to_csv(OUT / "hp_epocas.csv")
print(ge.round(4).to_string())
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
xs = [str(e) for e in ge.index]
cols = ["C0" if e == final["epochs"] else "lightgray" for e in ge.index]
axes[0].bar(xs, ge.QE, yerr=ge.QE_sd, capsize=4, color=cols, edgecolor="k", lw=0.5); axes[0].set_ylim(3.0, 3.25); axes[0].set_title("QE")
axes[1].bar(xs, ge.TE, yerr=ge.TE_sd, capsize=4, color=cols, edgecolor="k", lw=0.5); axes[1].set_ylim(bottom=0); axes[1].set_title("TE")
axes[2].bar(xs, ge.semillas_con_muertas, color=cols, edgecolor="k", lw=0.5); axes[2].set_title("semillas (de 10) con 1 unidad muerta")
for ax in axes:
    ax.set_xlabel("épocas")
fig.suptitle("Cantidad de épocas con 10 semillas: a partir de 100 las diferencias están dentro del ruido entre semillas")
save(fig, OUT / "hp_epocas.png")

# ================================================================= 3. curvas de QE
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for es, lab in [("inverse", "η = η₀/(1+época)"), ("linear", "η lineal η₀ → 0.01")]:
    for e0 in [0.1, 0.5, 0.9]:
        cfg = KohonenConfig(**{**final, "eta0": e0, "eta_schedule": es})
        som = Kohonen(n, cfg).train(X)
        axes[0].plot(som.history.quantization_error, ls="-" if es == "linear" else "--",
                     label=f"{lab}, η₀={e0}  (final {som.history.quantization_error[-1]:.2f})")
axes[0].set_xlabel("época"); axes[0].set_ylabel("QE"); axes[0].legend(fontsize=7)
axes[0].set_title("Tasa de aprendizaje: el esquema inverso se estanca a las ~50 épocas")
for r0, lab in [(1.0, "R = 1 constante"), (2.0, "R(0) = 2 → 1"), (float(final["k"]), f"R(0) = k = {final['k']} → 1")]:
    cfg = KohonenConfig(**{**final, "radius0": r0, "radius_schedule": "constant" if r0 == 1.0 else "linear"})
    som = Kohonen(n, cfg).train(X)
    axes[1].plot(som.history.quantization_error,
                 label=f"{lab}  (QE {som.quantization_error(X):.2f}, TE {som.topographic_error(X):.3f}, muertas {som.dead_units(X)})")
axes[1].set_xlabel("época"); axes[1].set_ylabel("QE"); axes[1].legend(fontsize=7)
axes[1].set_title("Radio: con R(0) grande el QE cae en escalones al achicarse el vecindario")
save(fig, OUT / "hp_curvas.png")
