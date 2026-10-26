"""Ejercicio 1.2 - Red de Kohonen sobre los creadores (variables seleccionadas en el EDA).

Uso:
    python3 ej1_kohonen.py                 # barrido de hiperparámetros + modelo final
    python3 ej1_kohonen.py --skip-search   # sólo el modelo final (configs/kohonen.json)

Salidas en resultados/kohonen/:
  - busqueda.csv / busqueda_*.png   métricas de cada configuración (QE, TE, unidades muertas)
  - hits.png                        cantidad de creadores por neurona (+ nombres)
  - umatrix.png                     Matriz U
  - planos_*.png                    valor medio de cada variable por neurona
  - perfiles_neuronas.csv           perfil promedio (z-score) de cada neurona
  - asignacion.csv                  neurona de cada creador + PC1/PC2
  - som_vs_pca.png                  mapa coloreado por PC1 y PC2 (comparación con biplot)
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.kohonen import Kohonen, KohonenConfig
from src.plots import save, heatmap
from src.preprocessing import Standardizer

OUT = Path("resultados/kohonen")
OUT.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--skip-search", action="store_true")
ap.add_argument("--config", default="configs/kohonen.json")
args = ap.parse_args()

df = pd.read_csv("datos/creadores.csv")
sel = json.load(open("resultados/eda/variables_seleccionadas.json"))["selected"]
tema = pd.read_csv("resultados/eda/tema_dominante.csv")["tema_dominante"].values
proj = pd.read_csv("resultados/pca/proyeccion.csv", index_col=0)
names = df["name"].values
scaler = Standardizer()
X = scaler.fit_transform(df[sel].values)       # "Importante: estandarizar todos los vectores"
P, n = X.shape

# ======================================================= 1. barrido de hiperparámetros
if not args.skip_search:
    grid = {
        "k": [3, 4, 5, 6],
        "radius0": ["k", 2.0, 1.0],          # toda la grilla / chico / constante 1
        "eta0": [0.1, 0.5, 0.9],
        "eta_schedule": ["inverse", "linear"],   # eta0/(1+epoca)  vs  lineal hasta eta_final
        "epochs": [100, 500],
        "init": ["samples", "uniform"],
    }
    seeds = [0, 1, 2]
    rows = []
    t0 = time.time()
    for k, r0, e0, es, ep, init in itertools.product(*grid.values()):
        R0 = float(k) if r0 == "k" else r0
        for s in seeds:
            cfg = KohonenConfig(k=k, radius0=R0, radius_final=1.0,
                                radius_schedule="constant" if R0 == 1.0 else "linear",
                                eta0=e0, eta_schedule=es, eta_final=0.01,
                                epochs=ep, init=init, seed=s)
            som = Kohonen(n, cfg).train(X)
            rows.append(dict(k=k, radius0=r0, eta0=e0, eta_schedule=es, epochs=ep, init=init, seed=s,
                             QE=som.quantization_error(X), TE=som.topographic_error(X),
                             dead=som.dead_units(X), umean=float(som.u_matrix().mean())))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "busqueda_raw.csv", index=False)
    agg = res.groupby(["k", "radius0", "eta0", "eta_schedule", "epochs", "init"]).agg(
        QE=("QE", "mean"), QE_std=("QE", "std"), TE=("TE", "mean"), dead=("dead", "mean"),
        umean=("umean", "mean")).reset_index()
    agg.round(4).to_csv(OUT / "busqueda.csv", index=False)
    print(f"búsqueda: {len(res)} entrenamientos en {time.time() - t0:.0f}s")
    print("\nMejores por QE:\n", agg.sort_values("QE").head(8).round(3).to_string(index=False))
    print("\nMejores por TE (con QE):\n", agg.sort_values(["TE", "QE"]).head(8).round(3).to_string(index=False))

    # (las figuras de hiperparámetros las genera ej1_kohonen_hp.py a partir de busqueda_raw.csv)

# ============================================================ 2. modelo final
cfg_json = json.load(open(args.config))
cfg = KohonenConfig(**cfg_json)
print("\nConfiguración final:", cfg)
som = Kohonen(n, cfg).train(X, verbose=True)
k = cfg.k
w = som.winners(X)
rows, cols = som.idx_to_rc(w)
print(f"QE={som.quantization_error(X):.4f}  TE={som.topographic_error(X):.4f}  "
      f"unidades muertas={som.dead_units(X)}  U media={som.u_matrix().mean():.3f}")

# curva de entrenamiento
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(som.history.quantization_error); ax.set_xlabel("época"); ax.set_ylabel("QE")
ax2 = ax.twinx(); ax2.plot(som.history.radius, color="C3", alpha=0.6); ax2.set_ylabel("R(i)", color="C3")
ax.set_title("Error de cuantización durante el entrenamiento")
save(fig, OUT / "entrenamiento.png")

# ---- hits + nombres
hits = som.hits(X)
annot = [["" for _ in range(k)] for _ in range(k)]
for i, name in enumerate(names):
    r, c = rows[i], cols[i]
    annot[r][c] += ("\n" if annot[r][c] else "") + name.replace("@", "")
fig, ax = plt.subplots(figsize=(2.6 * k + 1, 2.4 * k + 1))
heatmap(hits, title=f"Creadores por neurona (grilla {k}x{k})", annot=annot, cmap="YlOrBr",
        cbar_label="cantidad", vmin=0, ax=ax)
for (r, c), v in np.ndenumerate(hits):
    if v == 0:
        ax.text(c, r, "✕ (sin creadores)", ha="center", va="center", fontsize=9, color="C3")
save(fig, OUT / "hits.png")

# ---- hits por tema dominante (para interpretar regiones)
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, t in zip(axes, ["football", "dogs", "makeup", "art"]):
    m = tema == t
    H = np.bincount(w[m], minlength=k * k).reshape(k, k)
    heatmap(H, title=f"tema dominante: {t}", annot=True, fmt="{:.0f}", cmap="Blues", ax=ax)
save(fig, OUT / "hits_por_tema.png")

# ---- Matriz U
U = som.u_matrix(neighbors=8)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
heatmap(U, title="Matriz U (distancia media a vecinas, 8 vecinos)", annot=True, cmap="gray_r",
        cbar_label="distancia euclídea en el espacio estandarizado", ax=ax)
# superponer cantidad de hits como puntos
for r in range(k):
    for c in range(k):
        if hits[r, c]:
            ax.scatter(c, r + 0.33, s=18 * hits[r, c], color="C3", alpha=0.6, edgecolor="k", lw=0.4)
save(fig, OUT / "umatrix.png")

# ---- planos de componentes (promedio de la variable estandarizada por neurona)
groups = {
    "escala": ["average_impressions", "average_video_views", "price_usd", "non_follower_reach_pct", "monthly_posts", "argentina_audience_pct"],
    "engagement": ["likes_per_1000", "comments_per_1000", "saves_per_1000", "shares_per_1000", "profile_visits_per_1000", "average_watch_time_seconds"],
    "tema": ["topic_football_pct", "topic_makeup_pct", "topic_dogs_pct", "topic_art_pct", "sponsored_posts_pct", "trend_posts_pct"],
    "formato": ["content_tutorial_pct", "content_review_pct", "content_entertainment_pct", "content_showcase_pct", "video_posts_pct", "short_form_posts_pct"],
}
for gname, feats in groups.items():
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, f in zip(axes.ravel(), [f for f in feats if f in sel]):
        M = som.component_plane(sel.index(f))
        heatmap(M, title=f, annot=True, cmap="RdBu_r", vmin=-2, vmax=2, ax=ax)
    fig.suptitle(f"Pesos de la red por variable ({gname}; z-score, rojo = alto)")
    save(fig, OUT / f"planos_{gname}.png")

# ---- perfiles por neurona (z-score promedio de los creadores asignados)
prof = pd.DataFrame(X, columns=sel)
prof["neurona"] = [f"({r},{c})" for r, c in zip(rows, cols)]
perfiles = prof.groupby("neurona").mean()
perfiles["n"] = prof.groupby("neurona").size()
perfiles.round(2).to_csv(OUT / "perfiles_neuronas.csv")

asig = pd.DataFrame({"name": names, "tema": tema, "fila": rows, "col": cols,
                     "followers": df.followers.values, "price_usd": df.price_usd.values,
                     "PC1": proj.loc[names, "PC1"].values, "PC2": proj.loc[names, "PC2"].values})
asig.sort_values(["fila", "col"]).to_csv(OUT / "asignacion.csv", index=False)

# resumen textual por neurona: top-3 variables altas y bajas
with open(OUT / "resumen_neuronas.txt", "w") as f:
    for neu, row in perfiles.iterrows():
        z = row.drop("n").sort_values()
        f.write(f"neurona {neu}  n={int(row.n)}  "
                f"creadores={', '.join(asig[(asig.fila == int(neu[1])) & (asig.col == int(neu[3]))].name)}\n")
        f.write(f"   altas: {', '.join(f'{v} ({z[v]:+.1f})' for v in z.index[::-1][:4])}\n")
        f.write(f"   bajas: {', '.join(f'{v} ({z[v]:+.1f})' for v in z.index[:4])}\n")
print(open(OUT / "resumen_neuronas.txt").read())

# ---- comparación con PCA: promedio de PC1 y PC2 por neurona
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, pc in zip(axes, ["PC1", "PC2"]):
    M = np.full(k * k, np.nan)
    for j in range(k * k):
        m = w == j
        if m.any():
            M[j] = asig[pc].values[m].mean()
    heatmap(M.reshape(k, k), title=f"{pc} promedio por neurona", annot=True, cmap="RdBu_r", ax=ax)
fig.suptitle("¿El mapa reproduce la estructura del biplot?")
save(fig, OUT / "som_vs_pca.png")
