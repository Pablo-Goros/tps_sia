"""Ejercicio 1.1 - PCA sobre las variables seleccionadas en el EDA.

Salidas en resultados/pca/:
  - autovalores.csv / scree.png       varianza explicada por componente
  - cargas.csv / cargas_pc1_pc2.png   loadings de las primeras componentes
  - biplot.png                        creadores en (PC1, PC2) + vectores de cargas
  - indice_pc1.png / indice_pc1.csv   PC1 como índice para ordenar creadores
  - proyeccion.csv                    coordenadas en PC1..PC3 (para comparar con Kohonen/Oja)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.pca import PCA
from src.plots import save

OUT = Path("resultados/pca")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("datos/creadores.csv")
sel = json.load(open("resultados/eda/variables_seleccionadas.json"))["selected"]
tema = pd.read_csv("resultados/eda/tema_dominante.csv")["tema_dominante"].values
X = df[sel].values
names = df["name"].values

pca = PCA().fit(X)
# Orientación de signo (ver DECISIONES.md): PC1 con average_impressions positivo
# ("más PC1 = más visibilidad"), PC2 con topic_art_pct positivo.
pca.orient(0, sel.index("average_impressions")).orient(1, sel.index("topic_art_pct"))
Y = pca.transform(X)
evr = pca.explained_variance_ratio_
cum = pca.cumulative_variance()

# ------------------------------------------------------------ autovalores
tab = pd.DataFrame({"autovalor": pca.eigenvalues_, "var_explicada": evr, "acumulada": cum},
                   index=[f"PC{i+1}" for i in range(len(evr))])
tab.round(4).to_csv(OUT / "autovalores.csv")
print(tab.head(10).round(3).to_string())
for thr in (0.7, 0.8, 0.9):
    print(f"componentes para {thr:.0%} de la varianza: {pca.n_components_for(thr)}")
print("autovalores ~0 (redundancia exacta):", int((pca.eigenvalues_ < 1e-8).sum()))

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(1, len(evr) + 1), evr, label="varianza explicada")
ax.plot(range(1, len(evr) + 1), cum, "o-", color="C3", label="acumulada")
ax.axhline(0.8, ls="--", color="gray", lw=0.8)
ax.set_xlabel("componente"); ax.set_ylabel("proporción de varianza"); ax.legend()
ax.set_title("Scree plot (PCA sobre matriz de correlaciones)")
save(fig, OUT / "scree.png")

# ------------------------------------------------------------------ cargas
load = pd.DataFrame(pca.components_[:, :4], index=sel, columns=["PC1", "PC2", "PC3", "PC4"])
load.round(4).to_csv(OUT / "cargas.csv")
print("\nCargas PC1 (ordenadas):\n", load["PC1"].sort_values().round(3).to_string())
print("\nCargas PC2 (ordenadas):\n", load["PC2"].sort_values().round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 7))
for ax, pc in zip(axes, ["PC1", "PC2"]):
    s = load[pc].sort_values()
    ax.barh(s.index, s.values, color=np.where(s.values > 0, "C0", "C3"))
    ax.set_title(f"Cargas de {pc} ({evr[int(pc[-1]) - 1]:.1%} de la varianza)")
    ax.axvline(0, color="k", lw=0.8)
    ax.tick_params(axis="y", labelsize=7)
save(fig, OUT / "cargas_pc1_pc2.png")

# ------------------------------------------------------------------ biplot
colors = {"football": "C2", "dogs": "C1", "makeup": "C6", "art": "C0"}
fig, ax = plt.subplots(figsize=(11, 9))
for t, c in colors.items():
    m = tema == t
    ax.scatter(Y[m, 0], Y[m, 1], c=c, label=f"tema dominante: {t}", s=40, alpha=0.85)
for i, n in enumerate(names):
    ax.annotate(n, (Y[i, 0], Y[i, 1]), fontsize=6, alpha=0.8, xytext=(3, 2), textcoords="offset points")
scale = np.abs(Y[:, :2]).max() * 0.9
TURQ = "#20B2AA"
ends = np.column_stack([load.PC1.values, load.PC2.values]) * scale
for ex, ey in ends:
    ax.plot([0, ex], [0, ey], color=TURQ, lw=1.1, alpha=0.9)

# Etiquetas al final de cada línea, separadas para que no se superpongan:
# se ubican un poco más allá del extremo y se empujan verticalmente las que chocan.
ax.set_xlim(Y[:, 0].min() - 0.8, Y[:, 0].max() + 1.2)
ax.set_ylim(Y[:, 1].min() - 0.6, Y[:, 1].max() + 0.6)
fig.canvas.draw()
y_per_px = (ax.get_ylim()[1] - ax.get_ylim()[0]) / ax.bbox.height
x_per_px = (ax.get_xlim()[1] - ax.get_xlim()[0]) / ax.bbox.width
FS = 7
h = FS * 2.0 * y_per_px                       # alto aprox. de una etiqueta en datos
pos = []
for (ex, ey), v in zip(ends, sel):
    d = np.hypot(ex, ey) or 1.0
    px, py = ex + ex / d * 0.25, ey + ey / d * 0.25
    w = len(v) * FS * 0.72 * x_per_px          # ancho aprox. en datos
    ha = "left" if ex >= 0 else "right"
    pos.append([px, py, w, ha, v, ex, ey])
for _ in range(200):                           # relajación: separar pares que se solapan
    moved = False
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            a, b = pos[i], pos[j]
            ax0, ax1 = (a[0], a[0] + a[2]) if a[3] == "left" else (a[0] - a[2], a[0])
            bx0, bx1 = (b[0], b[0] + b[2]) if b[3] == "left" else (b[0] - b[2], b[0])
            if ax0 < bx1 and bx0 < ax1 and abs(a[1] - b[1]) < h:
                push = (h - abs(a[1] - b[1])) / 2 + 0.01
                if a[1] >= b[1]:
                    a[1] += push; b[1] -= push
                else:
                    a[1] -= push; b[1] += push
                moved = True
    if not moved:
        break
for px, py, w, ha, v, ex, ey in pos:
    if abs(py - ey) > h * 0.6:                 # si se desplazó, un conector fino al extremo
        ax.plot([ex, px], [ey, py], color=TURQ, lw=0.5, alpha=0.5, ls=":")
    ax.text(px, py, v, fontsize=FS, color=TURQ, ha=ha, va="center", zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel(f"PC1 ({evr[0]:.1%})"); ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
ax.set_title("Biplot: creadores y cargas en PC1-PC2"); ax.legend(fontsize=8)
save(fig, OUT / "biplot.png")

# ----------------------------------------------------- PC1 como índice
idx = pd.DataFrame({"name": names, "tema": tema, "followers": df.followers, "PC1": Y[:, 0], "PC2": Y[:, 1]})
idx.sort_values("PC1", ascending=False).round(3).to_csv(OUT / "indice_pc1.csv", index=False)
fig, ax = plt.subplots(figsize=(13, 4.5))
o = np.argsort(Y[:, 0])[::-1]
ax.bar(range(len(o)), Y[o, 0], color=[colors[t] for t in tema[o]])
ax.set_xticks(range(len(o))); ax.set_xticklabels(names[o], rotation=90, fontsize=6)
ax.set_ylabel("PC1"); ax.set_title("Índice PC1 por creador (color = tema dominante)")
save(fig, OUT / "indice_pc1.png")

pd.DataFrame(Y[:, :3], columns=["PC1", "PC2", "PC3"], index=names).to_csv(OUT / "proyeccion.csv")
np.save(OUT / "pc1_autovector.npy", pca.components_[:, 0])
