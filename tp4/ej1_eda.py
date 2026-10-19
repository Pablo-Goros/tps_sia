"""Ejercicio 1.1 - Exploratory Data Analysis y selección de variables.

Salidas en resultados/eda/:
  - describe.csv                estadísticos descriptivos
  - boxplot_crudo.png           escalas originales (muy distintas)
  - boxplot_estandarizado.png   luego de z-score
  - correlacion.png             matriz de correlaciones
  - derivadas.txt               verificación de variables derivadas exactas
  - variables_seleccionadas.json  lista final usada por PCA / Kohonen / Oja
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.plots import save, heatmap
from src.preprocessing import Standardizer

DATA = Path("datos/creadores.csv")
OUT = Path("resultados/eda")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
num = df.drop(columns=["creator_id", "name"])
print(f"{len(df)} creadores, {num.shape[1]} variables numéricas")

# ------------------------------------------------------------------ describe
desc = num.describe().T
desc["cv"] = desc["std"] / desc["mean"]
desc.round(3).to_csv(OUT / "describe.csv")
print(desc[["mean", "std", "min", "max"]].round(2).to_string())

# --------------------------------------------------- variables derivadas exactas
lines = []
def check(name, ok, detail=""):
    lines.append(f"{name}: {'REDUNDANTE' if ok else 'no exacta'} {detail}")
check("price_ars = 1500 * price_usd", np.allclose(df.price_ars, np.round(df.price_usd * 1500)))
check("organic_posts_pct = 100 - sponsored_posts_pct", np.allclose(df.organic_posts_pct, 100 - df.sponsored_posts_pct))
check("total_engagement_per_1000 = likes + comments + saves + shares",
      np.allclose(df.total_engagement_per_1000,
                  df[["likes_per_1000", "comments_per_1000", "saves_per_1000", "shares_per_1000"]].sum(axis=1)))
err = np.abs(df.estimated_argentina_reach - df.average_reach * df.argentina_audience_pct / 100).max()
check("estimated_argentina_reach = average_reach * argentina_audience_pct / 100", err < 1, f"(error max {err:.2f}, redondeo)")
topics = [c for c in num if c.startswith("topic_")]
contents = [c for c in num if c.startswith("content_")]
check("suma topic_*_pct = 100", np.allclose(num[topics].sum(axis=1), 100))
check("suma content_*_pct = 100", np.allclose(num[contents].sum(axis=1), 100))
(OUT / "derivadas.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# ------------------------------------------------------------------ boxplots
fig, ax = plt.subplots(figsize=(14, 5))
ax.boxplot([num[c] for c in num], tick_labels=num.columns)
ax.set_yscale("symlog")
ax.set_title("Variables originales (escala simétrica-log): escalas incomparables")
plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
save(fig, OUT / "boxplot_crudo.png")

Z = pd.DataFrame(Standardizer().fit_transform(num.values), columns=num.columns)
fig, ax = plt.subplots(figsize=(14, 5))
ax.boxplot([Z[c] for c in Z], tick_labels=Z.columns)
ax.set_title("Variables estandarizadas (z-score)")
plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
save(fig, OUT / "boxplot_estandarizado.png")

# --------------------------------------------------------------- correlación
corr = num.corr()
fig, ax = heatmap(corr.values, title="Matriz de correlaciones (todas las variables)",
                  cmap="coolwarm", vmin=-1, vmax=1, figsize=(12, 10),
                  xticks=corr.columns, yticks=corr.columns)
save(fig, OUT / "correlacion.png")

pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
pairs = pairs.reindex(pairs.abs().sort_values(ascending=False).index)
pairs.head(25).round(3).to_csv(OUT / "pares_mas_correlacionados.csv", header=["r"])

# -------------------------------------------------------- tema dominante (EDA)
dom = num[topics].idxmax(axis=1).str.replace("topic_", "").str.replace("_pct", "")
df["tema_dominante"] = dom
print("\nTema dominante por creador:\n", dom.value_counts().to_string())
df[["creator_id", "name", "tema_dominante"]].to_csv(OUT / "tema_dominante.csv", index=False)

# ------------------------------------------------------ selección de variables
# Se eliminan únicamente las columnas que son combinaciones EXACTAS de otras:
# aportan información redundante (correlación 1 con su original) y nada más.
drop = ["price_ars", "organic_posts_pct", "total_engagement_per_1000", "estimated_argentina_reach"]
selected = [c for c in num.columns if c not in drop]
json.dump({"selected": selected, "dropped": drop,
           "note": "topic_* y content_* suman 100 cada grupo: se conservan las 4 de cada uno por "
                   "interpretabilidad; eso introduce 2 autovalores nulos en PCA (ver DECISIONES.md)."},
          open(OUT / "variables_seleccionadas.json", "w"), indent=2, ensure_ascii=False)
print(f"\nVariables seleccionadas ({len(selected)}):", selected)

# Correlación del subconjunto seleccionado
corr_s = num[selected].corr()
fig, ax = heatmap(corr_s.values, title=f"Correlaciones - {len(selected)} variables seleccionadas",
                  cmap="coolwarm", vmin=-1, vmax=1, figsize=(11, 9),
                  xticks=corr_s.columns, yticks=corr_s.columns)
save(fig, OUT / "correlacion_seleccionadas.png")
