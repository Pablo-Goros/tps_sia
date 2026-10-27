"""Ejercicio 1.3 - Regla de Oja: índice de creadores y comparación con PC1.

Salidas en resultados/oja/:
  - barrido_eta.csv / barrido_eta.png   convergencia (|cos(w, PC1)|, ||w||) para distintos eta
  - convergencia.png                    modelo final
  - pesos_vs_pc1.png                    w final vs autovector PC1 (por variable)
  - indice_oja.csv / indice_oja_vs_pc1.png   índice por creador vs proyección PCA
  - ranking_por_tema.csv                ranking dentro de cada tema (para campañas)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.oja import Oja, OjaConfig, cosine
from src.plots import save
from src.preprocessing import Standardizer

OUT = Path("resultados/oja")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("datos/creadores.csv")
sel = json.load(open("resultados/eda/variables_seleccionadas.json"))["selected"]
tema = pd.read_csv("resultados/eda/tema_dominante.csv")["tema_dominante"].values
names = df["name"].values
X = Standardizer().fit_transform(df[sel].values)          # media 0 (requisito del algoritmo)
pc1 = np.load("resultados/pca/pc1_autovector.npy")       # autovector de referencia (PCA exacto)
proj_pc1 = pd.read_csv("resultados/pca/proyeccion.csv", index_col=0).loc[names, "PC1"].values
n = X.shape[1]
lambda1 = float(pd.read_csv("resultados/pca/autovalores.csv", index_col=0).autovalor.iloc[0])
print(f"lambda_1 = {lambda1:.3f}  ->  eta0 teórico 1/(1.2 lambda_1) = {1 / (1.2 * lambda1):.4f}")

# ---------------------------------------------- 1. barrido de tasa de aprendizaje
settings = [
    ("constante 1e-4", OjaConfig(eta0=1e-4, eta_schedule="constant", epochs=300)),
    ("constante 1e-3", OjaConfig(eta0=1e-3, eta_schedule="constant", epochs=300)),
    ("constante 1e-2", OjaConfig(eta0=1e-2, eta_schedule="constant", epochs=300)),
    ("constante 5e-2", OjaConfig(eta0=5e-2, eta_schedule="constant", epochs=300)),
    ("1/(1.2*lambda1)=%.3f" % (1 / (1.2 * lambda1)), OjaConfig(eta0=1 / (1.2 * lambda1), eta_schedule="constant", epochs=300)),
    ("0.5 -> 1/(1+epoca)", OjaConfig(eta0=0.5, eta_schedule="inverse", eta_final=1e-4, epochs=300)),
    ("0.1 -> 1/(1+epoca)", OjaConfig(eta0=0.1, eta_schedule="inverse", eta_final=1e-4, epochs=300)),
]
rows = []
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for label, cfg in settings:
    cos_hist, norm_hist = [], []
    for seed in (0, 1, 2, 3, 4):
        cfg.seed = seed
        m = Oja(n, cfg).train(X, reference=pc1)
        cos_hist.append(m.history.cos_to_ref)
        norm_hist.append(m.history.norm)
        rows.append(dict(eta=label, seed=seed, cos_final=m.history.cos_to_ref[-1],
                         norm_final=m.history.norm[-1],
                         epocas_hasta_0_99=int(np.argmax(np.array(m.history.cos_to_ref) > 0.99))
                         if max(m.history.cos_to_ref) > 0.99 else -1))
    c, nn = np.mean(cos_hist, axis=0), np.mean(norm_hist, axis=0)
    axes[0].plot(c, label=label)
    axes[1].plot(nn, label=label)
axes[0].set_xscale("log"); axes[0].set_ylim(0.5, 1.01)
axes[0].set_xlabel("época"); axes[0].set_ylabel("|cos(w, autovector PC1)|"); axes[0].set_title("Alineación con PC1")
axes[1].set_xscale("log"); axes[1].axhline(1, color="gray", ls="--", lw=0.8)
axes[1].set_xlabel("época"); axes[1].set_ylabel("||w||"); axes[1].set_title("Norma de w (debe tender a 1)")
axes[0].legend(fontsize=7)
save(fig, OUT / "barrido_eta.png")
res = pd.DataFrame(rows)
res["diverge"] = res.cos_final.isna()
bar = res.groupby("eta").agg(cos_final=("cos_final", "mean"), cos_min=("cos_final", "min"),
                                              norm_final=("norm_final", "mean"),
                                              epocas_0_99=("epocas_hasta_0_99", "mean"), diverge=("diverge", "mean"))
bar.round(4).to_csv(OUT / "barrido_eta.csv")
print(bar.round(4).to_string())

# -------------------------------------------------------------- 2. modelo final
cfg = OjaConfig(**json.load(open("configs/oja.json")))
oja = Oja(n, cfg).train(X, reference=pc1)
w = oja.w
# El autovector y su opuesto son equivalentes: alineamos el signo con PC1 para comparar.
if cosine(w, pc1) < 0:
    w = -w
print(f"\nconfig final: {cfg}")
print(f"cos(w, PC1) = {cosine(w, pc1):.6f}   ||w|| = {np.linalg.norm(w):.6f}")
print(f"max |w_i - a_i| = {np.abs(w - pc1).max():.4f}")

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(oja.history.cos_to_ref, label="|cos(w, PC1)|")
ax.plot(oja.history.norm, label="||w||")
ax.set_xlabel("época"); ax.legend(); ax.set_title(f"Convergencia de la regla de Oja (eta={cfg.eta0})")
save(fig, OUT / "convergencia.png")

cmp = pd.DataFrame({"oja_w": w, "pca_pc1": pc1}, index=sel)
cmp["diferencia"] = cmp.oja_w - cmp.pca_pc1
cmp.round(4).to_csv(OUT / "pesos_vs_pc1.csv")
fig, ax = plt.subplots(figsize=(12, 4.5))
xs = np.arange(n)
ax.bar(xs - 0.2, w, 0.4, label="Oja (w final)")
ax.bar(xs + 0.2, pc1, 0.4, label="PCA (autovector 1)")
ax.set_xticks(xs); ax.set_xticklabels(sel, rotation=90, fontsize=7)
ax.axhline(0, color="k", lw=0.6); ax.legend(); ax.set_title("Cargas: Oja vs PCA")
save(fig, OUT / "pesos_vs_pc1.png")

# -------------------------------------------------------------- 3. el índice
idx = oja.project(X) if cosine(oja.w, pc1) > 0 else -oja.project(X)
tabla = pd.DataFrame({"name": names, "tema": tema, "followers": df.followers.values,
                      "price_usd": df.price_usd.values, "indice_oja": idx, "pc1_pca": proj_pc1})
tabla["rank_oja"] = tabla.indice_oja.rank(ascending=False).astype(int)
tabla["rank_followers"] = tabla.followers.rank(ascending=False).astype(int)
tabla["rank_pc1"] = tabla.pc1_pca.rank(ascending=False).astype(int)
tabla.sort_values("indice_oja", ascending=False).round(3).to_csv(OUT / "indice_oja.csv", index=False)
print("\nTop 10 por índice de Oja:\n", tabla.sort_values("indice_oja", ascending=False).head(10)
      [["name", "tema", "followers", "price_usd", "indice_oja", "pc1_pca", "rank_followers"]].round(2).to_string(index=False))
print("\nÚltimos 5:\n", tabla.sort_values("indice_oja").head(5)
      [["name", "tema", "followers", "price_usd", "indice_oja", "pc1_pca"]].round(2).to_string(index=False))
rho = np.corrcoef(idx, proj_pc1)[0, 1]
rho_f = tabla[["rank_oja", "rank_followers"]].corr(method="spearman").iloc[0, 1]
print(f"\ncorrelación índice Oja vs PC1: {rho:.5f}   |  Spearman(rank Oja, rank followers) = {rho_f:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = {"football": "C2", "dogs": "C1", "makeup": "C6", "art": "C0"}
for t, c in colors.items():
    m = tema == t
    axes[0].scatter(proj_pc1[m], idx[m], c=c, label=t)
    axes[1].scatter(df.followers.values[m], idx[m], c=c, label=t)
lim = [min(proj_pc1.min(), idx.min()), max(proj_pc1.max(), idx.max())]
axes[0].plot(lim, lim, "k--", lw=0.8); axes[0].set_xlabel("PC1 (PCA)"); axes[0].set_ylabel("índice (Oja)")
axes[0].set_title(f"Oja vs PCA (r = {rho:.4f})"); axes[0].legend()
axes[1].set_xscale("log"); axes[1].set_xlabel("followers (log)"); axes[1].set_ylabel("índice (Oja)")
axes[1].set_title("Índice vs cantidad de seguidores")
save(fig, OUT / "indice_oja_vs_pc1.png")

# ranking dentro de cada tema: ¿elegir el máximo índice sirve para cualquier campaña?
top = (tabla.sort_values("indice_oja", ascending=False).groupby("tema").head(3)
       [["tema", "name", "indice_oja", "followers", "price_usd"]].sort_values(["tema", "indice_oja"], ascending=[True, False]))
top.round(2).to_csv(OUT / "ranking_por_tema.csv", index=False)
print("\nTop-3 por tema según índice:\n", top.round(2).to_string(index=False))
