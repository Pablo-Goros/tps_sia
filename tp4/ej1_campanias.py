"""Ejercicio 1.4 - Decisión para las campañas: 4 casos (short / long) x (fútbol / perros).

Criterio, explícito y reproducible:
  1. Vertical: tema dominante del creador  (topic_<tema>_pct >= 40).
  2. Mercado local: argentina_audience_pct >= 50 (los clientes cotizan en ARS y venden/adoptan acá).
  3. Formato:
       short  (descubrimiento / awareness)  -> mayor índice de Oja (= PC1: alcance masivo + formato viral)
       long   (reseña / tutorial / conversión) -> menor índice de Oja (= profundidad: saves, comentarios,
                                                  watch time, tutoriales, captions largos)
  4. Se reporta el costo por mil impresiones (CPM = price_usd / average_impressions * 1000) para que la
     decisión incluya presupuesto, y la neurona de Kohonen en la que cae cada candidato.

Salidas en resultados/campanias/: candidatos.csv, campanias.png, resumen.txt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.plots import save, heatmap

OUT = Path("resultados/campanias")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("datos/creadores.csv").set_index("name")
asig = pd.read_csv("resultados/kohonen/asignacion.csv").set_index("name")
oja = pd.read_csv("resultados/oja/indice_oja.csv").set_index("name")
d = df.join(asig[["fila", "col"]]).join(oja[["indice_oja", "rank_followers"]])
d["cpm_usd"] = d.price_usd / d.average_impressions * 1000
K = int(asig.fila.max()) + 1

AR_MIN, TOPIC_MIN, TOP = 50, 40, 3
casos = {
    ("football", "short"): "Ropa de fútbol – lanzamiento (short: descubrimiento masivo)",
    ("football", "long"):  "Ropa de fútbol – reseña de producto (long: profundidad)",
    ("dogs", "short"):     "Refugio – campaña de visibilidad (short)",
    ("dogs", "long"):      "Refugio – consultas de adopción (long: tutorial/educativo)",
}
cols_show = ["fila", "col", "indice_oja", "followers", "rank_followers", "average_impressions", "price_usd", "cpm_usd",
             "argentina_audience_pct", "non_follower_reach_pct", "short_form_posts_pct", "content_entertainment_pct",
             "content_tutorial_pct", "content_review_pct", "average_watch_time_seconds", "saves_per_1000",
             "comments_per_1000", "sponsored_posts_pct"]

rows, picks = [], {}
log = open(OUT / "resumen.txt", "w")
def P(*a):
    print(*a); print(*a, file=log)

for (tema, fmt), titulo in casos.items():
    sub = d[(d[f"topic_{tema}_pct"] >= TOPIC_MIN) & (d.argentina_audience_pct >= AR_MIN)]
    sub = sub.sort_values("indice_oja", ascending=(fmt == "long"))
    top = sub.head(TOP)
    picks[(tema, fmt)] = list(top.index)
    P(f"\n=== {titulo} ===   (candidatos: {len(sub)} con tema >= {TOPIC_MIN}% y audiencia AR >= {AR_MIN}%)")
    P(top[cols_show].round(1).to_string())
    for i, (n, r) in enumerate(top.iterrows()):
        rows.append(dict(caso=titulo, tema=tema, formato=fmt, orden=i + 1, name=n, **r[cols_show].round(2).to_dict()))
    # qué deja afuera el filtro de audiencia (para comentar en la presentación)
    excl = d[(d[f"topic_{tema}_pct"] >= TOPIC_MIN) & (d.argentina_audience_pct < AR_MIN)].sort_values("indice_oja", ascending=(fmt == "long")).head(2)
    if len(excl):
        P("   excluidos por audiencia argentina < 50%: " +
          ", ".join(f"{n} (AR {r.argentina_audience_pct:.0f}%, índice {r.indice_oja:+.1f})" for n, r in excl.iterrows()))

# comparación con la estrategia por followers dentro de cada vertical
P("\n=== Si se eligiera solo por followers (sin filtro de audiencia) ===")
for tema in ["football", "dogs"]:
    top_f = d[d[f"topic_{tema}_pct"] >= TOPIC_MIN].sort_values("followers", ascending=False).head(TOP)
    P(f"{tema}: " + ", ".join(f"{n} ({r.followers:,.0f} seg., AR {r.argentina_audience_pct:.0f}%, índice {r.indice_oja:+.1f})"
                              for n, r in top_f.iterrows()))

pd.DataFrame(rows).to_csv(OUT / "candidatos.csv", index=False)

# ------------------------------------------------------------------ figura
light = LinearSegmentedColormap.from_list("blues_claro", plt.get_cmap("Blues")(np.linspace(0.03, 0.45, 256)))
hits = np.zeros((K, K), int)
for _, r in asig.iterrows():
    hits[int(r.fila), int(r.col)] += 1
colores = {("football", "short"): "#2ca02c", ("football", "long"): "#98df8a",
           ("dogs", "short"): "#ff7f0e", ("dogs", "long"): "#ffbb78"}
marcas = {"short": "o", "long": "s"}
fig, ax = plt.subplots(figsize=(9.5, 8))
heatmap(hits, title="Mapa de Kohonen: dónde caen los candidatos de cada campaña", cmap=light,
        annot=[[f"{v}" if v else "" for v in row] for row in hits], vmin=0, ax=ax, fontsize=10, text_color="#4a4a4a",
        cbar_label="creadores por neurona")
ax.set_xticks(range(K)); ax.set_yticks(range(K))
slots = [(-0.3, -0.22), (0.3, -0.22), (-0.3, 0.28), (0.3, 0.28)]   # posiciones fijas dentro de la celda
usados = {}
for (tema, fmt), names in picks.items():
    for i, n in enumerate(names):
        r, c = int(d.loc[n, "fila"]), int(d.loc[n, "col"])
        k_slot = usados.get((r, c), 0); usados[(r, c)] = k_slot + 1
        dx, dy = slots[k_slot % len(slots)]
        ax.scatter(c + dx, r + dy, s=260, marker=marcas[fmt], color=colores[(tema, fmt)], edgecolor="k", lw=1, zorder=4)
        ax.annotate(f"{n}\n({i + 1}º, {d.loc[n, 'price_usd']:,.0f} USD)", (c + dx, r + dy), fontsize=6.5, ha="center", va="top",
                    xytext=(0, -12), textcoords="offset points", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
for (tema, fmt), col in colores.items():
    ax.scatter([], [], s=120, marker=marcas[fmt], color=col, edgecolor="k", label=f"{tema} · {fmt}")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=4, fontsize=8, frameon=False)
ax.text(K - 0.5, -0.65, "↘ índice (PC1) crece hacia (3,3): alcance masivo / formato corto", fontsize=8, ha="right", color="#555")
ax.text(-0.5, -0.65, "↖ (0,0): profundidad / captions largos / tutoriales", fontsize=8, ha="left", color="#555")
save(fig, OUT / "campanias.png")



# =====================================================================================
# Pregunta 1: fútbol - ¿alcance masivo, descubrimiento o profundidad educativa?
# Tres objetivos = tres scores explícitos (sobre las variables originales):
#   masivo        = average_impressions
#   descubrimiento= cuentas nuevas alcanzadas por post = average_reach * non_follower_reach_pct / 100
#   profundidad   = promedio de z-scores (dentro de fútbol) de saves, comments, watch_time,
#                   tutorial + review
# =====================================================================================
from src.preprocessing import Standardizer

fut = d[(d.topic_football_pct >= TOPIC_MIN)].copy()
fut["masivo"] = fut.average_impressions
fut["descubrimiento"] = fut.average_reach * fut.non_follower_reach_pct / 100
zf = pd.DataFrame(Standardizer().fit_transform(fut[["saves_per_1000", "comments_per_1000", "average_watch_time_seconds"]].values), index=fut.index)
fut["profundidad"] = zf.mean(axis=1) + Standardizer().fit_transform((fut.content_tutorial_pct + fut.content_review_pct).values.reshape(-1, 1)).ravel()
obj = ["masivo", "descubrimiento", "profundidad"]
Z = pd.DataFrame(Standardizer().fit_transform(fut[obj].values), index=fut.index, columns=obj)
rk = fut[obj].rank(ascending=False).astype(int)
fut = fut.sort_values("indice_oja", ascending=False)
Z = Z.loc[fut.index]; rk = rk.loc[fut.index]
tabla_obj = fut[["indice_oja", "fila", "col", "masivo", "descubrimiento", "profundidad", "cpm_usd", "price_usd", "argentina_audience_pct"]].copy()
tabla_obj["rank_masivo"], tabla_obj["rank_descubrimiento"], tabla_obj["rank_profundidad"] = rk.masivo, rk.descubrimiento, rk.profundidad
tabla_obj.round(2).to_csv(OUT / "futbol_objetivos.csv")
P("\n=== Fútbol: ranking según objetivo ===")
P(tabla_obj.round(1).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [3, 2]})
ax = axes[0]
Mz = Z.values
im = ax.imshow(Mz, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(3)); ax.set_xticklabels(["alcance masivo\n(impresiones)", "descubrimiento\n(cuentas nuevas/post)", "profundidad\n(saves, comentarios,\nwatch time, tutorial/reseña)"], fontsize=8)
ax.set_yticks(range(len(fut))); ax.set_yticklabels([f"{n}  ({int(r.fila)},{int(r.col)})" for n, r in fut.iterrows()], fontsize=8)
for i in range(len(fut)):
    for j, o in enumerate(obj):
        v = Mz[i, j]
        ax.text(j, i, f"{int(rk.iloc[i][o])}º", ha="center", va="center", fontsize=8, color="white" if abs(v) > 1.1 else "black", fontweight="bold" if rk.iloc[i][o] <= 3 else "normal")
ax.set_title("Creadores de fútbol: posición según el objetivo (z-score y puesto)", fontsize=10)
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="z-score dentro de fútbol")
ax = axes[1]
y = np.arange(len(fut))
ax.barh(y - 0.2, fut.cpm_usd, 0.4, color="C3", label="CPM (USD por 1000 impresiones)")
ax.barh(y + 0.2, fut.argentina_audience_pct / 10, 0.4, color="C0", label="audiencia argentina (%/10)")
ax.axvline(5, color="C0", ls="--", lw=0.8); ax.text(5.05, len(fut) - 0.5, "50 % AR", fontsize=7, color="C0")
ax.set_yticks(y); ax.set_yticklabels([]); ax.invert_yaxis(); ax.set_ylim(len(fut) - 0.5, -0.5)
ax.set_xlabel("USD  /  % ÷ 10")
ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False)   # arriba del gráfico
ax.set_title("Costo y mercado local", fontsize=10, pad=34)
axes[0].set_ylim(len(fut) - 0.5, -0.5)
fig.suptitle("Pregunta 1 - Campaña de fútbol: el 'mejor' creador depende del objetivo (orden: índice de Oja, de masivo a profundo)")
save(fig, OUT / "futbol_objetivos.png")

# =====================================================================================
# Pregunta 2: selección por followers vs. por el sistema
# =====================================================================================
tema = pd.read_csv("resultados/eda/tema_dominante.csv").set_index("name")["tema_dominante"]
d["tema"] = tema
d["rank_sistema"] = d.indice_oja.rank(ascending=False).astype(int)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes[0]
colores_t = {"football": "C2", "dogs": "C1", "makeup": "C6", "art": "C0"}
for t, c in colores_t.items():
    m = d.tema == t
    ax.scatter(d.rank_followers[m], d.rank_sistema[m], c=c, s=50, label=t, edgecolor="k", lw=0.4)
m = d.argentina_audience_pct < AR_MIN
ax.scatter(d.rank_followers[m], d.rank_sistema[m], s=160, facecolor="none", edgecolor="red", lw=1.3, label=f"audiencia AR < {AR_MIN}%")
ax.plot([1, 50], [1, 50], "k--", lw=0.8)
for n, r in d.iterrows():
    if abs(r.rank_followers - r.rank_sistema) >= 8:
        ax.annotate(n, (r.rank_followers, r.rank_sistema), fontsize=6.5, xytext=(4, 2), textcoords="offset points")
ax.set_xlabel("puesto por followers (1 = más seguidores)"); ax.set_ylabel("puesto por índice del sistema (1 = más alcance/viral)")
rho = d[["rank_followers", "rank_sistema"]].corr(method="spearman").iloc[0, 1]
ax.set_title(f"Los dos rankings casi coinciden (Spearman {rho:.2f})... y ninguno mira audiencia ni tema", fontsize=10)
ax.legend(fontsize=7); ax.invert_xaxis(); ax.invert_yaxis()

# qué compra cada estrategia: top-5 por followers vs top-5 por índice vs los 5 de "profundidad"
ax = axes[1]
sets = {"Top-5 followers": d.sort_values("followers", ascending=False).head(5),
        "Top-5 índice (short)": d.sort_values("indice_oja", ascending=False).head(5),
        "5 de menor índice (long)": d.sort_values("indice_oja").head(5)}
mets = {"audiencia AR (%)": "argentina_audience_pct", "alcance no-seguidores (%)": "non_follower_reach_pct",
        "saves /1000": "saves_per_1000", "watch time (s)": "average_watch_time_seconds",
        "tutorial+reseña (%)": "tut_rev", "CPM (USD)": "cpm_usd", "precio (USD /100)": "price_100"}
d["tut_rev"] = d.content_tutorial_pct + d.content_review_pct
d["price_100"] = d.price_usd / 100
for k_, v in sets.items():
    sets[k_] = d.loc[v.index]
W = 0.26
x = np.arange(len(mets))
for i, (lab, sub) in enumerate(sets.items()):
    vals = [sub[c].mean() for c in mets.values()]
    bars = ax.bar(x + (i - 1) * W, vals, W, label=lab, color=["gray", "C0", "C1"][i])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.0f}", ha="center", fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels(list(mets), rotation=25, ha="right", fontsize=8)
ax.set_title("Qué 'compra' cada estrategia (promedio de los 5 elegidos)", fontsize=10); ax.legend(fontsize=8)
fig.suptitle("Pregunta 2 - Estrategia por followers vs. sistema")
save(fig, OUT / "followers_vs_sistema.png")
P("\n=== Qué compra cada estrategia (promedios) ===")
P(pd.DataFrame({k_: [round(v[c].mean(), 1) for c in mets.values()] for k_, v in sets.items()}, index=list(mets)).to_string())
P("Spearman(rank followers, rank sistema) =", round(rho, 3))

# =====================================================================================
# Pregunta 3: mantenimiento - ¿qué pasa cuando cambian los datos?
#  (a) Llega un creador nuevo: se proyecta con el modelo entrenado (media/desvío, PC1 y SOM
#      guardados) vs. reentrenar con él incluido. Se simula sacando 5 creadores, entrenando con 45
#      y proyectando los 5 "nuevos".
#  (b) Robustez del índice y del mapa al cambiar la cantidad de creadores: se quitan 10/20/30 %
#      al azar, se reentrena y se compara con el modelo completo (Spearman del índice sobre los
#      creadores en común; acuerdo de pares juntos/separados en el SOM).
#  (c) Actualización mensual: se perturban todas las métricas con ruido relativo del 10/20/30 %
#      (un "mes nuevo") y se mide cuánto cambia el ranking.
# =====================================================================================
from src.pca import PCA
from src.kohonen import Kohonen, KohonenConfig
import json
sel = json.load(open("resultados/eda/variables_seleccionadas.json"))["selected"]
Xraw = df[sel].values
names_all = np.array(df.index)
final_cfg = json.load(open("configs/kohonen.json"))
rng = np.random.default_rng(0)

def fit_index(Xtrain):
    p = PCA().fit(Xtrain); p.orient(0, sel.index("average_impressions"))
    return p
def pair_agreement(lab_a, lab_b):
    n_ = len(lab_a); same = tot = 0
    for i in range(n_):
        for j in range(i + 1, n_):
            tot += 1; same += (lab_a[i] == lab_a[j]) == (lab_b[i] == lab_b[j])
    return same / tot

p_full = fit_index(Xraw); idx_full = p_full.transform(Xraw, 1).ravel()
som_full = Kohonen(len(sel), KohonenConfig(**final_cfg)).train(p_full.scaler.transform(Xraw))
lab_full = som_full.winners(p_full.scaler.transform(Xraw))

# (a) creadores nuevos
rows_a = []
for rep in range(20):
    new = rng.choice(50, 5, replace=False); old = np.setdiff1d(np.arange(50), new)
    p_old = fit_index(Xraw[old])
    proy_vieja = p_old.transform(Xraw[new], 1).ravel()          # proyectar con el modelo viejo
    p_new = fit_index(Xraw)                                      # reentrenar con todos
    proy_nueva = p_new.transform(Xraw[new], 1).ravel()
    # ranking de los 5 nuevos entre los 50, con cada modelo
    r_vieja = pd.Series(np.r_[p_old.transform(Xraw[old], 1).ravel(), proy_vieja]).rank(ascending=False).values[-5:]
    r_nueva = pd.Series(p_new.transform(Xraw, 1).ravel()).rank(ascending=False).values[new]
    som_old = Kohonen(len(sel), KohonenConfig(**final_cfg)).train(p_old.scaler.transform(Xraw[old]))
    w_old = som_old.winners(p_old.scaler.transform(Xraw[new]))
    for i in range(5):
        rows_a.append(dict(rep=rep, name=names_all[new[i]], indice_modelo_viejo=proy_vieja[i], indice_reentrenado=proy_nueva[i],
                           rank_viejo=r_vieja[i], rank_nuevo=r_nueva[i], neurona_vieja=int(w_old[i])))
A = pd.DataFrame(rows_a)
A.round(3).to_csv(OUT / "mantenimiento_nuevos.csv", index=False)
r_a = np.corrcoef(A.indice_modelo_viejo, A.indice_reentrenado)[0, 1]
P(f"\n=== Mantenimiento (a): creador nuevo proyectado con modelo viejo vs. reentrenado: r = {r_a:.4f}, "
  f"diferencia media de puesto = {np.abs(A.rank_viejo - A.rank_nuevo).mean():.2f} (de 50)")

# (b) quitar creadores
rows_b = []
for frac in [0.1, 0.2, 0.3]:
    for rep in range(15):
        keep = np.sort(rng.choice(50, int(round(50 * (1 - frac))), replace=False))
        p_k = fit_index(Xraw[keep]); idx_k = p_k.transform(Xraw[keep], 1).ravel()
        sp = pd.Series(idx_k).corr(pd.Series(idx_full[keep]), method="spearman")
        som_k = Kohonen(len(sel), KohonenConfig(**{**final_cfg, "seed": rep})).train(p_k.scaler.transform(Xraw[keep]))
        agree = pair_agreement(som_k.winners(p_k.scaler.transform(Xraw[keep])), lab_full[keep])
        cos = abs(float(p_k.components_[:, 0] @ p_full.components_[:, 0]))
        rows_b.append(dict(quitados=frac, rep=rep, spearman_indice=sp, cos_pc1=cos, acuerdo_pares_som=agree, muertas=som_k.dead_units(p_k.scaler.transform(Xraw[keep]))))
B = pd.DataFrame(rows_b)
B.round(4).to_csv(OUT / "mantenimiento_quitar.csv", index=False)
P("\n=== Mantenimiento (b): quitar creadores y reentrenar ===")
P(B.groupby("quitados")[["spearman_indice", "cos_pc1", "acuerdo_pares_som", "muertas"]].agg(["mean", "std"]).round(3).to_string())

# (c) mes nuevo = ruido relativo en todas las métricas
rows_c = []
for noise in [0.05, 0.1, 0.2, 0.3]:
    for rep in range(15):
        Xm = Xraw * (1 + rng.normal(0, noise, Xraw.shape))
        p_m = fit_index(Xm); idx_m = p_m.transform(Xm, 1).ravel()
        idx_proj = p_full.transform(Xm, 1).ravel()               # sin reentrenar: proyectar el mes nuevo con el modelo viejo
        sp_re = pd.Series(idx_m).corr(pd.Series(idx_full), method="spearman")
        sp_pr = pd.Series(idx_proj).corr(pd.Series(idx_full), method="spearman")
        top5_full = set(np.argsort(-idx_full)[:5]); top5_m = set(np.argsort(-idx_m)[:5])
        som_m = Kohonen(len(sel), KohonenConfig(**{**final_cfg, "seed": rep})).train(p_m.scaler.transform(Xm))
        agree = pair_agreement(som_m.winners(p_m.scaler.transform(Xm)), lab_full)
        rows_c.append(dict(ruido=noise, rep=rep, spearman_reentrenado=sp_re, spearman_proyectado=sp_pr,
                           top5_comun=len(top5_full & top5_m), acuerdo_pares_som=agree))
C = pd.DataFrame(rows_c)
C.round(4).to_csv(OUT / "mantenimiento_mes.csv", index=False)
P("\n=== Mantenimiento (c): 'mes nuevo' con ruido relativo ===")
P(C.groupby("ruido")[["spearman_reentrenado", "spearman_proyectado", "top5_comun", "acuerdo_pares_som"]].mean().round(3).to_string())

fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
ax = axes[0]
ax.scatter(A.indice_modelo_viejo, A.indice_reentrenado, s=18, alpha=0.6)
lim = [A[["indice_modelo_viejo", "indice_reentrenado"]].min().min(), A[["indice_modelo_viejo", "indice_reentrenado"]].max().max()]
ax.plot(lim, lim, "k--", lw=0.8); ax.set_xlabel("índice proyectado con el modelo viejo (45 creadores)"); ax.set_ylabel("índice reentrenando con los 50")
ax.set_title(f"(a) Creador nuevo: proyectar alcanza\n(r = {r_a:.3f}; puesto cambia {np.abs(A.rank_viejo - A.rank_nuevo).mean():.1f} lugares en promedio)", fontsize=9)
ax = axes[1]
gb = B.groupby("quitados")
for col, lab, c in [("spearman_indice", "Spearman del índice", "C0"), ("cos_pc1", "|cos(PC1 nuevo, PC1 viejo)|", "C2"), ("acuerdo_pares_som", "acuerdo de pares en el SOM", "C1")]:
    ax.errorbar(gb[col].mean().index * 100, gb[col].mean(), yerr=gb[col].std(), marker="o", capsize=3, label=lab, color=c)
ax.set_xlabel("% de creadores quitados antes de reentrenar"); ax.set_ylim(0.7, 1.02); ax.legend(fontsize=7)
ax.set_title("(b) Cambiar la cantidad de creadores:\nel índice es estable, el mapa cambia más", fontsize=9)
ax = axes[2]
gc = C.groupby("ruido")
ax.errorbar(gc.spearman_reentrenado.mean().index * 100, gc.spearman_reentrenado.mean(), yerr=gc.spearman_reentrenado.std(), marker="o", capsize=3, label="índice reentrenado vs. mes anterior", color="C0")
ax.errorbar(gc.spearman_proyectado.mean().index * 100, gc.spearman_proyectado.mean(), yerr=gc.spearman_proyectado.std(), marker="s", capsize=3, label="mes nuevo proyectado con modelo viejo", color="C3")
ax.errorbar(gc.acuerdo_pares_som.mean().index * 100, gc.acuerdo_pares_som.mean(), yerr=gc.acuerdo_pares_som.std(), marker="^", capsize=3, label="acuerdo de pares en el SOM", color="C1")
ax2 = ax.twinx(); ax2.plot(gc.top5_comun.mean().index * 100, gc.top5_comun.mean(), ":", color="gray", marker="d", label="top-5 en común (de 5)"); ax2.set_ylim(0, 5.3); ax2.set_ylabel("top-5 en común", color="gray")
ax.set_xlabel("ruido relativo en las métricas del mes (%)"); ax.set_ylim(0.7, 1.02); ax.legend(fontsize=7, loc="lower left")
ax.set_title("(c) Actualización mensual: el ranking se mantiene,\nel top-5 puede cambiar 1–2 nombres", fontsize=9)
fig.suptitle("Pregunta 3 - Mantenimiento del sistema (simulaciones sobre los 50 creadores)")
save(fig, OUT / "mantenimiento.png")

log.close()
