"""Ejercicio 2.1 (extensión) - Los 5 QR almacenados a la vez con la regla de la pseudoinversa.

Con Hebb, `ej2_qr.py` muestra que los 5 QR no son estables (están muy correlacionados). Acá se cambia
la regla de pesos, no la red:  W = K^T (K K^T)^-1 K  (src/hopfield.py, train(rule="pinv")).

  1. Estabilidad: los 5 son puntos fijos con energía -N/2 (Hebb: 0/5); espurios que quedan sin ruido.
  2. Ruido aleatorio sobre los 5 QR: a dónde termina la red (correcto / equivocado / negado / espurio),
     matrices de confusión y energía durante la recuperación.
  3. Desgastes realistas (esquina rota, mancha, rayón, borrado, tinta desvanecida) con un control de
     ruido al azar que cambia la misma cantidad de módulos.

Salidas en resultados/hopfield_qr_pinv/. Hiperparámetros en configs/hopfield_qr_pinv.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.deterioros import TODOS, aplicar
from src.hopfield import Hopfield, add_noise, hamming
from src.plots import heatmap, pattern_grid, save
from src.qr import QR_SIZE, decode, load_qr_folder

OUT = Path("resultados/hopfield_qr_pinv")
OUT.mkdir(parents=True, exist_ok=True)
cfg = json.load(open("configs/hopfield_qr_pinv.json"))
S = QR_SIZE
log = open(OUT / "salida.txt", "w", encoding="utf-8")


def P(*a):
    print(*a)
    print(*a, file=log)


qrs = load_qr_folder(cfg["carpeta"])
names = list(qrs)
K = np.array([qrs[n].reshape(-1) for n in names])
p, N = K.shape
PISO = -N / 2
COL = {"correcto": "#1F4E79", "equivocado": "#CD212A", "negado": "#E08A00", "espurio": "#7C8189"}
CATS = list(COL)


def red(rule, update):
    return Hopfield(update=update, seed=cfg["seed"]).train(K, rule=rule)


def destino(net, final):
    m, neg = net.match(final)
    return "espurio" if m is None else names[m] + ("(neg)" if neg else "")


def categoria(net, final, mu):
    m, neg = net.match(final)
    return "espurio" if m is None else ("negado" if neg else ("correcto" if m == mu else "equivocado"))


def wilson(k, n, z=1.96):
    ph = k / n
    centro = (ph + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    med = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return centro - med, centro + med


# ===================================================================== 1. estabilidad
P(f"{p} QR de {S}x{S} -> N = {N}; piso de energía con pinv: -N/2 = {PISO}")
net = red("pinv", "sync")
P(f"W = K^T (K K^T)^-1 K: rango de K = {np.linalg.matrix_rank(K)} de {p};  simétrica: {np.allclose(net.W, net.W.T)};  "
  f"proyector (W^2 = W): {np.allclose(net.W @ net.W, net.W)};  W xi = xi: {np.allclose(net.W @ K.T, K.T)};  traza = {np.trace(net.W):.3f}")
rows = []
for upd in ("sync", "async"):
    for rule in ("hebb", "pinv"):
        n_ = red(rule, upd)
        for n, x in zip(names, K):
            r = n_.recall(x, max_iter=cfg["max_iter"])
            rows.append(dict(regla=rule, update=upd, qr=n, punto_fijo=bool(np.array_equal(r.final, x)),
                             H_patron=n_.energy(x), H_final=n_.energy(r.final), hamming_final=hamming(r.final, x)))
est = pd.DataFrame(rows)
est.round(2).to_csv(OUT / "estabilidad.csv", index=False)
P("\n== Estabilidad de los 5 QR ==\n" + est.groupby(["regla", "update"]).agg(
    puntos_fijos=("punto_fijo", "sum"), H_patron_min=("H_patron", "min"), H_patron_max=("H_patron", "max")).round(1).to_string())

# espurios sin ruido: la red parte de estados aleatorios
net_e = red("pinv", cfg["update_espurios"])
rng = np.random.default_rng(cfg["seed"])
rows = []
for i in range(cfg["inicios_aleatorios"]):
    r = net_e.recall(rng.choice([-1, 1], size=N), max_iter=cfg["max_iter_espurios"])
    rows.append(dict(inicio=i, destino=destino(net_e, r.final), H_final=net_e.energy(r.final), ciclo=r.cycle))
esp_df = pd.DataFrame(rows)
esp_df.round(2).to_csv(OUT / "espurios_sin_ruido.csv", index=False)
H_esp = esp_df[esp_df.destino == "espurio"].H_final
P(f"\n== {cfg['inicios_aleatorios']} inicios aleatorios ({cfg['update_espurios']}) ==\n"
  f"a un QR o su negativo: {(esp_df.destino != 'espurio').sum()}, espurios: {len(H_esp)}, ciclos: {int(esp_df.ciclo.sum())}"
  + (f"; H de los espurios: mín {H_esp.min():.1f}, mediana {H_esp.median():.1f}, máx {H_esp.max():.1f}" if len(H_esp) else ""))

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
hebb = est[(est.regla == "hebb") & (est["update"] == cfg["update"])]
pin = est[(est.regla == "pinv") & (est["update"] == cfg["update"])]
axs[0].bar(range(p), hebb.H_patron, color="#5B7697", label="QR almacenado")
axs[0].axhline(hebb.H_final.iloc[0], color=COL["equivocado"], ls="--", label="atractor espurio al que van los 5")
axs[0].set_ylim(-2700, 50)
axs[0].set_xticks(range(p))
axs[0].set_xticklabels(names, rotation=30, fontsize=8)
axs[0].set_title("Hebb (diagonal en 0)")
axs[0].set_ylabel("energía H")
axs[1].plot(range(p), pin.H_patron, "o", color=COL["correcto"], ms=8, label="QR almacenado")
if len(H_esp):
    axs[1].scatter(np.full(len(H_esp), p) + np.random.default_rng(0).uniform(-0.25, 0.25, len(H_esp)), H_esp,
                   s=10, color=COL["equivocado"], alpha=0.6, label="estado espurio (inicio aleatorio)")
axs[1].axhline(PISO, color="gray", ls="--", label="piso -N/2")
axs[1].set_ylim(PISO - 25, PISO + 80)
axs[1].set_xticks(range(p + 1))
axs[1].set_xticklabels(names + ["espurios"], rotation=30, fontsize=8)
axs[1].set_title("Pseudoinversa (escala ampliada)")
for ax in axs:
    ax.legend(fontsize=7, loc="lower right")
fig.suptitle("Energía de cada QR almacenado")
save(fig, OUT / "estabilidad.png")

# ===================================================================== 2. ruido aleatorio
net = red("pinv", cfg["update"])
rng = np.random.default_rng(cfg["seed"])
rows = []
for pr in cfg["niveles_ruido"]:
    for mu, xi in enumerate(K):
        for _ in range(cfg["repeticiones"]):
            z = add_noise(xi, pr, rng)
            r = net.recall(z, max_iter=cfg["max_iter"])
            rows.append(dict(ruido=pr, qr=names[mu], dest=destino(net, r.final), cat=categoria(net, r.final, mu),
                             iter=r.iterations, ciclo=r.cycle, bits=hamming(z, xi), hamming_final=hamming(r.final, xi),
                             H0=r.energies[0], Hf=r.energies[-1], mono=bool(np.all(np.diff(r.energies) <= 1e-9)),
                             lector_in=decode(z) != "", lector_out=decode(r.final) != "",
                             traza=" ".join(f"{e:.2f}" for e in r.energies)))
ru = pd.DataFrame(rows)
ru.drop(columns="traza").round(3).to_csv(OUT / "ruido_raw.csv", index=False)
ru[["ruido", "qr", "traza"]].to_csv(OUT / "ruido_energias.csv", index=False)
arows = {}
for pr, g in ru.groupby("ruido"):
    lo, hi = wilson((g.cat == "correcto").sum(), len(g))
    arows[pr] = {**{c: (g.cat == c).mean() for c in CATS}, "correcto_lo95": lo, "correcto_hi95": hi,
                 "iter": g["iter"].mean(), "H0": g.H0.mean(), "Hf": g.Hf.mean(), "lector_in": g.lector_in.mean(),
                 "lector_out": g.lector_out.mean(), "energia_nunca_sube": g.mono.all()}
agg = pd.DataFrame.from_dict(arows, orient="index")
agg.index.name = "ruido"
agg.round(3).to_csv(OUT / "ruido.csv")
P(f"\n== Ruido aleatorio, 5 QR, pinv {cfg['update']} ({cfg['repeticiones']} corridas por QR y nivel = {cfg['repeticiones'] * p} por nivel) ==\n"
  + agg.round(3).to_string())
por_qr = ru.assign(ok=ru.cat == "correcto").groupby(["ruido", "qr"]).ok.mean().unstack() * 100
por_qr.round(1).to_csv(OUT / "ruido_por_qr.csv")

fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
x = agg.index * 100
axs[0].fill_between(x, agg.correcto_lo95 * 100, agg.correcto_hi95 * 100, color="#5B7697", alpha=0.25, lw=0)
axs[0].plot(x, agg.correcto * 100, "o-", color=COL["correcto"], label="QR recuperado exacto (IC 95 % Wilson)")
axs[0].plot(x, agg.lector_in * 100, "^--", color="gray", label="el lector decodifica la entrada con ruido")
axs[0].set_title(f"Recuperación vs ruido ({cfg['repeticiones'] * p} corridas por nivel)")
for mu_name in names:
    axs[1].plot(x, por_qr[mu_name], "o-", ms=4, label=mu_name)
axs[1].set_title("Por QR de partida")
for ax in axs:
    ax.set_xlabel("módulos invertidos al azar (%)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
axs[0].set_ylabel("%")
save(fig, OUT / "ruido.png")

fig, ax = plt.subplots(figsize=(7, 4))
bottom = np.zeros(len(agg))
for c in CATS:
    ax.bar(x, agg[c] * 100, bottom=bottom, width=3.5, color=COL[c], label=c)
    bottom += agg[c].values * 100
ax.set_xlabel("módulos invertidos al azar (%)")
ax.set_ylabel("% de las corridas")
ax.set_title("Cómo termina la red")
ax.legend(fontsize=8, loc="lower left")
save(fig, OUT / "categorias_vs_ruido.png")

# matrices de confusión (el negado es una columna aparte)
cols = names + ["negado", "espurio"]
fig, axs = plt.subplots(1, len(cfg["ruido_confusion"]), figsize=(6.2 * len(cfg["ruido_confusion"]), 4.6))
axs = np.atleast_1d(axs)
for ax, pr in zip(axs, cfg["ruido_confusion"]):
    d = ru[np.isclose(ru.ruido, pr)]
    dd = d.dest.where(~d.dest.str.endswith("(neg)"), "negado")
    M = pd.crosstab(d.qr, dd).reindex(index=names, columns=cols, fill_value=0)
    M.to_csv(OUT / f"confusion_{round(pr * 100)}.csv")
    P(f"\n== Matriz de confusión con {pr:.0%} de ruido (filas: QR de partida) ==\n{M.to_string()}")
    heatmap(M.values, title=f"{pr:.0%} de ruido: a dónde termina la red", xticks=cols, yticks=names, cmap="Blues",
            annot=M.values.astype(int).astype(str), fontsize=9, ax=ax)
save(fig, OUT / "confusion.png")

# energía durante la recuperación
fig, axs = plt.subplots(1, len(cfg["ruido_trayectorias"]), figsize=(4.2 * len(cfg["ruido_trayectorias"]), 3.6), sharey=True)
for ax, pr in zip(np.atleast_1d(axs), cfg["ruido_trayectorias"]):
    for c in ["correcto", "negado", "equivocado", "espurio"]:
        for t in ru[np.isclose(ru.ruido, pr) & (ru.cat == c)].traza:
            e = [float(v) for v in t.split()]
            ax.plot(e, "-", color=COL[c], alpha=0.3, lw=1)
            ax.plot(len(e) - 1, e[-1], "o", color=COL[c], ms=3.5, alpha=0.5)
    ax.axhline(PISO, color="gray", ls="--", lw=1)
    ax.set_title(f"{pr:.0%} de ruido")
    ax.set_xlabel("iteración")
np.atleast_1d(axs)[0].set_ylabel("energía H")
save(fig, OUT / "energia_trayectorias.png")

fin = ru[ru.ruido >= cfg["ruido_energia_final_min"] - 1e-9]
fig, ax = plt.subplots(figsize=(6, 3.8))
jit = np.random.default_rng(3)
for i, c in enumerate(CATS):
    h = fin[fin.cat == c].Hf
    ax.scatter(i + jit.uniform(-0.28, 0.28, len(h)), h, s=14, color=COL[c], alpha=0.5)
ax.axhline(PISO, color="gray", ls="--", lw=1)
ax.set_xticks(range(len(CATS)))
ax.set_xticklabels([f"{c}\n(n = {(fin.cat == c).sum()})" for c in CATS])
ax.set_ylabel("energía final H")
ax.set_title(f"Energía final por categoría (ruido ≥ {cfg['ruido_energia_final_min']:.0%})")
save(fig, OUT / "energia_final.png")
P("\n== Energía final por categoría (ruido >= %.0f %%) ==\n%s" % (cfg["ruido_energia_final_min"] * 100,
  fin.groupby("cat").Hf.agg(["count", "min", "median", "max"]).round(1).to_string()))

# ===================================================================== 3. desgastes realistas
rng = np.random.default_rng(cfg["seed"])
rows = []
for tipo in TODOS:
    for f in cfg["niveles_desgaste"]:
        if tipo == "desvanecido" and f > cfg["desvanecido_f_max"]:
            continue
        for mu, xi in enumerate(K):
            for _ in range(cfg["repeticiones"]):
                z = aplicar(tipo, xi, f, rng)
                k = int(np.sum(z != xi))
                r = net.recall(z, max_iter=cfg["max_iter"])
                zc = xi.copy()                                                  # control: mismos k módulos, al azar
                zc[rng.choice(N, size=k, replace=False)] *= -1
                rc = net.recall(zc, max_iter=cfg["max_iter"])
                rows.append(dict(tipo=tipo, f=f, qr=names[mu], k=k, cat=categoria(net, r.final, mu), iter=r.iterations,
                                 cat_ctrl=categoria(net, rc.final, mu), lector_in=decode(z) != "", lector_out=decode(r.final) != ""))
de = pd.DataFrame(rows)
de.to_csv(OUT / "desgastes_raw.csv", index=False)
rrows = {}
for (tipo, f), g in de.groupby(["tipo", "f"]):
    rrows[(tipo, f)] = {"modulos_cambiados_%": g.k.mean() / N * 100,
                        "correcto_%": (g.cat == "correcto").mean() * 100,
                        "control_correcto_%": (g.cat_ctrl == "correcto").mean() * 100,
                        "lector_lee_entrada_%": g.lector_in.mean() * 100,
                        "lector_lee_salida_si_correcto_%": g[g.cat == "correcto"].lector_out.mean() * 100}
res = pd.DataFrame.from_dict(rrows, orient="index")
res.index = pd.MultiIndex.from_tuples(res.index, names=["tipo", "f"])
res.round(1).to_csv(OUT / "desgastes.csv")
P("\n== Desgastes realistas, 5 QR, pinv (%d corridas por punto) ==\n%s" % (cfg["repeticiones"] * p, res.round(1).to_string()))

fig, ax = plt.subplots(figsize=(7, 4.2))
ctrl = de.assign(ok=de.cat_ctrl == "correcto").groupby("f").ok.mean() * 100
ax.plot(ctrl.index * 100, ctrl.values, "--", color="gray", lw=2, label="ruido al azar, mismos módulos cambiados")
for tipo in TODOS:
    gg = de[de.tipo == tipo].assign(ok=lambda d: d.cat == "correcto").groupby("f").ok.mean() * 100
    ax.plot(gg.index * 100, gg.values, "o-", ms=5, label=tipo)
ax.set_xlabel("superficie del QR dañada (%)")
ax.set_ylabel("QR recuperado exacto (%)")
ax.set_ylim(85, 101.5)
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=8, loc="lower left")
ax.set_title("Recuperación vs desgaste (150 corridas por punto)")
save(fig, OUT / "desgastes.png")

# ===================================================================== 4. control: el par de 2.2 con Hebb y con pinv
idx = [names.index(n) for n in cfg["par"]]
cmp_rows = []
for rule in ("hebb", "pinv"):
    net_par = Hopfield(update=cfg["update"], seed=cfg["seed"]).train(K[idx], rule=rule)
    rng = np.random.default_rng(cfg["seed"])                                       # mismo ruido para las dos reglas
    for pr in cfg["niveles_ruido"]:
        ok = [net_par.recall(add_noise(K[i], pr, rng), max_iter=cfg["max_iter"]).final for i in idx for _ in range(cfg["repeticiones"])]
        ref = [K[i] for i in idx for _ in range(cfg["repeticiones"])]
        cmp_rows.append(dict(regla=rule, ruido=pr, correcto=float(np.mean([np.array_equal(a, b) for a, b in zip(ok, ref)]))))
cmp_df = pd.DataFrame(cmp_rows).pivot(index="ruido", columns="regla", values="correcto")
cmp_df.round(3).to_csv(OUT / "par_hebb_vs_pinv.csv")
P(f"\n== Par {cfg['par']}: recuperación exacta con Hebb vs pinv (mismo ruido, {cfg['repeticiones'] * len(idx)} corridas por nivel) ==\n"
  + cmp_df.round(3).to_string())

ej = cfg["ejemplo_desgaste"]
rng_e = np.random.default_rng(ej["seed"])
mu = names.index(ej["qr"])
entradas, salidas, t_in, t_out = [], [], [], []
for tipo in TODOS:
    z = aplicar(tipo, K[mu], ej["f"], rng_e)
    r = net.recall(z, max_iter=cfg["max_iter"])
    entradas.append(z)
    salidas.append(r.final)
    t_in.append(f"{tipo}\n{int(np.sum(z != K[mu]))} módulos cambiados")
    t_out.append(f"salida: {categoria(net, r.final, mu)}")
fig = pattern_grid(entradas + salidas, t_in + t_out, S, ncols=len(TODOS), figsize=(2.3 * len(TODOS), 5.4),
                   suptitle=f"'{ej['qr']}' con {ej['f']:.0%} de superficie dañada: entrada (arriba) y salida de la red (abajo)")
save(fig, OUT / "desgastes_ejemplos.png")
log.close()
print("Salidas en", OUT)
