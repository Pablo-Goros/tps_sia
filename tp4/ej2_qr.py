"""Ejercicio 2.1 - Recuperación de códigos QR deteriorados con una red de Hopfield.

  - Cada QR se representa como matriz binaria 37x37 en {1, -1} (N = 1369 neuronas).
  - Se almacenan los 5 QR -> se analiza por qué la red FALLA (patrones muy correlacionados).
  - Se estudia qué subconjuntos sí se pueden almacenar (todos los pares funcionan, ningún trío).
  - Sobre un par se analiza ruido aleatorio, deterioros realistas, iteraciones, energía y si
    un lector de QR real (OpenCV) decodifica la salida de la red.
  - Capacidad: con patrones aleatorios la cota 0.15 N sí se cumple -> el problema es la
    correlación entre QR, no la cantidad.

Salidas en resultados/hopfield_qr/
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.hopfield import Hopfield, add_noise, hamming, orthogonality_score
from src.plots import save, heatmap, pattern_grid
from src.qr import QR_SIZE, load_qr_folder, decode

OUT = Path("resultados/hopfield_qr")
OUT.mkdir(parents=True, exist_ok=True)
cfg = json.load(open("configs/hopfield_qr.json"))
rng = np.random.default_rng(cfg["seed"])
S = QR_SIZE
N = S * S
log = open(OUT / "salida.txt", "w")
def P(*a):
    print(*a); print(*a, file=log)

# ================================================================ 1. cargar los QR
qrs = load_qr_folder(cfg["carpeta"])
names = list(qrs)
patterns = np.array([qrs[n].reshape(-1) for n in names])
P(f"{len(names)} QR de {S}x{S} -> N = {N}; cota de capacidad 0.15 N = {0.15 * N:.0f} patrones")
for n in names:
    P(f"  {n:10s} decodifica como: {decode(qrs[n])}")

sc = orthogonality_score(patterns)
G = pd.DataFrame(sc["gram"] / N, index=names, columns=names)
P("\nSolapamiento xi^mu . xi^nu / N  (0 = ortogonales, 1 = idénticos):\n" + G.round(3).to_string())
shared = (patterns == patterns[0]).all(axis=0)
P(f"módulos idénticos en los 5 QR: {shared.mean():.1%} ({shared.sum()} de {N})")
fig, ax = heatmap(G.values, title="Solapamiento normalizado entre QR", xticks=names, yticks=names,
                  cmap="coolwarm", vmin=-1, vmax=1, annot=True, figsize=(6, 5))
save(fig, OUT / "ortogonalidad.png")
fig = pattern_grid(list(patterns) + [np.where(shared, patterns[0], 0)], names + ["módulos comunes a los 5\n(gris = varían)"], S, ncols=6,
                   suptitle="Patrones almacenados (37x37) y estructura compartida")
save(fig, OUT / "patrones.png")

# ======================================= 2. almacenar los 5: ¿por qué falla?
net5 = Hopfield(update=cfg["update"], seed=cfg["seed"]).train(patterns)
P("\n== Red con los 5 QR almacenados ==")
rows = []
for mu, (n, xi) in enumerate(zip(names, patterns)):
    h = net5.W @ xi
    # h_i^nu = xi_i^nu * (N-1)/N  +  crosstalk ; el bit es inestable si sign(h_i) != xi_i
    inestables = int((np.sign(h) != xi).sum())
    r = net5.recall(xi)
    rows.append(dict(qr=n, bits_inestables=inestables, punto_fijo=np.array_equal(r.final, xi),
                     iter=r.iterations, H_patron=net5.energy(xi), H_final=r.energies[-1],
                     hamming_final=hamming(r.final, xi)))
t5 = pd.DataFrame(rows)
t5.round(2).to_csv(OUT / "cinco_qr_estabilidad.csv", index=False)
P(t5.round(2).to_string(index=False))
r0 = net5.recall(patterns[0])
attr = r0.final
P(f"Todos convergen al mismo atractor espurio? "
  f"{all(np.array_equal(net5.recall(x).final, attr) for x in patterns)}  "
  f"H(atractor) = {net5.energy(attr):.1f} < min H(patrones) = {t5.H_patron.min():.1f}")
P(f"overlaps del atractor con cada QR: {dict(zip(names, np.round(net5.overlaps(attr), 3)))}")
P(f"¿el lector decodifica el atractor? {decode(attr) or 'NO'}")
fig = pattern_grid([patterns[0], r0.states[1], attr, np.where(attr == patterns[0], 0, 1)],
                   [f"'{names[0]}' (consulta SIN ruido)\nH={r0.energies[0]:.0f}", f"t=1  H={r0.energies[1]:.0f}",
                    f"estado estable (espurio)\nH={r0.energies[-1]:.0f}", f"módulos que cambiaron\n(Hamming={hamming(attr, patterns[0])})"],
                   S, ncols=4, suptitle="Con los 5 QR almacenados ni el patrón original es estable")
save(fig, OUT / "cinco_qr_falla.png")

# =========================================== 3. ¿qué subconjuntos sí funcionan?
rows = []
for p in range(1, 6):
    for combo in itertools.combinations(range(5), p):
        pats = patterns[list(combo)]
        h = Hopfield(update=cfg["update"], seed=cfg["seed"]).train(pats)
        estables = np.mean([np.array_equal(h.recall(x).final, x) for x in pats])
        ok = np.mean([np.array_equal(h.recall(add_noise(x, 0.2, rng), max_iter=50).final, x) for x in pats for _ in range(10)])
        o = orthogonality_score(pats)["max_abs"] if p > 1 else 0.0
        rows.append(dict(p=p, qrs="+".join(names[i] for i in combo), solap_max=o, estables=estables, recuperacion_20pct=ok))
sub = pd.DataFrame(rows)
sub.round(3).to_csv(OUT / "subconjuntos.csv", index=False)
P("\n== Subconjuntos ==\n" + sub.groupby("p")[["solap_max", "estables", "recuperacion_20pct"]].mean().round(3).to_string())
P("(todos los pares funcionan; ningún trío/cuarteto. Detalle en subconjuntos.csv)")

# ============================================= 4. caso de uso con un par de QR
par = cfg["par"]
idx = [names.index(x) for x in par]
pats = patterns[idx]
net = Hopfield(update=cfg["update"], seed=cfg["seed"]).train(pats)
P(f"\n== Red con el par {par} (solapamiento {G.loc[par[0], par[1]]:.3f}) ==")
for n, xi in zip(par, pats):
    P(f"  '{n}' punto fijo: {np.array_equal(net.recall(xi).final, xi)}  H = {net.energy(xi):.1f}")

# ---- ejemplo paso a paso
z = add_noise(pats[0], cfg["ruido_ejemplo"], rng)
r = net.recall(z)
P(f"\nEjemplo '{par[0]}' con ruido {cfg['ruido_ejemplo']:.0%} ({hamming(z, pats[0])} módulos): {r.iterations} iteraciones, "
  f"recuperado={np.array_equal(r.final, pats[0])}; lector antes: {decode(z) or 'NO'} | después: {decode(r.final) or 'NO'}")
P("  energía por paso:", [round(e, 1) for e in r.energies])
P("  Hamming al original por paso:", [hamming(s, pats[0]) for s in r.states])
fig = pattern_grid(r.states, [f"t={t}  H={e:.0f}\nHamming={hamming(s, pats[0])}" for t, (s, e) in enumerate(zip(r.states, r.energies))],
                   S, ncols=len(r.states), suptitle=f"Recall de '{par[0]}' con {cfg['ruido_ejemplo']:.0%} de ruido")
save(fig, OUT / "ejemplo_evolucion.png")

# ---- barrido de ruido aleatorio
rows = []
for p_flip in cfg["niveles_ruido"]:
    for mu, xi in enumerate(pats):
        for _ in range(cfg["repeticiones"]):
            z = add_noise(xi, p_flip, rng)
            r = net.recall(z, max_iter=50)
            m, neg = net.match(r.final)
            rows.append(dict(ruido=p_flip, qr=par[mu], bits=hamming(z, xi), iter=r.iterations, ciclo=r.cycle,
                             ok=(m == mu and not neg), espurio=(m is None), otro=(m is not None and m != mu),
                             negativo=(m == mu and neg), hamming_final=hamming(r.final, xi),
                             H0=r.energies[0], Hf=r.energies[-1],
                             lector_consulta=decode(z) != "", lector_final=decode(r.final) != ""))
res = pd.DataFrame(rows)
res.to_csv(OUT / "ruido_raw.csv", index=False)
agg = res.groupby("ruido").agg(recuperado=("ok", "mean"), espurio=("espurio", "mean"), otro=("otro", "mean"),
                               negativo=("negativo", "mean"), ciclo=("ciclo", "mean"), iter=("iter", "mean"),
                               bits=("bits", "mean"), hamming_final=("hamming_final", "mean"),
                               H0=("H0", "mean"), Hf=("Hf", "mean"),
                               lector_consulta=("lector_consulta", "mean"), lector_final=("lector_final", "mean"))
agg.round(3).to_csv(OUT / "ruido.csv")
P("\n== Barrido de ruido (par) ==\n" + agg.round(3).to_string())

fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
axes[0].plot(agg.index, agg.recuperado, "o-", label="Hopfield recupera el QR exacto")
axes[0].plot(agg.index, agg.lector_final, "s-", label="lector decodifica la SALIDA de Hopfield")
axes[0].plot(agg.index, agg.lector_consulta, "^--", color="gray", label="lector decodifica el QR deteriorado (sin red)")
axes[0].plot(agg.index, agg.otro, "v:", color="C3", label="converge al otro QR")
axes[0].plot(agg.index, agg.negativo, "x:", color="C4", label="converge al QR negado")
axes[0].set_xlabel("prob. de invertir cada módulo"); axes[0].set_ylabel("fracción"); axes[0].legend(fontsize=7)
axes[0].set_title("Recuperación vs ruido")
axes[1].plot(agg.index, agg["iter"], "o-", color="C3"); axes[1].set_xlabel("prob. de invertir cada módulo")
axes[1].set_ylabel("iteraciones"); axes[1].set_title("Iteraciones hasta estabilizar")
axes[2].plot(agg.index, agg.H0, "o-", label="H(consulta)"); axes[2].plot(agg.index, agg.Hf, "s-", label="H(final)")
axes[2].axhline(net.energy(pats[0]), color="gray", ls="--", label="H(patrón almacenado)")
axes[2].set_xlabel("prob. de invertir cada módulo"); axes[2].set_ylabel("energía"); axes[2].legend(fontsize=8)
axes[2].set_title("Energía inicial y final")
save(fig, OUT / "ruido.png")

fig, ax = plt.subplots(figsize=(7, 4))
for p_flip in [0.1, 0.2, 0.3, 0.4, 0.45, 0.5]:
    r = net.recall(add_noise(pats[1], p_flip, rng), max_iter=50)
    m, neg = net.match(r.final)
    tag = "ok" if (m == 1 and not neg) else ("negado" if m == 1 else ("otro QR" if m is not None else "espurio"))
    ax.plot(r.energies, "o-", label=f"ruido {p_flip:.0%} -> {tag}{' (ciclo)' if r.cycle else ''}")
ax.set_xlabel("iteración"); ax.set_ylabel("H"); ax.set_title(f"Energía en cada paso ('{par[1]}')"); ax.legend(fontsize=7)
save(fig, OUT / "energia.png")

# ---- deterioros "realistas" del enunciado (esquina rota, manchas, rayones, rasgado)
def esquina_rota(x, size=10):
    m = x.reshape(S, S).copy(); m[-size:, -size:] = -1.0; return m.reshape(-1)
def mancha(x, cy=18, cx=18, rad=6):
    m = x.reshape(S, S).copy(); yy, xx = np.ogrid[:S, :S]
    m[(yy - cy) ** 2 + (xx - cx) ** 2 <= rad ** 2] = 1.0; return m.reshape(-1)
def rayon(x, fila=20, ancho=2):
    m = x.reshape(S, S).copy(); m[fila:fila + ancho, :] = -1.0; return m.reshape(-1)
def rasgado(x, cols=8):
    m = x.reshape(S, S).copy(); m[:, :cols] = rng.choice([-1.0, 1.0], size=(S, cols)); return m.reshape(-1)
def borrado(x, frac=0.3):
    m = x.copy(); m[rng.choice(N, int(frac * N), replace=False)] = -1.0; return m

casos = {"esquina rota 10x10": esquina_rota, "esquina rota 16x16": lambda x: esquina_rota(x, 16),
         "mancha r=6": mancha, "mancha r=10": lambda x: mancha(x, rad=10),
         "rayón 2 filas": rayon, "rayón 5 filas": lambda x: rayon(x, 16, 5),
         "rasgado 8 col.": rasgado, "borrado 30%": borrado, "borrado 50%": lambda x: borrado(x, 0.5)}
rows, imgs, titles = [], [], []
for cname, fn in casos.items():
    for mu, xi in enumerate(pats):
        z = fn(xi)
        r = net.recall(z, max_iter=50)
        rows.append(dict(caso=cname, qr=par[mu], bits=hamming(z, xi), iter=r.iterations, ok=np.array_equal(r.final, xi),
                         lector_antes=decode(z) != "", lector_despues=decode(r.final) != "", hamming_final=hamming(r.final, xi)))
        if mu == 0:
            imgs += [z, r.final]
            titles += [f"{cname} ({hamming(z, xi)} bits)\nlector: {'sí' if decode(z) else 'no'}",
                       f"Hopfield, {r.iterations} it.\n{'OK, lector: sí' if np.array_equal(r.final, xi) else f'Hamming={hamming(r.final, xi)}'}"]
det = pd.DataFrame(rows)
det.round(3).to_csv(OUT / "deterioros.csv", index=False)
P("\n== Deterioros realistas (par) ==\n" + det.round(3).to_string(index=False))
fig = pattern_grid(imgs, titles, S, ncols=6, figsize=(14, 2.9 * int(np.ceil(len(imgs) / 6))),
                   suptitle=f"Deterioros sobre '{par[0]}': consulta -> salida de la red")
save(fig, OUT / "deterioros.png")

# =========================== 5. capacidad: con patrones ALEATORIOS la cota 0.15 N se cumple
rows = []
for extra in [0, 25, 50, 100, 150, 200, 250, 300, 400]:
    fake = rng.choice([-1.0, 1.0], size=(extra, N))
    allp = np.vstack([pats, fake]) if extra else pats
    h = Hopfield(update="sync", seed=0).train(allp)
    estables = np.mean([np.array_equal(h.recall(x).final, x) for x in pats])
    ok = np.mean([np.array_equal(h.recall(add_noise(x, 0.15, rng), max_iter=50).final, x) for x in pats for _ in range(5)])
    rows.append(dict(patrones_totales=len(allp), p_sobre_N=len(allp) / N, qr_estables=estables, recuperacion_15pct=ok))
    P(f"  p={len(allp):4d} (p/N={len(allp) / N:.3f})  QR estables={estables:.2f}  recuperación 15% ruido={ok:.2f}")
cap = pd.DataFrame(rows)
cap.round(3).to_csv(OUT / "capacidad.csv", index=False)
# ... y sólo patrones aleatorios (≈ ortogonales entre sí): ahí sí se observa la cota 0.15 N
rows = []
for p in [10, 50, 100, 150, 200, 250, 300, 400]:
    fake = rng.choice([-1.0, 1.0], size=(p, N))
    h = Hopfield(update="sync", seed=0).train(fake)
    sample = fake[rng.choice(p, min(p, 20), replace=False)]
    estables = np.mean([np.array_equal(h.recall(x).final, x) for x in sample])
    ok = np.mean([np.array_equal(h.recall(add_noise(x, 0.15, rng), max_iter=50).final, x) for x in sample[:10] for _ in range(3)])
    rows.append(dict(p=p, p_sobre_N=p / N, estables=estables, recuperacion_15pct=ok))
    P(f"  aleatorios p={p:4d} (p/N={p / N:.3f})  estables={estables:.2f}  recuperación 15% ruido={ok:.2f}")
cap_r = pd.DataFrame(rows)
cap_r.round(3).to_csv(OUT / "capacidad_aleatorios.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(cap.patrones_totales, cap.qr_estables, "o-", label="los 2 QR siguen siendo puntos fijos")
axes[0].plot(cap.patrones_totales, cap.recuperacion_15pct, "s--", label="recuperación con 15% de ruido")
axes[0].set_title("2 QR (solapamiento 0.86) + patrones aleatorios"); axes[0].set_xlabel("patrones almacenados")
axes[1].plot(cap_r.p, cap_r.estables, "o-", label="patrones almacenados que son puntos fijos")
axes[1].plot(cap_r.p, cap_r.recuperacion_15pct, "s--", label="recuperación con 15% de ruido")
axes[1].set_title("sólo patrones aleatorios (≈ ortogonales)"); axes[1].set_xlabel("patrones almacenados")
for ax in axes:
    ax.axvline(0.15 * N, color="C3", ls="--", label=f"0.15 N = {0.15 * N:.0f}"); ax.set_ylabel("fracción"); ax.legend(fontsize=8)
fig.suptitle("Capacidad con N = 1369: la cota 0.15 N supone patrones poco correlacionados")
save(fig, OUT / "capacidad.png")
log.close()
