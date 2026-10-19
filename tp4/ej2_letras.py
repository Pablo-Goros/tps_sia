"""Ejercicio 2.2 - Hopfield con letras 5x5 (caso de uso sencillo, recomendado antes de los QR).

  1. Almacenar 4 patrones de letras y asociar matrices 5x5 ruidosas.
  2. Análisis de ortogonalidad y capacidad de almacenamiento.
  3. Ingresar un patrón muy ruidoso e identificar un estado espurio.

Salidas en resultados/hopfield_letras/
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.hopfield import Hopfield, add_noise, hamming, orthogonality_matrix, orthogonality_score
from src.letras import all_letters, letter_vector, to_ascii, best_orthogonal_subset
from src.plots import save, heatmap, pattern_grid

OUT = Path("resultados/hopfield_letras")
OUT.mkdir(parents=True, exist_ok=True)
cfg = json.load(open("configs/hopfield_letras.json"))
rng = np.random.default_rng(cfg["seed"])
N = 25

# ============================================ 1. ortogonalidad de todo el abecedario
letters = all_letters()
K = np.array(list(letters.values()))
G = orthogonality_matrix(K)
fig, ax = heatmap(G, title="Producto interno xi^mu . xi^nu entre letras 5x5 (0 = ortogonales, 25 = iguales)",
                  xticks=list(letters), yticks=list(letters), cmap="coolwarm", vmin=-25, vmax=25,
                  annot=True, fmt="{:.0f}", figsize=(12, 10))
save(fig, OUT / "ortogonalidad_abecedario.png")

ranking = best_orthogonal_subset(4)
pd.DataFrame([(max_, mean, "".join(c)) for max_, mean, c in ranking],
             columns=["max_abs_dot", "mean_abs_dot", "letras"]).head(20).to_csv(OUT / "mejores_grupos.csv", index=False)
pd.DataFrame([(max_, mean, "".join(c)) for max_, mean, c in ranking],
             columns=["max_abs_dot", "mean_abs_dot", "letras"]).tail(10).to_csv(OUT / "peores_grupos.csv", index=False)
print("Grupos de 4 letras más ortogonales:", [("".join(c), m) for m, _, c in ranking[:8]])
print("Grupos menos ortogonales:", [("".join(c), m) for m, _, c in ranking[-5:]])

chosen = cfg["letras"] if cfg.get("letras") else list(ranking[0][2])
worst = list(ranking[-1][2])
print(f"\nLetras elegidas: {chosen}   (peor grupo para comparar: {worst})")

# ====================================================== 2. almacenar 4 patrones
patterns = np.array([letter_vector(c) for c in chosen])
net = Hopfield(update=cfg["update"], seed=cfg["seed"]).train(patterns)
score = orthogonality_score(patterns)
print(f"ortogonalidad grupo elegido: Gram=\n{score['gram']}\n mean|.|/N={score['mean_abs']:.3f} max|.|/N={score['max_abs']:.3f}")
print(f"capacidad teórica 0.15*N = {net.capacity_limit():.2f} patrones; almacenados p = {len(patterns)}")
fig = pattern_grid(patterns, [f"'{c}'" for c in chosen], 5, ncols=4, suptitle="Patrones almacenados")
save(fig, OUT / "patrones.png")
fig, ax = heatmap(net.W, title="Matriz de pesos W = (1/N) K K^T, diag = 0", cmap="coolwarm", figsize=(6, 5))
save(fig, OUT / "pesos.png")

# Estabilidad de los patrones almacenados: ¿cada xi^mu es punto fijo?
for c, xi in zip(chosen, patterns):
    r = net.recall(xi)
    print(f"  '{c}' almacenado -> estable en {r.iterations} iter, igual al original: {np.array_equal(r.final, xi)}, H={r.energies[-1]:.2f}")

# ---- ejemplo detallado: letra con ruido, mostrando cada paso y la energía
c0 = chosen[0]
noisy = add_noise(patterns[0], cfg["ruido_ejemplo"], rng)
r = net.recall(noisy)
print(f"\nEjemplo: '{c0}' con {hamming(noisy, patterns[0])} bits invertidos -> "
      f"{'convergió' if r.converged else 'ciclo'} en {r.iterations} iteraciones; recuperado: {net.match(r.final)}")
print("energías:", [round(e, 2) for e in r.energies])
fig = pattern_grid(r.states, [f"t={t}\nH={e:.1f}" for t, e in enumerate(r.energies)], 5, ncols=len(r.states),
                   suptitle=f"Evolución desde '{c0}' ruidosa (ruido {cfg['ruido_ejemplo']:.0%})")
save(fig, OUT / "ejemplo_evolucion.png")

# ============================================= 3. barrido de ruido (todas las letras)
niveles = cfg["niveles_ruido"]
reps = cfg["repeticiones"]
rows = []
for p_flip in niveles:
    for mu, (c, xi) in enumerate(zip(chosen, patterns)):
        for _ in range(reps):
            z = add_noise(xi, p_flip, rng)
            r = net.recall(z)
            m, neg = net.match(r.final)
            rows.append(dict(ruido=p_flip, letra=c, bits=hamming(z, xi), iter=r.iterations,
                             ok=(m == mu and not neg), espurio=(m is None), ciclo=r.cycle,
                             otro_patron=(m is not None and m != mu), negativo=(m == mu and neg),
                             H0=r.energies[0], Hf=r.energies[-1]))
res = pd.DataFrame(rows)
agg = res.groupby("ruido").agg(recuperado=("ok", "mean"), espurio=("espurio", "mean"), otro=("otro_patron", "mean"),
                               negativo=("negativo", "mean"), ciclo=("ciclo", "mean"), iter=("iter", "mean"),
                               bits=("bits", "mean"))
agg.round(3).to_csv(OUT / "ruido.csv")
print("\nTasa de recuperación por nivel de ruido:\n", agg.round(3).to_string())
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(agg.index, agg.recuperado, "o-", label="patrón correcto")
axes[0].plot(agg.index, agg.espurio, "s--", label="estado espurio")
axes[0].plot(agg.index, agg.otro, "^:", label="otro patrón")
axes[0].plot(agg.index, agg.negativo, "v-.", label="patrón negado")
axes[0].set_xlabel("prob. de invertir cada bit"); axes[0].set_ylabel("fracción"); axes[0].legend()
axes[0].set_title(f"Resultado del recall ({reps} rep. x 4 letras)")
axes[1].plot(agg.index, agg["iter"], "o-", color="C3"); axes[1].set_xlabel("prob. de invertir cada bit")
axes[1].set_ylabel("iteraciones hasta estado estable"); axes[1].set_title("Iteraciones")
save(fig, OUT / "ruido.png")
per_letter = res.groupby(["ruido", "letra"]).ok.mean().unstack()
per_letter.round(3).to_csv(OUT / "ruido_por_letra.csv")

# ====================================================== 4. capacidad de almacenamiento
# Se almacenan p = 1..P letras (en el orden de mejor ortogonalidad) y se mide:
#   - si los patrones almacenados siguen siendo puntos fijos
#   - la recuperación desde 10% de ruido
cap_rows = []
# orden greedy: se agrega en cada paso la letra que menos crosstalk suma al conjunto
order = list(chosen)
while len(order) < 12:
    cur = np.array([letter_vector(c) for c in order])
    best = min((c for c in letters if c not in order),
               key=lambda c: (np.abs(cur @ letter_vector(c)).max(), np.abs(cur @ letter_vector(c)).mean()))
    order.append(best)
for p in range(1, len(order) + 1):
    pats = np.array([letter_vector(c) for c in order[:p]])
    h = Hopfield(update=cfg["update"], seed=cfg["seed"]).train(pats)
    estables = np.mean([np.array_equal(h.recall(xi).final, xi) for xi in pats])
    oks = []
    for mu, xi in enumerate(pats):
        for _ in range(reps):
            r = h.recall(add_noise(xi, 0.1, rng))
            oks.append(np.array_equal(r.final, xi))
    sc = orthogonality_score(pats) if p > 1 else {"mean_abs": 0, "max_abs": 0}
    cap_rows.append(dict(p=p, p_sobre_N=p / N, letras="".join(order[:p]), patrones_estables=estables,
                         recuperacion_10pct=np.mean(oks), crosstalk_medio=sc["mean_abs"], crosstalk_max=sc["max_abs"]))
cap = pd.DataFrame(cap_rows)
cap.round(3).to_csv(OUT / "capacidad.csv", index=False)
print("\nCapacidad:\n", cap.round(3).to_string(index=False))
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(cap.p, cap.patrones_estables, "o-", label="patrones almacenados que son puntos fijos")
ax.plot(cap.p, cap.recuperacion_10pct, "s--", label="recuperación con 10% de ruido")
ax.plot(cap.p, cap.crosstalk_medio, "^:", color="gray", label="crosstalk medio |xi.xi'|/N")
ax.axvline(0.15 * N, color="C3", ls="--", label=f"0.15 N = {0.15 * N:.2f}")
ax.set_xlabel("cantidad de patrones almacenados p"); ax.set_ylabel("fracción"); ax.legend(fontsize=8)
ax.set_title("Capacidad de almacenamiento (N = 25)")
save(fig, OUT / "capacidad.png")

# comparación: el peor grupo de 4 letras (muy correlacionadas) vs el elegido
cmp_rows = []
for label, grp in [("ortogonales " + "".join(chosen), chosen), ("correlacionadas " + "".join(worst), worst)]:
    pats = np.array([letter_vector(c) for c in grp])
    h = Hopfield(update=cfg["update"], seed=cfg["seed"]).train(pats)
    for p_flip in niveles:
        oks = [np.array_equal(h.recall(add_noise(xi, p_flip, rng)).final, xi) for xi in pats for _ in range(reps)]
        cmp_rows.append(dict(grupo=label, ruido=p_flip, recuperado=np.mean(oks)))
cmpdf = pd.DataFrame(cmp_rows).pivot(index="ruido", columns="grupo", values="recuperado")
cmpdf.round(3).to_csv(OUT / "ortogonal_vs_correlacionado.csv")
print("\nOrtogonales vs correlacionadas:\n", cmpdf.round(3).to_string())
fig, ax = plt.subplots(figsize=(7, 4))
cmpdf.plot(ax=ax, marker="o"); ax.set_ylabel("fracción recuperada"); ax.set_title("Efecto de la ortogonalidad (4 letras)")
save(fig, OUT / "ortogonal_vs_correlacionado.png")

# ========================================================= 5. estado espurio
# Patrón muy ruidoso (y también patrones aleatorios) -> buscar un punto fijo que no sea
# ninguno de los almacenados ni sus negativos.
spurious = None
for _ in range(2000):
    z = add_noise(patterns[rng.integers(len(patterns))], cfg["ruido_espurio"], rng)
    r = net.recall(z)
    if r.converged and net.is_spurious(r.final):
        spurious = (z, r)
        break
if spurious is None:
    for _ in range(5000):
        z = rng.choice([-1.0, 1.0], size=N)
        r = net.recall(z)
        if r.converged and net.is_spurious(r.final):
            spurious = (z, r)
            break
if spurious:
    z, r = spurious
    ov = net.overlaps(r.final)
    print(f"\nESTADO ESPURIO encontrado desde patrón con ruido {cfg['ruido_espurio']:.0%}: "
          f"{r.iterations} iter, H={r.energies[-1]:.2f}; overlaps con {chosen}: {np.round(ov, 2)}")
    print(to_ascii(r.final))
    # ¿Es combinación de los almacenados? (mezcla de 3 patrones = típico estado espurio)
    mix_found = None
    for a, b, c in combinations(range(len(patterns)), 3):
        for sa in (1, -1):
            for sb in (1, -1):
                for sc in (1, -1):
                    mix = np.sign(sa * patterns[a] + sb * patterns[b] + sc * patterns[c])
                    if np.array_equal(mix, r.final):
                        mix_found = f"sign({sa:+d}'{chosen[a]}' {sb:+d}'{chosen[b]}' {sc:+d}'{chosen[c]}')"
    print("mezcla de 3 patrones:", mix_found or "no es una mezcla simple de 3")
    energies = {c: net.energy(xi) for c, xi in zip(chosen, patterns)}
    print("energía de los patrones almacenados:", {k: round(v, 2) for k, v in energies.items()},
          f" | energía del espurio: {r.energies[-1]:.2f}")
    fig = pattern_grid([z] + r.states[1:] + list(patterns),
                       [f"consulta\nH={r.energies[0]:.1f}"] + [f"t={t}\nH={e:.1f}" for t, e in enumerate(r.energies[1:], 1)]
                       + [f"'{c}' H={energies[c]:.1f}" for c in chosen], 5, ncols=max(len(r.states), 4) ,
                       suptitle=f"Estado espurio: el punto fijo no es ninguna letra almacenada ({mix_found or 'mínimo local extra'})")
    save(fig, OUT / "espurio.png")
    json.dump({"consulta": z.tolist(), "final": r.final.tolist(), "energias": r.energies,
               "overlaps": dict(zip(chosen, ov.tolist())), "mezcla": mix_found}, open(OUT / "espurio.json", "w"), indent=1)

# energía: trayectorias para varios niveles de ruido
fig, ax = plt.subplots(figsize=(7, 4))
for p_flip in niveles:
    z = add_noise(patterns[1], p_flip, rng)
    r = net.recall(z)
    ax.plot(r.energies, "o-", label=f"ruido {p_flip:.0%} -> {'ok' if np.array_equal(r.final, patterns[1]) else ('espurio' if net.is_spurious(r.final) else 'otro')}")
ax.set_xlabel("iteración"); ax.set_ylabel("H"); ax.set_title(f"Energía durante el recall ('{chosen[1]}')"); ax.legend(fontsize=7)
save(fig, OUT / "energia.png")
