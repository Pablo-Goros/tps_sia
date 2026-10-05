"""Ejercicio 1 - OPCIONAL: perceptrón simple no lineal con activación ReLU (TP3 - SIA).

"Utilizar otra función de activación en el perceptrón simple no lineal, como ReLU,
¿qué efecto tiene en las conclusiones anteriores?"

Se repiten, con ReLU, los dos estudios del ejercicio, comparando contra el lineal y
la logística con las configuraciones que se eligieron en cada etapa:

    R1 Aprendizaje (7500 muestras): barrido de η, curvas de aprendizaje (5 semillas),
       fracción de muestras sin gradiente por época (ReLU: h <= 0, "neuronas
       muertas"; logística: sigmoide saturada) y error por zona del objetivo.
    R2 Generalización (k-fold 5 x 3 estratificado, mismo criterio: recall >= 95 % y
       máxima precision): grilla de η y features, barrido de épocas, y estimación
       final con precision / recall / F1 en entrenamiento y prueba por fold.

ReLU: θ(h) = max(0, h), θ'(h) = 1 si h > 0 y 0 si no. No tiene β.
Este script sólo ejecuta y persiste; el análisis está en graficos_relu.py.

    python -m tps_sia.tp3.ej1.src.experimentos_relu [--procesos 7]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict

import numpy as np

from .datos import Normalizador, cargar, estratos, k_fold
from .experimentos_generalizacion import (FEATURES_RUIDO, K, REPETICIONES_CV, UMBRAL_BIGMODEL,
                                          _ejecutar, _prf, _umbral_recall)
from .metricas import resumen
from .perceptron import PerceptronSimple

EJERCICIO = Path(__file__).resolve().parents[1]
RUTA_DATOS = EJERCICIO / "datos" / "fraud_dataset.csv"
DIR_SALIDA = EJERCICIO / "salidas" / "relu"

# Configuraciones elegidas en las etapas anteriores (README, Ejercicio 1).
APRENDIZAJE = {"lineal": {"eta": 1e-4, "beta": 0.5},
               "logistica": {"eta": 5e-3, "beta": 0.5}}
GENERALIZACION = {
    "logistica": {"eta": 1e-3, "beta": 0.25, "epocas": 60, "features": "sin ruido (6)"},
    "lineal": {"eta": 1e-4, "beta": 0.5, "epocas": 60, "features": "todos (9)"},
}

SEMILLAS = [1, 2, 3, 4, 5]
ETAS_R1 = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
EPOCAS_BARRIDO_ETA = 60
EPOCAS_CURVAS = 400

ETAS_R2 = [1e-4, 5e-4, 1e-3, 5e-3]
EPOCAS_R2 = 60
EPOCAS_BARRIDO = [10, 20, 40, 60, 100, 200, 400]


# --------------------------------------------------------------------------- #
def _tarea_aprendizaje(args) -> Dict:
    """Entrena con las 7500 muestras y devuelve curva, diagnóstico y predicciones."""
    clave, Xs, y, activacion, eta, beta, epocas, semilla = args
    m = PerceptronSimple(Xs.shape[1], activacion, eta=eta, beta=beta, tamano_lote=1,
                         semilla=semilla)
    h = m.entrenar(Xs, y, epocas=epocas, registrar_diagnostico=True)
    o = m.predecir(Xs)
    return {"clave": clave, "mse": h.mse, "sin_gradiente": h.fraccion_saturada,
            "pred": o, "w": m.w.copy(), "tiempo": h.tiempo_segundos}


def _ejecutar_aprendizaje(tareas, procesos, etiqueta) -> Dict:
    t0 = time.perf_counter()
    salida = {}
    with ProcessPoolExecutor(max_workers=procesos) as pool:
        for i, r in enumerate(pool.map(_tarea_aprendizaje, tareas), 1):
            salida[r["clave"]] = r
            if i % max(1, len(tareas) // 5) == 0 or i == len(tareas):
                print(f"  [{etiqueta}] {i}/{len(tareas)} ({time.perf_counter() - t0:.0f}s)")
    return salida


def _diverge(mse: float) -> bool:
    return not np.isfinite(mse) or mse > 10


def r1_aprendizaje(Xs, y, procesos) -> Dict:
    """R1: el estudio de aprendizaje con ReLU, junto al lineal y la logística."""
    # Barrido de η para ReLU (como E1 del estudio de aprendizaje).
    tareas = [(("eta", eta, s), Xs, y, "relu", eta, 0.5, EPOCAS_BARRIDO_ETA, s)
              for eta in ETAS_R1 for s in SEMILLAS[:3]]
    res = _ejecutar_aprendizaje(tareas, procesos, "R1 barrido η")
    barrido = []
    for eta in ETAS_R1:
        finales = [res[("eta", eta, s)]["mse"][-1] for s in SEMILLAS[:3]]
        ok = [m for m in finales if not _diverge(m)]
        barrido.append({"eta": eta, "mse_medio": float(np.mean(ok)) if ok else None,
                        "corridas_divergentes": len(finales) - len(ok),
                        "muertas_final": float(np.mean(
                            [res[("eta", eta, s)]["sin_gradiente"][-1] for s in SEMILLAS[:3]]))})
        print(f"  ReLU η={eta:<7g} MSE={barrido[-1]['mse_medio']}  "
              f"inactivas={barrido[-1]['muertas_final']:.1%}  div={barrido[-1]['corridas_divergentes']}")
    validos = [b for b in barrido if b["mse_medio"] is not None]
    eta_relu = min(validos, key=lambda b: b["mse_medio"])["eta"]
    print(f"  -> η ReLU = {eta_relu}")

    # Curvas largas, 5 semillas, para los tres modelos.
    configs = {**APRENDIZAJE, "relu": {"eta": eta_relu, "beta": 0.5}}
    tareas = [(("curva", act, s), Xs, y, act, cfg["eta"], cfg["beta"], EPOCAS_CURVAS, s)
              for act, cfg in configs.items() for s in SEMILLAS]
    res = _ejecutar_aprendizaje(tareas, procesos, "R1 curvas")

    zonas = {"bajo (ζ<0.2)": y < 0.2, "medio (0.2≤ζ≤0.8)": (y >= 0.2) & (y <= 0.8),
             "alto (ζ>0.8)": y > 0.8}
    modelos = {}
    for act, cfg in configs.items():
        corridas = [res[("curva", act, s)] for s in SEMILLAS]
        curvas = np.array([c["mse"] for c in corridas])
        metr = [resumen(y, c["pred"]) for c in corridas]
        mejor = int(np.argmin([m["mse"] for m in metr]))
        o = corridas[mejor]["pred"]
        curva = curvas.mean(axis=0)
        mejora = curva[0] - curva[-1]
        ep99 = int(np.where(curva <= curva[0] - 0.99 * mejora)[0][0] + 1)
        modelos[act] = {
            "config": cfg,
            "mse_medio_por_epoca": curva.tolist(),
            "mse_desvio_por_epoca": curvas.std(axis=0).tolist(),
            "sin_gradiente_por_epoca": np.mean([c["sin_gradiente"] for c in corridas], axis=0).tolist(),
            "metricas_medias": {k: float(np.mean([m[k] for m in metr])) for k in metr[0]},
            "metricas_desvio": {k: float(np.std([m[k] for m in metr])) for k in metr[0]},
            "epocas_99pct_mejora": ep99,
            "predicciones": o.tolist(),
            "error_por_zona": {z: {"n": int(mk.sum()), "mse": float(np.mean((o - y)[mk] ** 2)),
                                   "sesgo": float(np.mean((o - y)[mk]))}
                               for z, mk in zonas.items()},
            "salidas_en_cero": float(np.mean(o == 0.0)),
            "salidas_mayores_a_1": float(np.mean(o > 1.0)),
            "salidas_menores_a_0": float(np.mean(o < 0.0)),
            "tiempo_medio_s": float(np.mean([c["tiempo"] for c in corridas])),
        }
        m = modelos[act]["metricas_medias"]
        print(f"  {act:10s} MSE={m['mse']:.6f} R2={m['r2']:.4f} fuera [0,1]={m['fuera_de_rango']:.2%} "
              f"99% mejora en {ep99} épocas  sin gradiente al final="
              f"{modelos[act]['sin_gradiente_por_epoca'][-1]:.1%}")
    return {"barrido_eta": barrido, "eta_relu": eta_relu, "modelos": modelos}


# --------------------------------------------------------------------------- #
def r2_generalizacion(X, y, clase, est, features, procesos) -> Dict:
    """R2: el estudio de generalización con ReLU, con el mismo protocolo k-fold."""
    conjuntos = {"todos (9)": list(range(len(features))),
                 "sin ruido (6)": [i for i, f in enumerate(features) if f not in FEATURES_RUIDO]}
    particiones = {r: k_fold(len(y), K, np.random.default_rng(500 + r), est)
                   for r in range(REPETICIONES_CV)}

    def _oof_criterio(res, clave_base):
        oof = np.zeros((REPETICIONES_CV, len(y)))
        for r, folds in particiones.items():
            for f, (_, va) in enumerate(folds):
                oof[r, va] = res[clave_base + (r, f)]["pred_val"]
        crit = [_prf(clase, oof[r], _umbral_recall(clase, oof[r])) for r in range(REPETICIONES_CV)]
        return {k: float(np.mean([c[k] for c in crit])) for k in ("umbral", "precision", "recall", "f1")}

    # Grilla η x features (ReLU no tiene β).
    tareas = [((fs, eta, r, f), dict(Xtr=X[tr][:, cols], ytr=y[tr], Xva=X[va][:, cols], yva=y[va],
                                     eta=eta, beta=0.5, epocas=EPOCAS_R2, semilla=10 * r + f,
                                     activacion="relu"))
              for fs, cols in conjuntos.items() for eta in ETAS_R2
              for r, folds in particiones.items() for f, (tr, va) in enumerate(folds)]
    res = _ejecutar(tareas, procesos, "R2 grilla")
    grilla = []
    for fs in conjuntos:
        for eta in ETAS_R2:
            c = _oof_criterio(res, (fs, eta))
            mse = [res[(fs, eta, r, f)]["mse_validacion"][-1]
                   for r in range(REPETICIONES_CV) for f in range(K)]
            grilla.append({"features": fs, "eta": eta, "mse_validacion": float(np.mean(mse)), **c})
            print(f"  ReLU {fs:14s} η={eta:<7g} MSE val={grilla[-1]['mse_validacion']:.6f} "
                  f"recall={c['recall']:.3f} precision={c['precision']:.3f} F1={c['f1']:.3f}")
    elegida = max(grilla, key=lambda g: (round(g["precision"], 3), g["f1"]))
    cols = conjuntos[elegida["features"]]
    print(f"  -> ReLU elegida: {elegida['features']} η={elegida['eta']}")

    # Barrido de épocas para la configuración elegida.
    tareas = [((ep, r, f), dict(Xtr=X[tr][:, cols], ytr=y[tr], Xva=X[va][:, cols], yva=y[va],
                                eta=elegida["eta"], beta=0.5, epocas=ep, semilla=10 * r + f,
                                activacion="relu"))
              for ep in EPOCAS_BARRIDO for r, folds in particiones.items()
              for f, (tr, va) in enumerate(folds)]
    res = _ejecutar(tareas, procesos, "R2 épocas")
    epocas = []
    for ep in EPOCAS_BARRIDO:
        c = _oof_criterio(res, (ep,))
        epocas.append({"epocas": ep, **c})
        print(f"  {ep:4d} épocas recall={c['recall']:.3f} precision={c['precision']:.3f} F1={c['f1']:.3f}")
    ep_relu = max(epocas, key=lambda e: (round(e["precision"], 3), -e["epocas"]))["epocas"]
    print(f"  -> épocas ReLU: {ep_relu}")

    # Estimación final: particiones nuevas, umbral elegido en cada fold de entrenamiento,
    # métricas en entrenamiento y prueba. Los tres modelos sobre los mismos folds.
    finales = {
        "relu": {"eta": elegida["eta"], "beta": 0.5, "epocas": ep_relu,
                 "features": elegida["features"]},
        **GENERALIZACION,
    }
    part_eval = {r: k_fold(len(y), K, np.random.default_rng(900 + r), est)
                 for r in range(REPETICIONES_CV)}
    tareas = []
    for modelo, cfg in finales.items():
        c = conjuntos[cfg["features"]]
        for r, folds in part_eval.items():
            for f, (tr, va) in enumerate(folds):
                tareas.append(((modelo, r, f), dict(
                    Xtr=X[tr][:, c], ytr=y[tr], Xva=X[va][:, c], yva=y[va], eta=cfg["eta"],
                    beta=cfg["beta"], epocas=cfg["epocas"], semilla=10 * r + f, activacion=modelo)))
    res = _ejecutar(tareas, procesos, "R2 evaluación")
    evaluacion = {}
    for modelo, cfg in finales.items():
        folds, oof = [], np.zeros((REPETICIONES_CV, len(y)))
        for r, parts in part_eval.items():
            for f, (tr, va) in enumerate(parts):
                x = res[(modelo, r, f)]
                u = _umbral_recall(clase[tr], x["pred_ent"])
                oof[r, va] = x["pred_val"]
                folds.append({"repeticion": r, "fold": f, "umbral": u,
                              "entrenamiento": _prf(clase[tr], x["pred_ent"], u),
                              "prueba": _prf(clase[va], x["pred_val"], u),
                              "mse_prueba": float(np.mean((y[va] - x["pred_val"]) ** 2))})
        evaluacion[modelo] = {"config": cfg, "folds": folds, "oof": oof.tolist()}
        print(f"  {modelo:10s} " + "  ".join(
            f"{k}={np.mean([f['prueba'][k] for f in folds]):.3f}" for k in ("recall", "precision", "f1"))
            + f"  umbral={np.mean([f['umbral'] for f in folds]):.3f}")
    return {"grilla": grilla, "elegida": {"features": elegida["features"], "eta": elegida["eta"]},
            "epocas": epocas, "epocas_relu": ep_relu, "evaluacion": evaluacion}


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    os.makedirs(DIR_SALIDA, exist_ok=True)
    t0 = time.perf_counter()

    X, y, features, df = cargar(RUTA_DATOS)
    clase = df["flagged_fraud"].to_numpy()
    assert np.array_equal(clase == 1, y >= UMBRAL_BIGMODEL)
    Xs = Normalizador.ajustar(X).aplicar(X)

    print("== R1: aprendizaje con ReLU (7500 muestras) ==")
    aprendizaje = r1_aprendizaje(Xs, y, args.procesos)
    print("\n== R2: generalización con ReLU (k-fold) ==")
    generalizacion = r2_generalizacion(X, y, clase, estratos(y, clase), features, args.procesos)

    resultados = {"aprendizaje": aprendizaje, "generalizacion": generalizacion,
                  "y": y.tolist(), "clase": clase.tolist(),
                  "tiempo_total_s": time.perf_counter() - t0}
    ruta = os.path.join(DIR_SALIDA, "resultados.json")
    with open(ruta, "w") as fh:
        json.dump(resultados, fh, default=lambda o: o.tolist() if isinstance(o, np.ndarray)
                  else float(o))
    print(f"\nResultados en {ruta} ({resultados['tiempo_total_s'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
