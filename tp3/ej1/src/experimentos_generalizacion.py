"""Ejercicio 1 - ESTUDIO DE GENERALIZACIÓN (TP3 - SIA).

Perceptrón seleccionado en la etapa de aprendizaje: NO LINEAL con activación
LOGÍSTICA. Se responde:

    a) métricas (elegidas por el grupo): precision, recall y F1, con el RECALL
       como prioridad (un fraude no detectado es lo más caro); se descarta accuracy.
    b) ¿qué estrategia utilizaron para manipular el conjunto de datos?
       ¿cómo se elige el mejor conjunto de entrenamiento?
    c) ¿cuál es el mejor modelo para presentar al cliente?
    +  recomendación del umbral de detección de fraude.

Estrategia: VALIDACIÓN CRUZADA k-FOLD sobre las 7500 muestras. Los datos se
dividen en k partes; se entrena con k-1 y se evalúa con la restante, rotando,
y se reporta media ± desvío entre folds. Se repite con 3 particiones distintas.

    G1 ¿Qué k y con qué partición? k = 3, 5, 10, aleatoria vs. estratificada;
       y curva de tamaño del conjunto de entrenamiento.
    G2 Selección de hiperparámetros (η, β, features) por k-fold, con el criterio
       de las métricas elegidas: recall ≥ 95 % y, entre los que lo cumplen, la
       mayor precision. El MSE es sólo el costo que minimiza el entrenamiento.
    G2b Épocas como hiperparámetro para la configuración elegida (mismo criterio):
       si el modelo elegido no termina de converger, cuántas épocas entrenar
       también es una decisión que hay que validar.
    G3 Umbral de detección a partir de las predicciones fuera de fold (cada
       muestra predicha por un modelo que no la vio), con el mismo criterio.
    G4 Estimación de generalización del modelo elegido por k-fold: en cada fold
       el umbral se elige SÓLO con el fold de entrenamiento y se mide en el de
       validación. Después se entrena el modelo final con las 7500 muestras.

En todos los casos el normalizador z-score se ajusta SÓLO con el fold de
entrenamiento (si no, la validación "ve" su propia media y desvío).
flagged_fraud nunca es entrada ni objetivo: sólo estratifica y evalúa el umbral.
Este script sólo ejecuta y persiste; el análisis está en graficos_generalizacion.py.

    python3 -m tp3.src.experimentos_generalizacion [--procesos 7]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Sequence

import numpy as np

from .datos import Normalizador, cargar, estratos, k_fold
from .metricas import clasificacion, resumen
from .perceptron import PerceptronSimple

RUTA_DATOS = "tp3/datos/fraud_dataset.csv"
DIR_SALIDA = "tp3/salidas/generalizacion"

K = 5
REPETICIONES_CV = 3
SEMILLAS_FINAL = [1, 2, 3]

# Punto de partida: lo elegido en la etapa de aprendizaje.
CONFIG_APRENDIZAJE = {"eta": 5e-3, "beta": 0.5}
ETAS = [1e-3, 5e-3, 1e-2]
BETAS = [0.25, 0.5, 1.0]
# Features con |r| < 0.03 contra el objetivo en la exploración (ver README §2).
FEATURES_RUIDO = ["timestamp", "time_since_last_login_s", "device_screen_resolution"]
EPOCAS_CV = 60
EPOCAS_BARRIDO = [10, 20, 40, 60, 100, 200, 400]   # G2b: épocas como hiperparámetro
EPOCAS_G1 = 30

KS_G1 = [3, 5, 10]
FRACCIONES_TAMANO = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
ACTUALIZACIONES_TAMANO = 30 * 6000   # mismo nº de actualizaciones para cada tamaño

UMBRALES = np.round(np.arange(0.50, 0.99 + 1e-9, 0.005), 3)
UMBRAL_BIGMODEL = 0.85       # flagged_fraud == (big_model_fraud_probability >= 0.85)
RECALL_MINIMO = 0.95
RAZONES_COSTO = [1, 2, 5, 10, 20]   # costo de un fraude no detectado / costo de una falsa alarma


# --------------------------------------------------------------------------- #
def _entrenar(Xtr, ytr, Xva, yva, eta, beta, epocas, semilla, activacion="logistica") -> Dict:
    """Ajusta el normalizador en entrenamiento, entrena y evalúa en validación."""
    norm = Normalizador.ajustar(Xtr)
    a, b = norm.aplicar(Xtr), norm.aplicar(Xva)
    m = PerceptronSimple(Xtr.shape[1], activacion, eta=eta, beta=beta,
                         tamano_lote=1, semilla=semilla)
    h = m.entrenar(a, ytr, epocas=epocas, registrar_diagnostico=False, X_val=b, y_val=yva)
    return {"mse_entrenamiento": h.mse, "mse_validacion": h.mse_validacion,
            "pred_ent": m.predecir(a), "pred_val": m.predecir(b), "w": m.w.copy(),
            "media": norm.media, "desvio": norm.desvio}


def _tarea(args) -> Dict:
    clave, kwargs = args
    return {"clave": clave, **_entrenar(**kwargs)}


def _ejecutar(tareas: List, procesos: int, etiqueta: str) -> Dict:
    t0 = time.perf_counter()
    salida = {}
    with ProcessPoolExecutor(max_workers=procesos) as pool:
        for i, r in enumerate(pool.map(_tarea, tareas), 1):
            salida[r["clave"]] = r
            if i % max(1, len(tareas) // 10) == 0 or i == len(tareas):
                print(f"  [{etiqueta}] {i}/{len(tareas)} entrenamientos "
                      f"({time.perf_counter() - t0:.0f}s)")
    return salida


def _umbral_recall(clase, puntaje) -> float:
    """Criterio del grupo: el umbral de mayor precision entre los que dan recall ≥ 95 %.

    Recorre los umbrales de mayor a menor y devuelve el primero (el más alto, o sea
    el de menos falsas alarmas) que alcanza el recall mínimo.
    """
    for u in UMBRALES[::-1]:
        if clasificacion(clase, puntaje, u)["recall"] >= RECALL_MINIMO:
            return float(u)
    return float(UMBRALES[0])


def _prf(clase, puntaje, umbral) -> Dict:
    m = clasificacion(clase, puntaje, umbral)
    return {k: m[k] for k in ("umbral", "precision", "recall", "f1", "vp", "fp", "fn", "vn")}


# --------------------------------------------------------------------------- #
def g1_estrategia_kfold(X, y, clase, est, procesos) -> Dict:
    """G1a: ¿qué k y con qué partición? (mismo modelo, cambia sólo la partición)

    Para cada k y cada forma de partir (aleatoria / estratificada) se mide la
    dispersión ENTRE FOLDS de precision, recall y F1 (umbral elegido con el fold
    de entrenamiento): cuanto más parecidos son los folds entre sí, más confiable
    es la media que reporta la validación cruzada.
    """
    tareas = []
    for k in KS_G1:
        for estrategia in ("aleatoria", "estratificada"):
            for r in range(REPETICIONES_CV):
                rng = np.random.default_rng(1000 + 10 * k + r)
                folds = k_fold(len(y), k, rng, est if estrategia == "estratificada" else None)
                for f, (tr, va) in enumerate(folds):
                    tareas.append(((k, estrategia, r, f), dict(
                        Xtr=X[tr], ytr=y[tr], Xva=X[va], yva=y[va], epocas=EPOCAS_G1,
                        semilla=10 * r + f, **CONFIG_APRENDIZAJE), va))
    idx_val = {t[0]: t[2] for t in tareas}
    res = _ejecutar([t[:2] for t in tareas], procesos, "G1 estrategia k-fold")

    salida = []
    for k in KS_G1:
        for estrategia in ("aleatoria", "estratificada"):
            folds = []
            for r in range(REPETICIONES_CV):
                for f in range(k):
                    c = (k, estrategia, r, f)
                    va, p = idx_val[c], res[c]["pred_val"]
                    tr = np.setdiff1d(np.arange(len(y)), va)
                    u = _umbral_recall(clase[tr], res[c]["pred_ent"])
                    folds.append({
                        "repeticion": r, "fold": f, "n_validacion": len(va),
                        "fraudes_validacion": int(clase[va].sum()),
                        "mse_validacion": res[c]["mse_validacion"][-1],
                        "mse_entrenamiento": res[c]["mse_entrenamiento"][-1],
                        "proporcion_fraude_validacion": float(clase[va].mean()),
                        **_prf(clase[va], p, u),
                    })
            fila = {"k": k, "estrategia": estrategia, "folds": folds,
                    "n_entrenamientos": k * REPETICIONES_CV}
            for met in ("recall", "precision", "f1", "mse_validacion"):
                v = [f[met] for f in folds]
                medias = [np.mean([f[met] for f in folds if f["repeticion"] == r])
                          for r in range(REPETICIONES_CV)]
                fila[met] = {"media": float(np.mean(v)), "desvio_entre_folds": float(np.std(v, ddof=1)),
                             "desvio_entre_repeticiones": float(np.std(medias, ddof=1))}
            fr = [f["fraudes_validacion"] for f in folds]
            fila["fraudes_por_fold"] = {"min": int(min(fr)), "max": int(max(fr))}
            salida.append(fila)
            print(f"  k={k:<2d} {estrategia:13s} " + "  ".join(
                f"{met}={fila[met]['media']:.3f}±{fila[met]['desvio_entre_folds']:.3f}"
                for met in ("recall", "precision", "f1")) +
                f"  fraudes/fold {min(fr)}–{max(fr)}")
    return {"configuraciones": salida}


def g1_tamano(X, y, est, procesos) -> List[Dict]:
    """G1b: curva de tamaño de entrenamiento (k-fold estratificado).

    Todos los tamaños reciben el mismo número de actualizaciones de pesos, para
    que la diferencia sea de DATOS y no de presupuesto de entrenamiento.
    """
    folds = k_fold(len(y), K, np.random.default_rng(7), est)
    tareas = []
    for frac in FRACCIONES_TAMANO:
        for f, (tr, va) in enumerate(folds):
            rng = np.random.default_rng(100 * f + int(frac * 1000))
            n = max(10, int(round(frac * len(tr))))
            sub = np.sort(rng.choice(tr, size=n, replace=False))
            epocas = int(np.ceil(ACTUALIZACIONES_TAMANO / n))
            tareas.append(((frac, f), dict(Xtr=X[sub], ytr=y[sub], Xva=X[va], yva=y[va],
                                           epocas=epocas, semilla=f, **CONFIG_APRENDIZAJE)))
    res = _ejecutar(tareas, procesos, "G1 tamaño")
    filas = []
    for frac in FRACCIONES_TAMANO:
        ent = [res[(frac, f)]["mse_entrenamiento"][-1] for f in range(K)]
        val = [res[(frac, f)]["mse_validacion"][-1] for f in range(K)]
        n = max(10, int(round(frac * len(folds[0][0]))))
        filas.append({"fraccion": frac, "n_entrenamiento": n,
                      "mse_entrenamiento": float(np.mean(ent)),
                      "mse_entrenamiento_desvio": float(np.std(ent, ddof=1)),
                      "mse_validacion": float(np.mean(val)),
                      "mse_validacion_desvio": float(np.std(val, ddof=1))})
        print(f"  n={n:5d}  MSE ent={filas[-1]['mse_entrenamiento']:.5f}  "
              f"val={filas[-1]['mse_validacion']:.5f} ± {filas[-1]['mse_validacion_desvio']:.5f}")
    return filas


def g2_validacion_cruzada(X, y, clase, est, features, procesos) -> Dict:
    """G2: grilla η x β x conjunto de features, k-fold estratificado x repeticiones.

    Criterio de selección = el de las métricas elegidas: sobre las predicciones
    fuera de fold, el umbral de mayor precision con recall ≥ 95 %; gana la
    configuración con mayor precision a ese recall (desempate: F1).
    """
    conjuntos = {
        "todos (9)": list(range(len(features))),
        "sin ruido (6)": [i for i, f in enumerate(features) if f not in FEATURES_RUIDO],
    }
    particiones = {r: k_fold(len(y), K, np.random.default_rng(500 + r), est)
                   for r in range(REPETICIONES_CV)}
    tareas = []
    for nombre_fs, cols in conjuntos.items():
        for eta, beta in itertools.product(ETAS, BETAS):
            for r, folds in particiones.items():
                for f, (tr, va) in enumerate(folds):
                    tareas.append(((nombre_fs, eta, beta, r, f), dict(
                        Xtr=X[tr][:, cols], ytr=y[tr], Xva=X[va][:, cols], yva=y[va],
                        eta=eta, beta=beta, epocas=EPOCAS_CV, semilla=10 * r + f)))
    res = _ejecutar(tareas, procesos, "G2 validación cruzada")

    configs = []
    for nombre_fs in conjuntos:
        for eta, beta in itertools.product(ETAS, BETAS):
            claves = [(nombre_fs, eta, beta, r, f)
                      for r in range(REPETICIONES_CV) for f in range(K)]
            val = np.array([res[c]["mse_validacion"] for c in claves])
            ent = np.array([res[c]["mse_entrenamiento"] for c in claves])
            # Predicciones fuera de fold: una por muestra y por repetición.
            oof = np.zeros((REPETICIONES_CV, len(y)))
            for (_, _, _, r, f) in claves:
                oof[r, particiones[r][f][1]] = res[(nombre_fs, eta, beta, r, f)]["pred_val"]
            criterio = [_prf(clase, oof[r], _umbral_recall(clase, oof[r]))
                        for r in range(REPETICIONES_CV)]
            curva = val.mean(axis=0)
            configs.append({
                "features": nombre_fs, "eta": eta, "beta": beta,
                "columnas": [features[i] for i in conjuntos[nombre_fs]],
                "mse_validacion": float(val[:, -1].mean()),
                "mse_validacion_desvio": float(val[:, -1].std(ddof=1)),
                "mse_entrenamiento": float(ent[:, -1].mean()),
                "mejor_epoca": int(np.argmin(curva) + 1),
                "mse_validacion_mejor_epoca": float(curva.min()),
                "curva_validacion": curva.tolist(),
                "curva_validacion_desvio": val.std(axis=0, ddof=1).tolist(),
                "curva_entrenamiento": ent.mean(axis=0).tolist(),
                **{f"{k}_criterio": float(np.mean([c[k] for c in criterio]))
                   for k in ("umbral", "precision", "recall", "f1")},
                "precision_criterio_desvio": float(np.std([c["precision"] for c in criterio], ddof=1)),
                "oof": oof.tolist(),
            })
            c = configs[-1]
            print(f"  {nombre_fs:14s} η={eta:<6g} β={beta:<5g} MSE val={c['mse_validacion']:.6f} "
                  f"| recall={c['recall_criterio']:.3f} precision={c['precision_criterio']:.3f} "
                  f"F1={c['f1_criterio']:.3f} (u={c['umbral_criterio']:.3f})")

    mejor = max(configs, key=lambda c: (round(c["precision_criterio"], 3), c["f1_criterio"]))
    print(f"  -> elegido: {mejor['features']} η={mejor['eta']} β={mejor['beta']} "
          f"(precision {mejor['precision_criterio']:.3f} con recall {mejor['recall_criterio']:.3f})")
    # Las épocas del modelo final se eligen después, en G2b.
    return {"k": K, "repeticiones": REPETICIONES_CV, "epocas": EPOCAS_CV,
            "configuraciones": configs,
            "elegida": {k: mejor[k] for k in ("features", "eta", "beta", "columnas")}}


def g2b_epocas(X, y, clase, est, cols, cfg, procesos) -> Dict:
    """G2b: barrido de épocas para la configuración elegida, con las particiones de G2.

    Además del criterio sobre las predicciones fuera de fold (con el que se eligen las
    épocas), en cada fold se calculan precision, recall y F1 en ENTRENAMIENTO y en PRUEBA
    con el umbral elegido en el fold de entrenamiento: es el procedimiento de la
    diapositiva 17 (métricas en ambos conjuntos para cada cantidad de épocas) y la base de
    la clasificación underfitting / overfitting de la diapositiva 30.
    """
    particiones = {r: k_fold(len(y), K, np.random.default_rng(500 + r), est)
                   for r in range(REPETICIONES_CV)}
    tareas = [((ep, r, f), dict(Xtr=X[tr][:, cols], ytr=y[tr], Xva=X[va][:, cols], yva=y[va],
                                eta=cfg["eta"], beta=cfg["beta"], epocas=ep, semilla=10 * r + f))
              for ep in EPOCAS_BARRIDO for r, folds in particiones.items()
              for f, (tr, va) in enumerate(folds)]
    res = _ejecutar(tareas, procesos, "G2b épocas")
    filas = []
    for ep in EPOCAS_BARRIDO:
        oof = np.zeros((REPETICIONES_CV, len(y)))
        for r, folds in particiones.items():
            for f, (_, va) in enumerate(folds):
                oof[r, va] = res[(ep, r, f)]["pred_val"]
        crit = [_prf(clase, oof[r], _umbral_recall(clase, oof[r])) for r in range(REPETICIONES_CV)]
        mse = [res[(ep, r, f)]["mse_validacion"][-1]
               for r in range(REPETICIONES_CV) for f in range(K)]
        por_fold = []
        for r, folds in particiones.items():
            for f, (tr, va) in enumerate(folds):
                x = res[(ep, r, f)]
                u = _umbral_recall(clase[tr], x["pred_ent"])
                por_fold.append({"repeticion": r, "fold": f, "umbral": u,
                                 "entrenamiento": _prf(clase[tr], x["pred_ent"], u),
                                 "prueba": _prf(clase[va], x["pred_val"], u)})
        filas.append({"epocas": ep, "mse_validacion": float(np.mean(mse)),
                      "folds": por_fold,
                      **{k: float(np.mean([c[k] for c in crit]))
                         for k in ("umbral", "precision", "recall", "f1")},
                      "precision_desvio": float(np.std([c["precision"] for c in crit], ddof=1)),
                      "oof": oof.tolist()})
        f1e = np.mean([x["entrenamiento"]["f1"] for x in por_fold])
        f1p = np.mean([x["prueba"]["f1"] for x in por_fold])
        print(f"  {ep:4d} épocas  MSE val={filas[-1]['mse_validacion']:.6f}  "
              f"recall={filas[-1]['recall']:.3f} precision={filas[-1]['precision']:.3f} "
              f"F1={filas[-1]['f1']:.3f} (u={filas[-1]['umbral']:.3f})  "
              f"| por fold F1 ent={f1e:.3f} prueba={f1p:.3f}")
    # Mayor precision (a 3 decimales); a igualdad, menos épocas (más barato).
    mejor = max(filas, key=lambda f: (round(f["precision"], 3), -f["epocas"]))
    print(f"  -> épocas elegidas: {mejor['epocas']}")
    return {"filas": filas, "epocas_elegidas": mejor["epocas"]}


def _barrido_umbral(clase, puntajes: np.ndarray) -> List[Dict]:
    """Métricas por umbral, promediadas sobre las filas de `puntajes` (repeticiones)."""
    filas = []
    for u in UMBRALES:
        ms = [clasificacion(clase, p, u) for p in puntajes]
        filas.append({k: float(np.mean([m[k] for m in ms])) for k in ms[0]})
    return filas


def g3_umbral(clase, oof: np.ndarray) -> Dict:
    """G3: umbral de detección a partir de las predicciones fuera de fold."""
    barrido = _barrido_umbral(clase, oof)
    f1 = max(barrido, key=lambda f: f["f1"])
    con_recall = [f for f in barrido if f["recall"] >= RECALL_MINIMO]
    alto_recall = max(con_recall, key=lambda f: f["precision"]) if con_recall else None
    por_costo = []
    for razon in RAZONES_COSTO:
        costo = [razon * f["fn"] + f["fp"] for f in barrido]
        i = int(np.argmin(costo))
        por_costo.append({"razon_costo_fn_fp": razon, "umbral": barrido[i]["umbral"],
                          "precision": barrido[i]["precision"], "recall": barrido[i]["recall"],
                          "fn": barrido[i]["fn"], "fp": barrido[i]["fp"]})
    ingenuo = next(f for f in barrido if abs(f["umbral"] - UMBRAL_BIGMODEL) < 1e-9)
    print(f"  F1 máximo:         u={f1['umbral']:.3f}  P={f1['precision']:.3f} "
          f"R={f1['recall']:.3f} F1={f1['f1']:.3f}")
    if alto_recall:
        print(f"  recall ≥ {RECALL_MINIMO}:     u={alto_recall['umbral']:.3f}  "
              f"P={alto_recall['precision']:.3f} R={alto_recall['recall']:.3f}")
    print(f"  ingenuo (0.85):    P={ingenuo['precision']:.3f} R={ingenuo['recall']:.3f} "
          f"F1={ingenuo['f1']:.3f}")
    for c in por_costo:
        print(f"  costo FN = {c['razon_costo_fn_fp']:>2}×FP -> u={c['umbral']:.3f} "
              f"(P={c['precision']:.3f}, R={c['recall']:.3f})")
    return {"barrido": barrido, "umbral_f1": f1["umbral"],
            "umbral_alto_recall": alto_recall["umbral"] if alto_recall else None,
            "umbral_recomendado": alto_recall["umbral"] if alto_recall else f1["umbral"],
            "por_costo": por_costo, "ingenuo": ingenuo}


def g4_evaluacion(X, y, clase, est, cols, cfg, epocas, procesos) -> Dict:
    """G4: estimación de generalización del modelo elegido por k-fold.

    Particiones nuevas (otras semillas que G2). En cada fold:
      - se entrena con k-1 partes,
      - el umbral se elige con el criterio (recall ≥ 95 %, máxima precision)
        SÓLO sobre el fold de entrenamiento,
      - y se mide todo (regresión y clasificación) en el fold de validación.
    Así ni los hiperparámetros ni el umbral se eligen con los datos que se usan
    para medir. Se incluye el perceptrón lineal como referencia.
    """
    particiones = {r: k_fold(len(y), K, np.random.default_rng(900 + r), est)
                   for r in range(REPETICIONES_CV)}
    tareas = []
    for modelo in ("logistica", "lineal"):
        for r, folds in particiones.items():
            for f, (tr, va) in enumerate(folds):
                kw = (dict(Xtr=X[tr][:, cols], Xva=X[va][:, cols], eta=cfg["eta"], beta=cfg["beta"])
                      if modelo == "logistica" else
                      dict(Xtr=X[tr], Xva=X[va], eta=1e-4, beta=0.5, activacion="lineal"))
                tareas.append(((modelo, r, f), dict(ytr=y[tr], yva=y[va], epocas=epocas,
                                                    semilla=10 * r + f, **kw)))
    res = _ejecutar(tareas, procesos, "G4 evaluación k-fold")

    salida = {"k": K, "repeticiones": REPETICIONES_CV, "epocas": epocas, "config": cfg}
    for modelo in ("logistica", "lineal"):
        folds, oof = [], np.zeros((REPETICIONES_CV, len(y)))
        for r, parts in particiones.items():
            for f, (tr, va) in enumerate(parts):
                x = res[(modelo, r, f)]
                u = _umbral_recall(clase[tr], x["pred_ent"])
                oof[r, va] = x["pred_val"]
                folds.append({"repeticion": r, "fold": f,
                              "mse_entrenamiento": x["mse_entrenamiento"][-1],
                              "regresion": resumen(y[va], x["pred_val"]),
                              "umbral_fold": u,
                              "clasificacion_entrenamiento": _prf(clase[tr], x["pred_ent"], u),
                              "clasificacion": _prf(clase[va], x["pred_val"], u)})
        salida[modelo] = {"folds": folds, "oof": oof.tolist()}
        us = [f["umbral_fold"] for f in folds]
        print(f"  {modelo:10s} umbral por fold={np.mean(us):.3f} [{min(us):.3f}, {max(us):.3f}]  " +
              "  ".join(f"{k}={np.mean([f['clasificacion'][k] for f in folds]):.3f}"
                        f"±{np.std([f['clasificacion'][k] for f in folds], ddof=1):.3f}"
                        for k in ("recall", "precision", "f1")) +
              f"  (F1 entrenamiento={np.mean([f['clasificacion_entrenamiento']['f1'] for f in folds]):.3f})")
    return salida


def g4_modelo_final(X, y, cols, cfg, epocas, procesos) -> List[Dict]:
    """Modelo para entregar: entrenado con las 7500 muestras (3 semillas, para ver estabilidad)."""
    tareas = [((s,), dict(Xtr=X[:, cols], ytr=y, Xva=X[:, cols], yva=y, eta=cfg["eta"],
                          beta=cfg["beta"], epocas=epocas, semilla=s)) for s in SEMILLAS_FINAL]
    res = _ejecutar(tareas, procesos, "modelo final")
    return [{"semilla": s, "mse_entrenamiento": res[(s,)]["mse_entrenamiento"][-1],
             "w": res[(s,)]["w"].tolist(), "media": res[(s,)]["media"].tolist(),
             "desvio": res[(s,)]["desvio"].tolist()} for s in SEMILLAS_FINAL]


def guardar_modelo_final(ruta: str, m: Dict, cfg: Dict, epocas: int, columnas: Sequence[str],
                         umbral: float) -> None:
    """Pesos + normalización + umbral: todo lo necesario para inferir."""
    modelo = {"tipo": "perceptron_simple", "activacion": "logistica",
              "beta": cfg["beta"], "eta": cfg["eta"], "epocas": epocas, "semilla": m["semilla"],
              "entrenado_con": "las 7500 muestras",
              "features": list(columnas), "normalizacion": "z-score",
              "media": m["media"], "desvio": m["desvio"],
              "w": m["w"], "umbral_fraude": umbral,
              "uso": "p = 1/(1+exp(-2*beta*(w[0] + w[1:]·((x-media)/desvio)))); "
                     "fraude si p >= umbral_fraude"}
    with open(ruta, "w") as fh:
        json.dump(modelo, fh, indent=2, ensure_ascii=False)


def predecir_con_modelo(ruta: str, X: np.ndarray) -> np.ndarray:
    """Inferencia con el modelo guardado (X con las columnas de `features`, sin normalizar)."""
    with open(ruta) as fh:
        m = json.load(fh)
    Xs = (X - np.array(m["media"])) / np.array(m["desvio"])
    w = np.array(m["w"])
    return 1.0 / (1.0 + np.exp(-2 * m["beta"] * (w[0] + Xs @ w[1:])))


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    os.makedirs(DIR_SALIDA, exist_ok=True)
    t0 = time.perf_counter()

    X, y, features, df = cargar(RUTA_DATOS)
    clase = df["flagged_fraud"].to_numpy()
    assert np.array_equal(clase == 1, y >= UMBRAL_BIGMODEL), \
        "se esperaba flagged_fraud == (BigModel >= 0.85)"
    est = estratos(y, clase)
    print(f"Criterio: recall ≥ {RECALL_MINIMO:.0%} y máxima precision. "
          f"{len(y)} muestras ({clase.mean():.2%} fraude). Validación cruzada {K}-fold "
          f"estratificada × {REPETICIONES_CV} repeticiones.")

    print("\n== G1a: ¿qué k y qué partición? ==")
    estrategia = g1_estrategia_kfold(X, y, clase, est, args.procesos)
    print("\n== G1b: tamaño del conjunto de entrenamiento ==")
    tamano = g1_tamano(X, y, est, args.procesos)
    print("\n== G2: selección de hiperparámetros por k-fold ==")
    cv = g2_validacion_cruzada(X, y, clase, est, features, args.procesos)
    elegida = next(c for c in cv["configuraciones"]
                   if (c["features"], c["eta"], c["beta"]) ==
                   (cv["elegida"]["features"], cv["elegida"]["eta"], cv["elegida"]["beta"]))
    cols = [features.index(c) for c in cv["elegida"]["columnas"]]
    cfg = {"eta": elegida["eta"], "beta": elegida["beta"]}
    print("\n== G2b: épocas como hiperparámetro ==")
    epocas = g2b_epocas(X, y, clase, est, cols, cfg, args.procesos)
    cv["epocas_final"] = epocas["epocas_elegidas"]
    fila_ep = next(f for f in epocas["filas"] if f["epocas"] == cv["epocas_final"])
    print("\n== G3: umbral de detección (predicciones fuera de fold) ==")
    umbral = g3_umbral(clase, np.array(fila_ep["oof"]))
    for f in epocas["filas"]:
        if f is not fila_ep:
            del f["oof"]   # sólo se guarda la de las épocas elegidas
    print("\n== G4: estimación de generalización por k-fold ==")
    evaluacion = g4_evaluacion(X, y, clase, est, cols, cfg, cv["epocas_final"], args.procesos)
    # Con las 7500 muestras cada época tiene k/(k-1) más actualizaciones que en un fold de
    # entrenamiento: se escalan las épocas para que el modelo final quede igual de entrenado
    # que el que se evaluó (importa si la configuración elegida no llega a converger).
    epocas_modelo_final = int(round(cv["epocas_final"] * (K - 1) / K))
    final = g4_modelo_final(X, y, cols, cfg, epocas_modelo_final, args.procesos)

    ruta_modelo = os.path.join(DIR_SALIDA, "modelo_final.json")
    guardar_modelo_final(ruta_modelo, final[0], cfg, epocas_modelo_final,
                         cv["elegida"]["columnas"], umbral["umbral_recomendado"])
    p = predecir_con_modelo(ruta_modelo, X[:, cols])
    assert np.isclose(np.mean((y - p) ** 2), final[0]["mse_entrenamiento"])
    print(f"  modelo final (7500 muestras, {epocas_modelo_final} épocas) "
          f"MSE={final[0]['mse_entrenamiento']:.6f}, "
          f"guardado y verificado -> {ruta_modelo}")

    resultados = {
        "n": len(y), "fraude": float(clase.mean()),
        "features": features, "features_ruido": FEATURES_RUIDO,
        "umbral_bigmodel": UMBRAL_BIGMODEL, "recall_minimo": RECALL_MINIMO,
        "g1_estrategia": estrategia, "g1_tamano": tamano,
        "g2_validacion_cruzada": cv, "g2b_epocas": epocas, "g3_umbral": umbral,
        "g4_evaluacion": evaluacion, "modelo_final": final,
        "epocas_modelo_final": epocas_modelo_final,
        "y": y.tolist(), "clase": clase.tolist(),
        "tiempo_total_s": time.perf_counter() - t0,
    }
    ruta = os.path.join(DIR_SALIDA, "resultados.json")
    with open(ruta, "w") as fh:
        json.dump(resultados, fh, default=lambda o: o.tolist() if isinstance(o, np.ndarray)
                  else float(o))
    print(f"\nResultados en {ruta} ({resultados['tiempo_total_s'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
