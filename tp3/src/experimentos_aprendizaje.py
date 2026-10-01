"""Ejercicio 1 - ESTUDIO DE APRENDIZAJE (TP3 - SIA).

Compara el perceptrón simple LINEAL contra el NO LINEAL (activación logística)
entrenando con TODAS las muestras del conjunto de datos, tal como aclara el
enunciado, para responder:

    a) ¿observan underfitting?
    b) ¿observan saturación de las capacidades?
    c) ¿cuál seleccionarían para el estudio de generalización?

Guarda todos los resultados crudos en salidas/aprendizaje/ para que el análisis
y los gráficos se hagan por separado (graficos_aprendizaje.py).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

from .datos import (Normalizador, cargar, explorar, referencia_logistica_analitica,
                    solucion_analitica_lineal)
from .metricas import resumen
from .perceptron import PerceptronSimple

RUTA_DATOS = "tp3/datos/fraud_dataset.csv"
DIR_SALIDA = "tp3/salidas/aprendizaje"

SEMILLAS = [1, 2, 3, 4, 5]
ETAS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0]
BETAS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
EPOCAS_BARRIDO = 60
EPOCAS_LARGO = 400


def _np2list(d: Dict) -> Dict:
    return json.loads(json.dumps(d, default=lambda o: o.tolist()
                                 if isinstance(o, np.ndarray) else float(o)))


# --------------------------------------------------------------------------- #
def experimento_barrido_eta(Xs, y, estados) -> List[Dict]:
    """E1: ¿el piso de error del lineal es por mala optimización o por capacidad?

    Si el mínimo error no mejora para NINGÚN eta, el techo es estructural.
    """
    filas = []
    for activacion, beta in (("lineal", 0.5), ("logistica", 0.5)):
        for eta in ETAS:
            mses, divergio = [], 0
            for s in SEMILLAS[:3]:
                m = PerceptronSimple(Xs.shape[1], activacion, eta=eta, beta=beta,
                                     tamano_lote=1, semilla=s)
                h = m.entrenar(Xs, y, epocas=EPOCAS_BARRIDO, registrar_diagnostico=False)
                mse = h.mse[-1]
                if not np.isfinite(mse) or mse > 10:
                    divergio += 1
                    mse = float("nan")
                mses.append(mse)
            filas.append({
                "activacion": activacion, "eta": eta, "beta": beta,
                "mse_medio": float(np.nanmean(mses)) if divergio < len(mses) else None,
                "mse_desvio": float(np.nanstd(mses)) if divergio < len(mses) else None,
                "mse_mejor": float(np.nanmin(mses)) if divergio < len(mses) else None,
                "corridas_divergentes": divergio,
            })
            print(f"  [eta] {activacion:10s} eta={eta:<7g} "
                  f"MSE={filas[-1]['mse_medio']}  div={divergio}")
    return filas


def experimento_barrido_beta(Xs, y, eta_logistica) -> List[Dict]:
    """E2: efecto de beta sobre la saturación de la sigmoidea."""
    filas = []
    for beta in BETAS:
        mses, sat, der = [], [], []
        for s in SEMILLAS[:3]:
            m = PerceptronSimple(Xs.shape[1], "logistica", eta=eta_logistica, beta=beta,
                                 tamano_lote=1, semilla=s)
            h = m.entrenar(Xs, y, epocas=EPOCAS_BARRIDO)
            mses.append(h.mse[-1]); sat.append(h.fraccion_saturada[-1])
            der.append(h.derivada_media[-1])
        filas.append({
            "beta": beta,
            "mse_medio": float(np.mean(mses)), "mse_desvio": float(np.std(mses)),
            "fraccion_saturada": float(np.mean(sat)),
            "derivada_media": float(np.mean(der)),
        })
        print(f"  [beta] beta={beta:<5g} MSE={filas[-1]['mse_medio']:.6f} "
              f"saturadas={filas[-1]['fraccion_saturada']:.1%}")
    return filas


def experimento_curvas(Xs, y, config_lineal, config_logistica) -> Dict:
    """E3: curvas de aprendizaje largas (media +- desvío sobre semillas)."""
    salida = {}
    for nombre, cfg in (("lineal", config_lineal), ("logistica", config_logistica)):
        curvas, finales, modelos = [], [], []
        for s in SEMILLAS:
            m = PerceptronSimple(Xs.shape[1], nombre, eta=cfg["eta"], beta=cfg["beta"],
                                 tamano_lote=1, semilla=s)
            h = m.entrenar(Xs, y, epocas=EPOCAS_LARGO, verbose=0)
            curvas.append(h.mse)
            o = m.predecir(Xs)
            finales.append(resumen(y, o))
            modelos.append((m, h))
            print(f"  [curva] {nombre:10s} semilla={s} MSE={h.mse[-1]:.6f} "
                  f"R2={finales[-1]['r2']:.4f} ({h.tiempo_segundos:.1f}s)")
        arr = np.array(curvas)
        mejor = int(np.argmin([f["mse"] for f in finales]))
        m_mejor, h_mejor = modelos[mejor]
        salida[nombre] = {
            "config": cfg,
            "mse_medio_por_epoca": arr.mean(axis=0).tolist(),
            "mse_desvio_por_epoca": arr.std(axis=0).tolist(),
            "metricas_por_semilla": finales,
            "metricas_medias": {k: float(np.mean([f[k] for f in finales])) for k in finales[0]},
            "metricas_desvio": {k: float(np.std([f[k] for f in finales])) for k in finales[0]},
            "mejor_semilla": SEMILLAS[mejor],
            "pesos_mejor": m_mejor.w.tolist(),
            "predicciones_mejor": m_mejor.predecir(Xs).tolist(),
            "fraccion_saturada_por_epoca": h_mejor.fraccion_saturada,
            "derivada_media_por_epoca": h_mejor.derivada_media,
            "tiempo_medio_s": float(np.mean([h.tiempo_segundos for _, h in modelos])),
        }
    return salida


def experimento_capacidad(Xs, y, config_lineal, config_logistica) -> Dict:
    """E4: ¿el error deja de bajar aunque se multipliquen las épocas?

    Mide el MSE a 25/50/100/200/400/800 épocas: si la mejora marginal tiende a
    cero, el modelo agotó su capacidad (saturación de capacidades).
    """
    cortes = [1, 2, 5, 10, 25, 50, 100, 200, 400, 800]
    salida = {}
    for nombre, cfg in (("lineal", config_lineal), ("logistica", config_logistica)):
        m = PerceptronSimple(Xs.shape[1], nombre, eta=cfg["eta"], beta=cfg["beta"],
                             tamano_lote=1, semilla=SEMILLAS[0])
        h = m.entrenar(Xs, y, epocas=max(cortes))
        salida[nombre] = {
            "cortes": cortes,
            "mse_en_corte": [h.mse[c - 1] for c in cortes],
            "mejora_relativa_vs_corte_previo": [
                None if i == 0 else float((h.mse[cortes[i - 1] - 1] - h.mse[c - 1])
                                          / h.mse[cortes[i - 1] - 1])
                for i, c in enumerate(cortes)
            ],
            "curva_completa": h.mse,
        }
        print(f"  [capacidad] {nombre:10s} " +
              "  ".join(f"{c}ép:{h.mse[c-1]:.5f}" for c in cortes))
    return salida


def experimento_diagnostico(Xs, y, config_lineal, config_logistica) -> Dict:
    """E7: diagnóstico fino de underfitting y de saturación de la sigmoidea.

    - ¿Dónde se equivoca cada modelo? (error por zona del objetivo)
    - ¿Cuántas épocas necesita cada uno para agotar su mejora? (velocidad de
      saturación de la capacidad)
    - ¿Qué pasa con las 43 muestras con zeta = 1.0, inalcanzables para una
      logística que tiende asintóticamente a 1?
    """
    salida = {}
    zonas = {"bajo (ζ<0.2)": y < 0.2, "medio (0.2≤ζ≤0.8)": (y >= 0.2) & (y <= 0.8),
             "alto (ζ>0.8)": y > 0.8, "extremo (ζ>0.99)": y > 0.99}

    for nombre, cfg in (("lineal", config_lineal), ("logistica", config_logistica)):
        m = PerceptronSimple(Xs.shape[1], nombre, eta=cfg["eta"], beta=cfg["beta"],
                             tamano_lote=1, semilla=SEMILLAS[0])
        h = m.entrenar(Xs, y, epocas=EPOCAS_LARGO)
        o = m.predecir(Xs)
        r = o - y

        curva = np.array(h.mse)
        mejora_total = curva[0] - curva[-1]
        def epocas_para(frac):
            objetivo = curva[0] - frac * mejora_total
            alcanzado = np.where(curva <= objetivo)[0]
            return int(alcanzado[0] + 1) if len(alcanzado) else None

        salida[nombre] = {
            "error_por_zona": {
                z: {"n": int(mask.sum()),
                    "mse": float(np.mean(r[mask] ** 2)),
                    "sesgo_medio": float(np.mean(r[mask]))}
                for z, mask in zonas.items()
            },
            "epocas_para_90pct_mejora": epocas_para(0.90),
            "epocas_para_99pct_mejora": epocas_para(0.99),
            "epocas_para_999pct_mejora": epocas_para(0.999),
            "mejora_ultimas_100_epocas_pct": float(100 * (curva[-101] - curva[-1]) / curva[-101]),
            "salida_maxima": float(o.max()), "salida_minima": float(o.min()),
            "residuo_medio_en_zeta_igual_1": float(np.mean(r[y == 1.0])),
            "n_zeta_igual_1": int((y == 1.0).sum()),
            "fraccion_saturada_final": h.fraccion_saturada[-1] if h.fraccion_saturada else None,
            "fraccion_saturada_inicial": h.fraccion_saturada[0] if h.fraccion_saturada else None,
            "pesos": m.w.tolist(),
        }
        d = salida[nombre]
        print(f"  [diag] {nombre:10s} 99% de la mejora en {d['epocas_para_99pct_mejora']} épocas; "
              f"últimas 100 épocas mejoran {d['mejora_ultimas_100_epocas_pct']:.4f}%; "
              f"salida ∈ [{d['salida_minima']:.3f}, {d['salida_maxima']:.3f}]")
    return salida


def experimento_regimen(Xs, y, config_lineal, config_logistica) -> List[Dict]:
    """E5: online vs mini-batch vs batch, a igual presupuesto de épocas."""
    filas = []
    for nombre, cfg in (("lineal", config_lineal), ("logistica", config_logistica)):
        for etiqueta, lote in (("online", 1), ("mini-batch-32", 32), ("batch", None)):
            m = PerceptronSimple(Xs.shape[1], nombre, eta=cfg["eta"], beta=cfg["beta"],
                                 tamano_lote=lote, semilla=SEMILLAS[0])
            h = m.entrenar(Xs, y, epocas=200, registrar_diagnostico=False)
            filas.append({"activacion": nombre, "regimen": etiqueta,
                          "mse_final": h.mse[-1], "tiempo_s": h.tiempo_segundos})
            print(f"  [régimen] {nombre:10s} {etiqueta:14s} MSE={h.mse[-1]:.6f} "
                  f"({h.tiempo_segundos:.1f}s)")
    return filas


def experimento_sin_normalizar(X, y, config_lineal, config_logistica) -> List[Dict]:
    """E6: justifica el preprocesamiento. Sin estandarizar, el lineal diverge y
    la logística arranca completamente saturada."""
    filas = []
    for nombre, cfg in (("lineal", config_lineal), ("logistica", config_logistica)):
        m = PerceptronSimple(X.shape[1], nombre, eta=cfg["eta"], beta=cfg["beta"],
                             tamano_lote=1, semilla=SEMILLAS[0])
        h = m.entrenar(X, y, epocas=20)
        mse = h.mse[-1]
        filas.append({
            "activacion": nombre,
            "mse_final": None if not np.isfinite(mse) else float(mse),
            "divergio": bool(not np.isfinite(mse) or mse > 10),
            "fraccion_saturada_epoca_1": h.fraccion_saturada[0] if h.fraccion_saturada else None,
        })
        print(f"  [sin normalizar] {nombre:10s} MSE={filas[-1]['mse_final']} "
              f"divergió={filas[-1]['divergio']}")
    return filas


# --------------------------------------------------------------------------- #
def main() -> None:
    os.makedirs(DIR_SALIDA, exist_ok=True)
    print("== Cargando y explorando el conjunto de datos ==")
    X, y, features, df = cargar(RUTA_DATOS)
    exploracion = explorar(df)
    exploracion["features_usados"] = features
    exploracion["nota_flagged_fraud"] = ("Excluida del entrenamiento: la documentación "
                                         "prohíbe usarla. Se reserva para el estudio de "
                                         "generalización / elección de umbral.")

    norm = Normalizador.ajustar(X)
    Xs = norm.aplicar(X)

    # Cota inferior analítica del perceptrón lineal (óptimo global exacto).
    w_ols, pred_ols = solucion_analitica_lineal(Xs, y)
    piso_lineal = resumen(y, pred_ols)
    print(f"\nPiso analítico del LINEAL (mínimos cuadrados): MSE={piso_lineal['mse']:.6f} "
          f"R2={piso_lineal['r2']:.4f}  fuera de [0,1]={piso_lineal['fuera_de_rango']:.2%}")

    _, pred_log_ref = referencia_logistica_analitica(Xs, y)
    ref_logistica = resumen(y, pred_log_ref)
    print(f"Referencia analítica de la LOGÍSTICA (mín. cuadrados en logit): "
          f"MSE={ref_logistica['mse']:.6f} R2={ref_logistica['r2']:.4f}")

    print("\n== E1: barrido de tasa de aprendizaje ==")
    barrido_eta = experimento_barrido_eta(Xs, y, None)
    mejor = {}
    for act in ("lineal", "logistica"):
        cand = [f for f in barrido_eta if f["activacion"] == act and f["mse_medio"] is not None]
        mejor[act] = min(cand, key=lambda f: f["mse_medio"])["eta"]
    cfg_lineal = {"eta": mejor["lineal"], "beta": 0.5}
    cfg_log = {"eta": mejor["logistica"], "beta": 0.5}
    print(f"  -> eta elegido: lineal={cfg_lineal['eta']}  logistica={cfg_log['eta']}")

    print("\n== E2: barrido de beta (logística) ==")
    barrido_beta = experimento_barrido_beta(Xs, y, cfg_log["eta"])
    cfg_log["beta"] = min(barrido_beta, key=lambda f: f["mse_medio"])["beta"]
    print(f"  -> beta elegido: {cfg_log['beta']}")

    print("\n== E3: curvas de aprendizaje ==")
    curvas = experimento_curvas(Xs, y, cfg_lineal, cfg_log)

    print("\n== E4: saturación de capacidades ==")
    capacidad = experimento_capacidad(Xs, y, cfg_lineal, cfg_log)

    print("\n== E7: diagnóstico fino ==")
    diagnostico = experimento_diagnostico(Xs, y, cfg_lineal, cfg_log)

    print("\n== E5: régimen de entrenamiento ==")
    regimen = experimento_regimen(Xs, y, cfg_lineal, cfg_log)

    print("\n== E6: sin normalizar (justificación del preprocesamiento) ==")
    sin_norm = experimento_sin_normalizar(X, y, cfg_lineal, cfg_log)

    resultados = {
        "exploracion": exploracion,
        "configuracion": {
            "semillas": SEMILLAS, "epocas_barrido": EPOCAS_BARRIDO,
            "epocas_largo": EPOCAS_LARGO, "normalizacion": "z-score sobre los 9 features",
            "regimen": "online (tamano_lote=1)",
            "config_lineal": cfg_lineal, "config_logistica": cfg_log,
        },
        "piso_analitico_lineal": piso_lineal,
        "referencia_analitica_logistica": ref_logistica,
        "pesos_ols": w_ols.tolist(),
        "predicciones_ols": pred_ols.tolist(),
        "barrido_eta": barrido_eta,
        "barrido_beta": barrido_beta,
        "curvas": curvas,
        "capacidad": capacidad,
        "diagnostico": diagnostico,
        "regimen": regimen,
        "sin_normalizar": sin_norm,
        "y_real": y.tolist(),
        "flagged_fraud": df["flagged_fraud"].tolist(),
    }

    ruta = os.path.join(DIR_SALIDA, "resultados.json")
    with open(ruta, "w") as fh:
        json.dump(_np2list(resultados), fh)
    print(f"\nResultados guardados en {ruta}")


if __name__ == "__main__":
    main()
