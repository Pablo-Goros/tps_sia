"""Análisis de las repeticiones del estudio de aprendizaje (TP3 - SIA).

Lee salidas/aprendizaje/repeticiones/corrida_*.json (producidos por
repeticiones_aprendizaje.py) y responde: ¿los resultados y las conclusiones de
(a), (b) y (c) son estables entre corridas, o dependen de la semilla?

Genera:
    repeticiones/resumen_repeticiones.json   media, desvío, mín, máx y CV de cada métrica
    repeticiones/tabla_repeticiones.md       tablas para el informe / la presentación
    09_repeticiones_curvas.png               curvas de aprendizaje de cada repetición
    10_repeticiones_barridos.png             barridos de η y β de cada repetición

    python -m tps_sia.tp3.ej1.src.analisis_repeticiones
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .graficos_aprendizaje import C_LINEAL, C_LOGIS, C_REF, TINTA, _guardar

DIR = Path(__file__).resolve().parents[1] / "salidas" / "aprendizaje"
DIR_REP = os.path.join(DIR, "repeticiones")
MODELOS = (("lineal", "Lineal"), ("logistica", "Logística"))


def cargar_corridas() -> List[Dict]:
    rutas = sorted(glob.glob(os.path.join(DIR_REP, "corrida_*.json")),
                   key=lambda r: int(r.rsplit("_", 1)[1].split(".")[0]))
    if not rutas:
        raise SystemExit(f"No hay corridas en {DIR_REP}. Correr antes:\n"
                         "  python -m tps_sia.tp3.ej1.src.repeticiones_aprendizaje")
    corridas = []
    for ruta in rutas:
        with open(ruta) as fh:
            corridas.append(json.load(fh))
    return corridas


# --------------------------------------------------------------------------- #
def metricas_de_corrida(R: Dict) -> Dict[str, float]:
    """Aplana una corrida a {nombre_métrica: valor escalar}."""
    m: Dict[str, float] = {
        "eta_lineal": R["configuracion"]["config_lineal"]["eta"],
        "eta_logistica": R["configuracion"]["config_logistica"]["eta"],
        "beta_logistica": R["configuracion"]["config_logistica"]["beta"],
    }
    piso = R["piso_analitico_lineal"]["mse"]
    for nombre, _ in MODELOS:
        med = R["curvas"][nombre]["metricas_medias"]
        for k in ("mse", "rmse", "mae", "r2", "error_max", "fuera_de_rango"):
            m[f"{nombre}.{k}"] = med[k]
        m[f"{nombre}.mse_desvio_entre_semillas"] = R["curvas"][nombre]["metricas_desvio"]["mse"]
        m[f"{nombre}.tiempo_s"] = R["curvas"][nombre]["tiempo_medio_s"]

        d = R["diagnostico"][nombre]
        m[f"{nombre}.epocas_99pct_mejora"] = d["epocas_para_99pct_mejora"]
        m[f"{nombre}.mejora_ultimas_100_epocas_pct"] = d["mejora_ultimas_100_epocas_pct"]
        for zona, v in d["error_por_zona"].items():
            m[f"{nombre}.mse_zona[{zona}]"] = v["mse"]
        if d["fraccion_saturada_final"] is not None and nombre == "logistica":
            m[f"{nombre}.saturadas_inicial"] = d["fraccion_saturada_inicial"]
            m[f"{nombre}.saturadas_final"] = d["fraccion_saturada_final"]

        cap = R["capacidad"][nombre]
        for c, v in zip(cap["cortes"], cap["mse_en_corte"]):
            if c in (1, 10, 25, 800):
                m[f"{nombre}.mse_a_{c}_epocas"] = v

        for f in R["regimen"]:
            if f["activacion"] == nombre:
                m[f"{nombre}.regimen[{f['regimen']}]"] = f["mse_final"]

    m["lineal.brecha_al_optimo_analitico"] = m["lineal.mse"] - piso
    m["mejora_logistica_vs_lineal_pct"] = 100 * (1 - m["logistica.mse"] / m["lineal.mse"])
    return m


def agregar(valores: List[float]) -> Dict[str, float]:
    v = np.array([np.nan if x is None else x for x in valores], dtype=float)
    media = float(np.nanmean(v))
    # Desvío muestral (ddof=1): con 3 repeticiones el poblacional lo subestima.
    desvio = float(np.nanstd(v, ddof=1)) if np.sum(np.isfinite(v)) > 1 else 0.0
    return {
        "valores": [None if not np.isfinite(x) else float(x) for x in v],
        "media": media, "desvio": desvio,
        "min": float(np.nanmin(v)), "max": float(np.nanmax(v)),
        "cv_pct": float(100 * desvio / abs(media)) if media else 0.0,
    }


def metricas_todas_las_semillas(corridas: List[Dict]) -> Dict:
    """E3 agrupando las semillas de todas las repeticiones (3 x 5 = 15 entrenamientos)."""
    salida = {}
    for nombre, _ in MODELOS:
        por_semilla = [f for R in corridas for f in R["curvas"][nombre]["metricas_por_semilla"]]
        salida[nombre] = {
            k: {"media": float(np.mean([f[k] for f in por_semilla])),
                "desvio": float(np.std([f[k] for f in por_semilla], ddof=1)),
                "n": len(por_semilla)}
            for k in ("mse", "rmse", "mae", "r2", "error_max", "fuera_de_rango")
        }
    return salida


def chequear_conclusiones(corridas: List[Dict], por_corrida: List[Dict]) -> Dict[str, bool]:
    """¿Las respuestas a (a), (b) y (c) se sostienen en TODAS las repeticiones?"""
    def todas(cond):
        return bool(all(cond(m) for m in por_corrida))
    return {
        "(a) el lineal llega a su óptimo analítico (brecha < 0.1 %)": all(
            m["lineal.brecha_al_optimo_analitico"] < 1e-3 * R["piso_analitico_lineal"]["mse"]
            for m, R in zip(por_corrida, corridas)),
        "(a) el lineal saca salidas fuera de [0,1]": todas(lambda m: m["lineal.fuera_de_rango"] > 0),
        "(b) el lineal agota el 99 % de su mejora en ≤ 25 épocas":
            todas(lambda m: m["lineal.epocas_99pct_mejora"] <= 25),
        "(b) la logística agota el 99 % de su mejora en ≤ 25 épocas":
            todas(lambda m: m["logistica.epocas_99pct_mejora"] <= 25),
        "(b) la fracción saturada de la logística crece durante el entrenamiento":
            todas(lambda m: m["logistica.saturadas_final"] > m["logistica.saturadas_inicial"]),
        "(c) la logística tiene menor MSE que el lineal":
            todas(lambda m: m["logistica.mse"] < m["lineal.mse"]),
        "(c) la logística no saca salidas fuera de [0,1]":
            todas(lambda m: m["logistica.fuera_de_rango"] == 0),
        "se elige el mismo η y β en todas las repeticiones": len({
            (m["eta_lineal"], m["eta_logistica"], m["beta_logistica"]) for m in por_corrida}) == 1,
    }


# --------------------------------------------------------------------------- #
def g_curvas(corridas: List[Dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    estilos = ["-", "--", ":", "-.", (0, (5, 1))]
    piso = corridas[0]["piso_analitico_lineal"]["mse"]
    for i, R in enumerate(corridas):
        for nombre, color, etiqueta in (("lineal", C_LINEAL, "Lineal"),
                                        ("logistica", C_LOGIS, "Logística")):
            m = np.array(R["curvas"][nombre]["mse_medio_por_epoca"])
            axes[0].plot(np.arange(1, len(m) + 1), m, color=color, lw=1.8,
                         ls=estilos[i % len(estilos)],
                         label=f"{etiqueta} - rep. {R['repeticion']}")
    axes[0].axhline(piso, color=C_REF, ls="--", lw=1.2, label="Óptimo analítico del lineal")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("Época (escala log)"); axes[0].set_ylabel("MSE (media de 5 semillas)")
    axes[0].set_title("Las curvas de cada repetición se superponen", loc="left", fontsize=11)
    axes[0].legend(fontsize=8, ncol=2)

    x = np.arange(len(corridas))
    ancho = 0.38
    for j, (nombre, color, etiqueta) in enumerate((("lineal", C_LINEAL, "Lineal"),
                                                   ("logistica", C_LOGIS, "Logística"))):
        med = [R["curvas"][nombre]["metricas_medias"]["mse"] for R in corridas]
        des = [R["curvas"][nombre]["metricas_desvio"]["mse"] for R in corridas]
        axes[1].bar(x + (j - 0.5) * ancho, med, ancho, yerr=des, capsize=4,
                    color=color, label=etiqueta)
        for xi, v in zip(x + (j - 0.5) * ancho, med):
            axes[1].text(xi, v, f"{v:.5f}", ha="center", va="bottom", fontsize=8, color=TINTA)
    axes[1].set_xticks(x, [f"Rep. {R['repeticion']}" for R in corridas])
    axes[1].set_ylabel("MSE final (± desvío entre semillas)")
    axes[1].set_title("MSE final por repetición", loc="left", fontsize=11)
    axes[1].set_ylim(0, 1.25 * max(R["curvas"]["lineal"]["metricas_medias"]["mse"]
                                   for R in corridas))
    axes[1].legend(fontsize=9, loc="upper right", ncol=2)
    axes[1].grid(axis="x", visible=False)
    fig.suptitle(f"Estabilidad del estudio de aprendizaje entre {len(corridas)} repeticiones",
                 x=0.012, ha="left", fontsize=12.5, color=TINTA, y=1.04)
    _guardar(fig, "09_repeticiones_curvas.png")


def g_barridos(corridas: List[Dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    marcadores = ["o", "s", "^", "D", "v"]
    for i, R in enumerate(corridas):
        mk = marcadores[i % len(marcadores)]
        for nombre, color, etiqueta in (("lineal", C_LINEAL, "Lineal"),
                                        ("logistica", C_LOGIS, "Logística")):
            fs = [f for f in R["barrido_eta"]
                  if f["activacion"] == nombre and f["mse_medio"] is not None]
            axes[0].plot([f["eta"] for f in fs], [f["mse_medio"] for f in fs],
                         marker=mk, color=color, lw=1.2, ms=6, alpha=0.8,
                         label=f"{etiqueta} - rep. {R['repeticion']}")
        fb = R["barrido_beta"]
        axes[1].plot([f["beta"] for f in fb], [f["mse_medio"] for f in fb], marker=mk,
                     color=C_LOGIS, lw=1.2, ms=6, alpha=0.8, label=f"rep. {R['repeticion']}")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("Tasa de aprendizaje η"); axes[0].set_ylabel("MSE tras 60 épocas")
    axes[0].set_title("Barrido de η", loc="left", fontsize=11)
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("β de la función logística"); axes[1].set_ylabel("MSE tras 60 épocas")
    axes[1].set_title("Barrido de β (logística)", loc="left", fontsize=11)
    axes[1].legend(fontsize=8)
    fig.suptitle("Los barridos de hiperparámetros se repiten entre corridas",
                 x=0.012, ha="left", fontsize=12.5, color=TINTA, y=1.04)
    _guardar(fig, "10_repeticiones_barridos.png")


# --------------------------------------------------------------------------- #
def _fmt(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e4):
        return f"{x:.3e}"
    return f"{x:.6g}"


def tabla(resumen: Dict, todas: Dict, conclusiones: Dict, n: int) -> str:
    txt = [f"## Repeticiones del estudio de aprendizaje ({n} corridas)", "",
           "Cada repetición corre E1-E7 completos con semillas distintas "
           "(rep. k usa 5k-4 … 5k). El desvío es el muestral (ddof=1) entre repeticiones; "
           "CV = desvío / media.", "",
           "| Métrica | " + " | ".join(f"Rep. {i}" for i in range(1, n + 1)) +
           " | Media | Desvío | CV |",
           "|---|" + "---|" * (n + 3)]
    for k, a in resumen.items():
        txt.append(f"| `{k}` | " + " | ".join(_fmt(v) for v in a["valores"]) +
                   f" | {_fmt(a['media'])} | {_fmt(a['desvio'])} | {a['cv_pct']:.2f} % |")

    txt += ["", f"### E3 agrupando las {todas['lineal']['mse']['n']} semillas", "",
            "| Modelo | MSE | RMSE | MAE | R² | Error máx. | Fuera de [0,1] |",
            "|---|---|---|---|---|---|---|"]
    for nombre, etiqueta in MODELOS:
        t = todas[nombre]
        txt.append(f"| {etiqueta} | " + " | ".join(
            f"{_fmt(t[k]['media'])} ± {_fmt(t[k]['desvio'])}"
            for k in ("mse", "rmse", "mae", "r2", "error_max", "fuera_de_rango")) + " |")

    txt += ["", "### ¿Las conclusiones se sostienen en todas las repeticiones?", "",
            "| Afirmación | ¿Se cumple? |", "|---|---|"]
    txt += [f"| {k} | {'sí' if v else '**NO**'} |" for k, v in conclusiones.items()]
    return "\n".join(txt) + "\n"


def main() -> None:
    corridas = cargar_corridas()
    print(f"Analizando {len(corridas)} repeticiones...")
    por_corrida = [metricas_de_corrida(R) for R in corridas]
    claves = list(por_corrida[0].keys())
    resumen = {k: agregar([m.get(k) for m in por_corrida]) for k in claves}
    todas = metricas_todas_las_semillas(corridas)
    conclusiones = chequear_conclusiones(corridas, por_corrida)

    with open(os.path.join(DIR_REP, "resumen_repeticiones.json"), "w") as fh:
        json.dump({"repeticiones": [R["repeticion"] for R in corridas],
                   "semillas": [R["configuracion"]["semillas"] for R in corridas],
                   "metricas": resumen, "e3_todas_las_semillas": todas,
                   "conclusiones": conclusiones}, fh, indent=2, ensure_ascii=False)

    md = tabla(resumen, todas, conclusiones, len(corridas))
    with open(os.path.join(DIR_REP, "tabla_repeticiones.md"), "w") as fh:
        fh.write(md)
    print(md)

    g_curvas(corridas)
    g_barridos(corridas)


if __name__ == "__main__":
    main()
