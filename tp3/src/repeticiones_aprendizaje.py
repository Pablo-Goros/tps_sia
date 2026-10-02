"""Repite el estudio de aprendizaje completo (E1-E7) varias veces (TP3 - SIA).

Cada repetición corre TODO el pipeline de experimentos_aprendizaje.py con un
conjunto de semillas disjunto (repetición k usa las semillas 5k+1 ... 5k+5), así
que cambian la inicialización de los pesos y el orden de presentación de las
muestras en cada época. Con eso se mide si los resultados (η y β elegidos, MSE,
R², épocas hasta saturar, etc.) son estables o dependen de la corrida.

Las repeticiones son independientes y se corren en paralelo (un proceso por
repetición). Este script SÓLO ejecuta y persiste; el análisis está en
analisis_repeticiones.py.

    python3 -m tp3.src.repeticiones_aprendizaje                  # 3 repeticiones
    python3 -m tp3.src.repeticiones_aprendizaje --repeticiones 5 --procesos 2
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

from .experimentos_aprendizaje import DIR_SALIDA, correr

DIR_REPETICIONES = os.path.join(DIR_SALIDA, "repeticiones")
SEMILLAS_POR_REPETICION = 5


def semillas_de(repeticion: int) -> List[int]:
    """Repetición 1 -> [1..5] (la corrida original), 2 -> [6..10], 3 -> [11..15]."""
    base = (repeticion - 1) * SEMILLAS_POR_REPETICION
    return [base + i for i in range(1, SEMILLAS_POR_REPETICION + 1)]


def _correr_repeticion(repeticion: int) -> dict:
    """Corre una repetición completa; la salida de consola va a corrida_k.log."""
    semillas = semillas_de(repeticion)
    ruta_log = os.path.join(DIR_REPETICIONES, f"corrida_{repeticion}.log")
    t0 = time.perf_counter()
    with open(ruta_log, "w", buffering=1) as log, contextlib.redirect_stdout(log):
        print(f"Repetición {repeticion} - semillas {semillas}\n")
        resultados = correr(semillas)
    resultados["repeticion"] = repeticion
    resultados["tiempo_total_s"] = time.perf_counter() - t0

    ruta = os.path.join(DIR_REPETICIONES, f"corrida_{repeticion}.json")
    with open(ruta, "w") as fh:
        json.dump(resultados, fh)
    return {"repeticion": repeticion, "semillas": semillas, "ruta": ruta,
            "tiempo_s": resultados["tiempo_total_s"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeticiones", type=int, default=3)
    parser.add_argument("--procesos", type=int, default=None,
                        help="procesos en paralelo (por defecto, uno por repetición)")
    args = parser.parse_args()

    os.makedirs(DIR_REPETICIONES, exist_ok=True)
    repeticiones = list(range(1, args.repeticiones + 1))
    procesos = args.procesos or len(repeticiones)
    print(f"Corriendo {len(repeticiones)} repeticiones del estudio de aprendizaje "
          f"en {procesos} procesos (≈ 11 min cada una).")
    for r in repeticiones:
        print(f"  repetición {r}: semillas {semillas_de(r)}  "
              f"(progreso en {DIR_REPETICIONES}/corrida_{r}.log)")

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=procesos) as pool:
        futuros = [pool.submit(_correr_repeticion, r) for r in repeticiones]
        for f in as_completed(futuros):
            info = f.result()
            print(f"  ok  repetición {info['repeticion']} terminada en "
                  f"{info['tiempo_s'] / 60:.1f} min -> {info['ruta']}")

    print(f"\nListo en {(time.perf_counter() - t0) / 60:.1f} min. Para el análisis:\n"
          f"  python3 -m tp3.src.analisis_repeticiones")


if __name__ == "__main__":
    main()
