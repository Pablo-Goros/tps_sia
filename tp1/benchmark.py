#!/usr/bin/env python3
"""Corridas comparativas para el informe: todos los niveles x todos los metodos.

Ejemplos:

    python3 benchmark.py
    python3 benchmark.py --niveles niveles/nivel_0[1-4]*.txt --repeticiones 5
    python3 benchmark.py --algoritmos astar --heuristicas manhattan matching
    python3 benchmark.py --graficos

Genera un CSV, una tabla en Markdown y (opcionalmente) graficos PNG en salidas/.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from sokoban import (  # noqa: E402
    ALGORITMOS,
    HEURISTICAS,
    HEURISTICA_POR_DEFECTO,
    ORDEN_ALGORITMOS,
    NivelInvalido,
    cargar_nivel,
    ejecutar_busqueda,
)

COLUMNAS = [
    "nivel",
    "cajas",
    "algoritmo",
    "heuristica",
    "exito",
    "motivo",
    "costo",
    "nodos_expandidos",
    "nodos_frontera",
    "nodos_generados",
    "tiempo_seg",
    "tiempo_desvio",
    "repeticiones",
]


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--niveles",
        nargs="+",
        default=["niveles/*.txt"],
        help="rutas o patrones glob de niveles (default: niveles/*.txt)",
    )
    parser.add_argument(
        "--algoritmos",
        nargs="+",
        default=list(ORDEN_ALGORITMOS),
        choices=list(ORDEN_ALGORITMOS),
    )
    parser.add_argument(
        "--heuristicas",
        nargs="+",
        default=["manhattan", "matching"],
        choices=sorted(HEURISTICAS),
        help="se usan solo con greedy y astar",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-nodos", type=int)
    parser.add_argument(
        "--repeticiones",
        type=int,
        default=1,
        help="corridas por configuracion; se informa media y desvio del tiempo",
    )
    parser.add_argument("--sin-poda", action="store_true")
    parser.add_argument("--csv", default="salidas/benchmark.csv")
    parser.add_argument("--markdown", default="salidas/benchmark.md")
    parser.add_argument(
        "--graficos", action="store_true", help="genera PNGs con matplotlib"
    )
    return parser


def expandir_niveles(patrones: List[str]) -> List[str]:
    rutas: List[str] = []
    for patron in patrones:
        encontrados = sorted(glob.glob(patron))
        rutas.extend(encontrados if encontrados else [patron])
    return rutas


def configuraciones(algoritmos: List[str], heuristicas: List[str]):
    for clave in algoritmos:
        if ALGORITMOS[clave].usa_heuristica:
            for heuristica in heuristicas:
                yield clave, heuristica
        else:
            yield clave, None


def correr(args: argparse.Namespace) -> List[Dict]:
    filas: List[Dict] = []
    for ruta in expandir_niveles(args.niveles):
        try:
            mapa, estado_inicial = cargar_nivel(ruta)
        except (OSError, NivelInvalido) as error:
            print("  [!] {}: {}".format(ruta, error))
            continue

        print("\n{} ({} cajas)".format(ruta, len(estado_inicial.cajas)))
        for clave, heuristica in configuraciones(args.algoritmos, args.heuristicas):
            tiempos = []
            resultado = None
            for _ in range(max(1, args.repeticiones)):
                resultado = ejecutar_busqueda(
                    clave,
                    estado_inicial,
                    mapa,
                    heuristica=heuristica or HEURISTICA_POR_DEFECTO,
                    podar_deadlocks=not args.sin_poda,
                    max_nodos=args.max_nodos,
                    timeout=args.timeout,
                    guardar_estados=False,
                )
                tiempos.append(resultado.tiempo_seg)

            fila = {
                "nivel": os.path.basename(ruta),
                "cajas": len(estado_inicial.cajas),
                "algoritmo": resultado.algoritmo,
                "heuristica": heuristica or "-",
                "exito": resultado.exito,
                "motivo": resultado.motivo,
                "costo": resultado.costo,
                "nodos_expandidos": resultado.nodos_expandidos,
                "nodos_frontera": resultado.nodos_frontera,
                "nodos_generados": resultado.nodos_generados,
                "tiempo_seg": round(statistics.mean(tiempos), 6),
                "tiempo_desvio": round(
                    statistics.stdev(tiempos) if len(tiempos) > 1 else 0.0, 6
                ),
                "repeticiones": len(tiempos),
            }
            filas.append(fila)
            print(
                "  {:<8} {:<18} {:<12} costo={:<7} expandidos={:<10} t={:.4f}s".format(
                    fila["algoritmo"],
                    fila["heuristica"],
                    "exito" if fila["exito"] else fila["motivo"],
                    str(fila["costo"]),
                    fila["nodos_expandidos"],
                    fila["tiempo_seg"],
                )
            )
    return filas


def guardar_csv(ruta: str, filas: List[Dict]) -> None:
    _asegurar_directorio(ruta)
    with open(ruta, "w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS)
        escritor.writeheader()
        escritor.writerows(filas)
    print("\nCSV: {}".format(ruta))


def guardar_markdown(ruta: str, filas: List[Dict]) -> None:
    _asegurar_directorio(ruta)
    lineas = ["# Resultados comparativos\n"]
    niveles = sorted({fila["nivel"] for fila in filas})
    for nivel in niveles:
        del_nivel = [f for f in filas if f["nivel"] == nivel]
        lineas.append("\n## {} ({} cajas)\n".format(nivel, del_nivel[0]["cajas"]))
        lineas.append(
            "| Algoritmo | Heuristica | Resultado | Costo | Expandidos | "
            "Frontera | Generados | Tiempo (s) |"
        )
        lineas.append("|---|---|---|---|---|---|---|---|")
        for f in del_nivel:
            lineas.append(
                "| {} | {} | {} | {} | {} | {} | {} | {:.4f} |".format(
                    f["algoritmo"],
                    f["heuristica"],
                    "exito" if f["exito"] else f["motivo"],
                    f["costo"] if f["costo"] is not None else "-",
                    f["nodos_expandidos"],
                    f["nodos_frontera"],
                    f["nodos_generados"],
                    f["tiempo_seg"],
                )
            )
    with open(ruta, "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(lineas) + "\n")
    print("Markdown: {}".format(ruta))


def generar_graficos(filas: List[Dict], carpeta: str = "salidas/graficos") -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib no esta instalado: se omiten los graficos")
        return

    os.makedirs(carpeta, exist_ok=True)
    for nivel in sorted({fila["nivel"] for fila in filas}):
        del_nivel = [f for f in filas if f["nivel"] == nivel]
        etiquetas = [
            f["algoritmo"] if f["heuristica"] == "-" else "{}\n{}".format(
                f["algoritmo"], f["heuristica"]
            )
            for f in del_nivel
        ]
        expandidos = [f["nodos_expandidos"] for f in del_nivel]
        tiempos = [f["tiempo_seg"] for f in del_nivel]
        colores = ["#3f79d6" if f["exito"] else "#c0562f" for f in del_nivel]

        figura, ejes = plt.subplots(1, 2, figsize=(13, 4.5))
        for eje, valores, titulo in (
            (ejes[0], expandidos, "Nodos expandidos"),
            (ejes[1], tiempos, "Tiempo (s)"),
        ):
            eje.bar(range(len(valores)), valores, color=colores)
            eje.set_xticks(range(len(etiquetas)))
            eje.set_xticklabels(etiquetas, fontsize=7, rotation=30, ha="right")
            eje.set_yscale("log")
            eje.set_title(titulo)
            eje.grid(axis="y", alpha=0.3)
        figura.suptitle("{} (rojo = sin solucion / corte)".format(nivel))
        figura.tight_layout()
        destino = os.path.join(carpeta, nivel.replace(".txt", "") + ".png")
        figura.savefig(destino, dpi=130)
        plt.close(figura)
        print("Grafico: {}".format(destino))


def _asegurar_directorio(ruta: str) -> None:
    carpeta = os.path.dirname(os.path.abspath(ruta))
    if carpeta and not os.path.isdir(carpeta):
        os.makedirs(carpeta, exist_ok=True)


def main(argv: Optional[List[str]] = None) -> int:
    args = construir_parser().parse_args(argv)
    filas = correr(args)
    if not filas:
        print("no se ejecuto ninguna configuracion")
        return 1
    guardar_csv(args.csv, filas)
    guardar_markdown(args.markdown, filas)
    if args.graficos:
        generar_graficos(filas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
