#!/usr/bin/env python3
"""Motor de busqueda para Sokoban (SIA - TP1, ejercicio 2).

Ejemplos:

    python3 main.py --nivel niveles/nivel_03_medio.txt
    python3 main.py --nivel niveles/nivel_04_dificil.txt --algoritmo todos
    python3 main.py --nivel niveles/nivel_02_facil.txt --pasos
    python3 main.py --listar
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from sokoban import (  # noqa: E402
    ALGORITMOS,
    HEURISTICAS,
    HEURISTICA_POR_DEFECTO,
    ORDEN_ALGORITMOS,
    NivelInvalido,
    Resultado,
    cargar_nivel,
    ejecutar_busqueda,
)
from sokoban.visualizacion import (  # noqa: E402
    estados_desde_camino,
    generar_gif,
    render_texto,
    reproducir_en_consola,
)

MOTIVOS = {
    "exito": "EXITO",
    "sin_solucion": "FRACASO (se agoto el espacio de estados)",
    "limite_nodos": "FRACASO (se alcanzo el limite de nodos)",
    "timeout": "FRACASO (se agoto el tiempo)",
}


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Motor de busqueda para Sokoban (BFS, DFS, IDDFS, Greedy, A*).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--nivel", help="ruta al archivo de nivel (formato XSB)")
    parser.add_argument(
        "--algoritmo",
        default="astar",
        choices=list(ORDEN_ALGORITMOS) + ["todos"],
        help="metodo de busqueda a ejecutar (default: astar)",
    )
    parser.add_argument(
        "--heuristica",
        default=HEURISTICA_POR_DEFECTO,
        choices=sorted(HEURISTICAS),
        help="heuristica para greedy y astar (default: {})".format(
            HEURISTICA_POR_DEFECTO
        ),
    )
    parser.add_argument(
        "--sin-poda",
        action="store_true",
        help="desactiva la deteccion de deadlocks (util para medir su impacto)",
    )
    parser.add_argument("--max-nodos", type=int, help="corta tras expandir N nodos")
    parser.add_argument("--timeout", type=float, help="corta tras T segundos")
    parser.add_argument(
        "--pasos", action="store_true", help="imprime el tablero paso a paso"
    )
    parser.add_argument(
        "--pausa", type=float, default=0.0, help="segundos entre pasos con --pasos"
    )
    parser.add_argument("--gif", help="guarda la solucion como GIF animado")
    parser.add_argument(
        "--ms-por-paso", type=int, default=300, help="duracion de cada frame del GIF"
    )
    parser.add_argument(
        "--escala", type=int, default=44, help="lado de cada celda del GIF en pixeles"
    )
    parser.add_argument(
        "--animar", action="store_true", help="reproduce la solucion con pygame"
    )
    parser.add_argument("--json", dest="salida_json", help="guarda el resultado en JSON")
    parser.add_argument("--csv", dest="salida_csv", help="guarda el resultado en CSV")
    parser.add_argument(
        "--listar",
        action="store_true",
        help="lista algoritmos y heuristicas disponibles y termina",
    )
    return parser


def listar() -> None:
    print("Algoritmos:")
    for clave in ORDEN_ALGORITMOS:
        descripcion = ALGORITMOS[clave]
        print(
            "  {:<8} {:<8} {:<20} {}".format(
                clave,
                descripcion.nombre,
                "usa heuristica" if descripcion.usa_heuristica else "desinformado",
                "optimo" if descripcion.optimo else "no garantiza optimalidad",
            )
        )
    print("\nHeuristicas:")
    for nombre in sorted(HEURISTICAS):
        h = HEURISTICAS[nombre]
        print(
            "  {:<20} {:<15} {}".format(
                nombre, "admisible" if h.admisible else "NO admisible", h.descripcion
            )
        )


def imprimir_resultado(resultado: Resultado, camino_completo: bool) -> None:
    print("Algoritmo:              {}".format(resultado.algoritmo), end="")
    if resultado.heuristica:
        print("  (heuristica: {})".format(resultado.heuristica))
    else:
        print()
    print("Resultado:              {}".format(MOTIVOS[resultado.motivo]))
    print(
        "Costo de la solucion:   {}".format(
            "{} movimientos".format(resultado.costo) if resultado.exito else "-"
        )
    )
    print("Nodos expandidos:       {}".format(resultado.nodos_expandidos))
    print("Nodos frontera:         {}".format(resultado.nodos_frontera))
    print("Nodos generados:        {}".format(resultado.nodos_generados))
    print("Tiempo de procesamiento: {:.4f} s".format(resultado.tiempo_seg))
    if resultado.exito:
        print("Solucion (U/D/L/R):     {}".format(resultado.camino_corto))
        if camino_completo:
            print("Solucion:               {}".format(", ".join(resultado.camino)))


def imprimir_tabla(resultados: List[Resultado]) -> None:
    encabezado = "{:<8} {:<18} {:<10} {:>7} {:>12} {:>10} {:>11}".format(
        "Algoritmo",
        "Heuristica",
        "Resultado",
        "Costo",
        "Expandidos",
        "Frontera",
        "Tiempo (s)",
    )
    print(encabezado)
    print("-" * len(encabezado))
    for r in resultados:
        print(
            "{:<8} {:<18} {:<10} {:>7} {:>12} {:>10} {:>11.4f}".format(
                r.algoritmo,
                r.heuristica or "-",
                "exito" if r.exito else r.motivo,
                r.costo if r.costo is not None else "-",
                r.nodos_expandidos,
                r.nodos_frontera,
                r.tiempo_seg,
            )
        )


def guardar_json(ruta: str, datos) -> None:
    _asegurar_directorio(ruta)
    with open(ruta, "w", encoding="utf-8") as archivo:
        json.dump(datos, archivo, indent=2, ensure_ascii=False)
    print("\nResultado guardado en {}".format(ruta))


def guardar_csv(ruta: str, filas: List[Dict]) -> None:
    _asegurar_directorio(ruta)
    columnas = [
        "algoritmo",
        "heuristica",
        "exito",
        "motivo",
        "costo",
        "nodos_expandidos",
        "nodos_frontera",
        "nodos_generados",
        "tiempo_seg",
        "camino_corto",
    ]
    with open(ruta, "w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas, extrasaction="ignore")
        escritor.writeheader()
        for fila in filas:
            escritor.writerow(fila)
    print("Resultado guardado en {}".format(ruta))


def _asegurar_directorio(ruta: str) -> None:
    carpeta = os.path.dirname(os.path.abspath(ruta))
    if carpeta and not os.path.isdir(carpeta):
        os.makedirs(carpeta, exist_ok=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = construir_parser()
    args = parser.parse_args(argv)

    if args.listar:
        listar()
        return 0
    if not args.nivel:
        parser.error("hace falta --nivel (o --listar)")

    try:
        mapa, estado_inicial = cargar_nivel(args.nivel)
    except (OSError, NivelInvalido) as error:
        print("Error al cargar el nivel: {}".format(error), file=sys.stderr)
        return 2

    print("Nivel: {}".format(args.nivel))
    print(render_texto(estado_inicial, mapa))
    print(
        "\nCajas: {} | Objetivos: {} | Poda de deadlocks: {}\n".format(
            len(estado_inicial.cajas),
            len(mapa.objetivos),
            "no" if args.sin_poda else "si",
        )
    )

    claves = list(ORDEN_ALGORITMOS) if args.algoritmo == "todos" else [args.algoritmo]
    resultados: List[Resultado] = []
    for clave in claves:
        resultado = ejecutar_busqueda(
            clave,
            estado_inicial,
            mapa,
            heuristica=args.heuristica,
            podar_deadlocks=not args.sin_poda,
            max_nodos=args.max_nodos,
            timeout=args.timeout,
        )
        resultados.append(resultado)

    if len(resultados) == 1:
        imprimir_resultado(resultados[0], camino_completo=True)
    else:
        imprimir_tabla(resultados)

    # Se visualiza la mejor solucion encontrada, no la del primer algoritmo.
    exitosos = [r for r in resultados if r.exito]
    principal = min(exitosos, key=lambda r: r.costo) if exitosos else resultados[0]

    if principal.exito and (args.pasos or args.gif or args.animar):
        estados = estados_desde_camino(estado_inicial, mapa, principal.camino)
        if args.pasos:
            print()
            reproducir_en_consola(
                estados, mapa, principal.camino, limpiar=args.pausa > 0, pausa=args.pausa
            )
        if args.gif:
            _asegurar_directorio(args.gif)
            nombre_nivel = os.path.splitext(os.path.basename(args.nivel))[0]
            subtitulo = "  ·  ".join(
                parte
                for parte in (
                    principal.heuristica,
                    "{} movimientos".format(principal.costo),
                )
                if parte
            )
            generar_gif(
                estados,
                mapa,
                args.gif,
                principal.camino,
                ms_por_paso=args.ms_por_paso,
                titulo="{}  ·  {}".format(nombre_nivel, principal.algoritmo),
                subtitulo=subtitulo,
                celda=args.escala,
            )
            print("\nGIF guardado en {}".format(args.gif))
        if args.animar:
            from sokoban.visualizacion import animar

            animar(estados, mapa, ms_por_paso=args.ms_por_paso)

    if args.salida_json:
        guardar_json(args.salida_json, [r.como_dict() for r in resultados])
    if args.salida_csv:
        guardar_csv(args.salida_csv, [r.como_dict() for r in resultados])

    return 0 if any(r.exito for r in resultados) else 1


if __name__ == "__main__":
    raise SystemExit(main())
