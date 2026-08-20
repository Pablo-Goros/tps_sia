#!/usr/bin/env python3
"""Anima paso a paso la solucion de un nivel, como GIF o como video.

Ejemplos:

    python3 visualizar.py --nivel niveles/nivel_03_medio.txt
    python3 visualizar.py --nivel niveles/nivel_03_medio.txt --formato mp4
    python3 visualizar.py --nivel niveles/nivel_04_dificil.txt --algoritmo dfs --formato mp4
    python3 visualizar.py --nivel niveles/nivel_04_dificil.txt --algoritmo todos
    python3 visualizar.py --nivel niveles/nivel_02_facil.txt --frames salidas/frames

El GIF se topea en 400 frames; el video no tiene tope (sirve para ver caminos de
decenas de miles de movimientos sin saltos) y requiere ffmpeg en el PATH.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

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
    EXTENSIONES_VIDEO,
    MAX_FRAMES,
    FfmpegNoDisponible,
    duracion_formateada,
    estados_desde_camino,
    fps_sugerido,
    generar_gif,
    generar_video,
    guardar_frames,
    indices_de_frames,
    render_texto,
)


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Anima paso a paso las soluciones encontradas (GIF o video).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--nivel", required=True, help="ruta al archivo de nivel")
    parser.add_argument(
        "--algoritmo",
        default="astar",
        choices=list(ORDEN_ALGORITMOS) + ["todos"],
        help="metodo a visualizar (default: astar)",
    )
    parser.add_argument(
        "--heuristica", default=HEURISTICA_POR_DEFECTO, choices=sorted(HEURISTICAS)
    )
    parser.add_argument(
        "--formato",
        default="gif",
        choices=["gif", "mp4"],
        help="formato de salida (default: gif). mp4 requiere ffmpeg",
    )
    parser.add_argument(
        "--salida",
        help="ruta del archivo (default: salidas/<nivel>_<algoritmo>.<formato>). "
        "La extension define el formato. Con --algoritmo todos se usa como "
        "carpeta de destino.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        help="cuadros por segundo del video (default: {} para caminos cortos, "
        "{} para los largos)".format(12, 60),
    )
    parser.add_argument(
        "--frames", help="ademas del GIF, guarda un PNG por paso en esta carpeta"
    )
    parser.add_argument(
        "--ms", type=int, default=300, help="milisegundos por paso (default: 300)"
    )
    parser.add_argument(
        "--escala", type=int, default=44, help="lado de cada celda en pixeles"
    )
    parser.add_argument(
        "--submuestreo",
        type=int,
        default=1,
        help="dibuja 1 de cada N pasos (para caminos de cientos de movimientos)",
    )
    parser.add_argument(
        "--sin-encabezado",
        action="store_true",
        help="dibuja solo el tablero, sin barra de progreso ni contadores",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="tope de frames: si el camino es mas largo se dibuja 1 de cada N "
        "pasos. Default {} para GIF y sin tope para video (0 = sin tope)".format(
            MAX_FRAMES
        ),
    )
    parser.add_argument("--sin-poda", action="store_true")
    parser.add_argument("--max-nodos", type=int)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser


def ruta_de_salida(args: argparse.Namespace, clave: str) -> str:
    nombre_nivel = os.path.splitext(os.path.basename(args.nivel))[0]
    predeterminada = os.path.join(
        "salidas", "{}_{}.{}".format(nombre_nivel, clave, args.formato)
    )
    if not args.salida:
        return predeterminada
    if args.algoritmo == "todos":
        return os.path.join(args.salida, os.path.basename(predeterminada))
    return args.salida


def es_video(ruta: str) -> bool:
    return os.path.splitext(ruta)[1].lower() in EXTENSIONES_VIDEO


def tope_de_frames(args: argparse.Namespace, video: bool) -> Optional[int]:
    """El GIF se topea por defecto; el video, no (los frames van en streaming)."""
    if args.max_frames is None:
        return None if video else MAX_FRAMES
    return args.max_frames or None


def titulo_de(resultado: Resultado, nombre_nivel: str) -> str:
    return "{}  ·  {}".format(nombre_nivel, resultado.algoritmo)


def subtitulo_de(resultado: Resultado) -> str:
    partes = []
    if resultado.heuristica:
        partes.append(resultado.heuristica)
    partes.append("{} movimientos".format(resultado.costo))
    return "  ·  ".join(partes)


def _mostrar_progreso(escritos: int, total: int) -> None:
    print(
        "\r         renderizando {}/{} frames ({:.0f}%)".format(
            escritos, total, 100.0 * escritos / total
        ),
        end="\n" if escritos >= total else "",
        flush=True,
    )


def _asegurar_directorio(ruta: str) -> None:
    carpeta = os.path.dirname(os.path.abspath(ruta))
    if carpeta and not os.path.isdir(carpeta):
        os.makedirs(carpeta, exist_ok=True)


def main(argv: Optional[List[str]] = None) -> int:
    args = construir_parser().parse_args(argv)

    try:
        mapa, estado_inicial = cargar_nivel(args.nivel)
    except (OSError, NivelInvalido) as error:
        print("Error al cargar el nivel: {}".format(error), file=sys.stderr)
        return 2

    nombre_nivel = os.path.splitext(os.path.basename(args.nivel))[0]
    print("Nivel: {}".format(args.nivel))
    print(render_texto(estado_inicial, mapa))
    print()

    claves = list(ORDEN_ALGORITMOS) if args.algoritmo == "todos" else [args.algoritmo]
    generados = 0

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

        if not resultado.exito:
            print(
                "{:<8} sin salida: {} (expandio {} nodos en {:.2f} s)".format(
                    ALGORITMOS[clave].nombre,
                    resultado.motivo,
                    resultado.nodos_expandidos,
                    resultado.tiempo_seg,
                )
            )
            continue

        estados = estados_desde_camino(estado_inicial, mapa, resultado.camino)
        destino = ruta_de_salida(args, clave)
        _asegurar_directorio(destino)
        video = es_video(destino)
        tope = tope_de_frames(args, video)
        cantidad = len(indices_de_frames(len(estados), args.submuestreo, tope))

        print(
            "{:<8} {:>6} movimientos, {:>9} nodos expandidos, {:6.2f} s".format(
                resultado.algoritmo,
                resultado.costo,
                resultado.nodos_expandidos,
                resultado.tiempo_seg,
            )
        )

        if video:
            fps = args.fps or fps_sugerido(cantidad)
            print(
                "         {} frames a {} fps ({}) -> {}".format(
                    cantidad, fps, duracion_formateada(cantidad, fps), destino
                )
            )
            try:
                generar_video(
                    estados,
                    mapa,
                    destino,
                    resultado.camino,
                    fps=fps,
                    titulo=titulo_de(resultado, nombre_nivel),
                    subtitulo=subtitulo_de(resultado),
                    celda=args.escala,
                    encabezado=not args.sin_encabezado,
                    submuestreo=args.submuestreo,
                    max_frames=tope,
                    progreso=_mostrar_progreso,
                )
            except FfmpegNoDisponible as error:
                print("         {}".format(error), file=sys.stderr)
                continue
        else:
            generar_gif(
                estados,
                mapa,
                destino,
                resultado.camino,
                ms_por_paso=args.ms,
                titulo=titulo_de(resultado, nombre_nivel),
                subtitulo=subtitulo_de(resultado),
                celda=args.escala,
                encabezado=not args.sin_encabezado,
                submuestreo=args.submuestreo,
                max_frames=tope,
            )
            aviso = (
                "  (1 de cada {} pasos)".format(-(-len(estados) // cantidad))
                if cantidad < len(estados)
                else ""
            )
            print("         {} frames -> {}{}".format(cantidad, destino, aviso))

        generados += 1

        if args.frames:
            carpeta = (
                os.path.join(args.frames, clave) if len(claves) > 1 else args.frames
            )
            rutas = guardar_frames(
                estados,
                mapa,
                carpeta,
                resultado.camino,
                titulo=titulo_de(resultado, nombre_nivel),
                subtitulo=subtitulo_de(resultado),
                celda=args.escala,
                encabezado=not args.sin_encabezado,
                max_frames=args.max_frames or None,
            )
            print("         {} PNG en {}".format(len(rutas), carpeta))

    return 0 if generados else 1


if __name__ == "__main__":
    raise SystemExit(main())
