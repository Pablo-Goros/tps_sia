"""Interfaz de linea de comandos del paquete Sokoban."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, TextIO

from .busqueda import ALGORITMOS, ORDEN_ALGORITMOS, Resultado, resolver
from .estado import ABREVIATURAS, Direccion, Estado
from .experimentos import ErrorConfiguracion, ejecutar_desde_configuracion
from .heuristicas import HEURISTICAS, HEURISTICA_POR_DEFECTO
from .nivel import NivelInvalido, cargar_nivel, render_texto
from .problema import ProblemaSokoban


COMANDOS = ("resolver", "experimentar", "graficar")


class AnalizadorArgumentos(argparse.ArgumentParser):
    """ArgumentParser con las etiquetas visibles en español."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["add_help"] = False
        super().__init__(*args, **kwargs)
        self._positionals.title = "argumentos posicionales"
        self._optionals.title = "opciones"
        self.add_argument(
            "-h",
            "--help",
            "--ayuda",
            action="help",
            help="muestra esta ayuda y termina",
        )

    def format_help(self) -> str:
        return super().format_help().replace("usage:", "uso:", 1)

    def format_usage(self) -> str:
        return super().format_usage().replace("usage:", "uso:", 1)


class ErrorCLI(ValueError):
    """Error de configuración que puede mostrarse sin un traceback."""


def _entero_positivo(texto: str) -> int:
    try:
        valor = int(texto)
    except ValueError as error:
        raise argparse.ArgumentTypeError("debe ser un número entero") from error
    if valor <= 0:
        raise argparse.ArgumentTypeError("debe ser mayor que cero")
    return valor


def construir_parser() -> AnalizadorArgumentos:
    parser = AnalizadorArgumentos(
        prog="python -m sokoban",
        description="Resolución y análisis de niveles de Sokoban.",
    )
    subcomandos = parser.add_subparsers(dest="comando", metavar="COMANDO")

    resolver_parser = subcomandos.add_parser(
        "resolver",
        help="resuelve un nivel y muestra sus métricas",
        description="Resuelve un nivel de Sokoban.",
    )
    resolver_parser.add_argument(
        "--nivel",
        required=True,
        metavar="RUTA",
        help="ruta del archivo de nivel en formato XSB",
    )
    resolver_parser.add_argument(
        "--algoritmo",
        choices=ORDEN_ALGORITMOS,
        default="astar",
        help="algoritmo de búsqueda (por defecto: astar)",
    )
    resolver_parser.add_argument(
        "--heuristica",
        choices=tuple(sorted(HEURISTICAS)),
        default=None,
        help=(
            "heurística para greedy o astar "
            "(por defecto: {})".format(HEURISTICA_POR_DEFECTO)
        ),
    )
    resolver_parser.add_argument(
        "--max-expandidos",
        type=_entero_positivo,
        metavar="N",
        help="corta la búsqueda antes de expandir más de N nodos",
    )
    resolver_parser.add_argument(
        "--mostrar-estados",
        action="store_true",
        help="muestra en ASCII el estado inicial y cada movimiento",
    )

    experimentar = subcomandos.add_parser(
        "experimentar",
        help="ejecuta experimentos definidos por configuración",
        description="Ejecuta experimentos reproducibles.",
    )
    experimentar.add_argument(
        "--configuracion",
        default="configuracion.json",
        help="ruta del archivo de configuración",
    )
    graficar = subcomandos.add_parser(
        "graficar",
        help="genera figuras desde resultados experimentales",
        description="Genera figuras a partir de resultados experimentales.",
    )
    graficar.add_argument(
        "--configuracion",
        default="configuracion.json",
        help="ruta del archivo de configuración",
    )
    return parser


def _validar_resolver(argumentos: argparse.Namespace) -> None:
    descripcion = ALGORITMOS[argumentos.algoritmo]
    if argumentos.heuristica is not None and not descripcion.usa_heuristica:
        raise ErrorCLI(
            "el algoritmo {} no acepta heurísticas".format(argumentos.algoritmo)
        )


def _abreviar_movimientos(movimientos: Sequence[Direccion]) -> str:
    return "".join(ABREVIATURAS[movimiento] for movimiento in movimientos)


def imprimir_resultado(
    resultado: Resultado,
    salida: Optional[TextIO] = None,
) -> None:
    """Imprime todos los campos públicos de un resultado de búsqueda."""
    if salida is None:
        salida = sys.stdout
    print("Estado:                  {}".format(resultado.estado), file=salida)
    print("Algoritmo:               {}".format(resultado.algoritmo), file=salida)
    print(
        "Heurística:              {}".format(resultado.heuristica or "-"),
        file=salida,
    )
    print(
        "Costo:                   {}".format(
            resultado.costo if resultado.costo is not None else "-"
        ),
        file=salida,
    )
    print(
        "Movimientos:             {}".format(
            ", ".join(resultado.movimientos) or "-"
        ),
        file=salida,
    )
    print(
        "Secuencia (U/D/L/R):     {}".format(
            _abreviar_movimientos(resultado.movimientos) or "-"
        ),
        file=salida,
    )
    print(
        "Nodos expandidos:        {}".format(resultado.nodos_expandidos),
        file=salida,
    )
    print(
        "Nodos en frontera:       {}".format(resultado.nodos_frontera),
        file=salida,
    )
    print(
        "Máximo en frontera:      {}".format(resultado.max_nodos_frontera),
        file=salida,
    )
    print(
        "Tiempo (segundos):       {:.6f}".format(resultado.tiempo_segundos),
        file=salida,
    )
    print("Motivo:                  {}".format(resultado.motivo), file=salida)


def _estados_del_camino(
    problema: ProblemaSokoban,
    movimientos: Sequence[Direccion],
) -> list[Estado]:
    estados = [problema.estado_inicial]
    actual = problema.estado_inicial
    for movimiento in movimientos:
        siguiente = problema.aplicar_accion(actual, movimiento)
        if siguiente is None:
            raise RuntimeError("la solución contiene un movimiento inválido")
        estados.append(siguiente)
        actual = siguiente
    return estados


def imprimir_estados(
    problema: ProblemaSokoban,
    movimientos: Sequence[Direccion],
    salida: Optional[TextIO] = None,
) -> None:
    """Reproduce una solución usando solamente el formato XSB de texto."""
    if salida is None:
        salida = sys.stdout
    for paso, estado in enumerate(_estados_del_camino(problema, movimientos)):
        if paso == 0:
            encabezado = "Paso 0 (estado inicial)"
        else:
            encabezado = "Paso {} ({})".format(paso, movimientos[paso - 1])
        print("\n{}".format(encabezado), file=salida)
        print(render_texto(estado, problema.mapa), file=salida)


def _resolver(argumentos: argparse.Namespace) -> int:
    _validar_resolver(argumentos)
    try:
        mapa, estado_inicial = cargar_nivel(Path(argumentos.nivel))
    except (OSError, NivelInvalido) as error:
        raise ErrorCLI("no se pudo cargar el nivel: {}".format(error)) from error

    problema = ProblemaSokoban(mapa, estado_inicial)
    resultado = resolver(
        problema,
        argumentos.algoritmo,
        heuristica=argumentos.heuristica,
        max_expandidos=argumentos.max_expandidos,
    )
    imprimir_resultado(resultado)

    if argumentos.mostrar_estados and resultado.exito:
        imprimir_estados(problema, resultado.movimientos)

    return 0 if resultado.exito else 1


def _experimentar(argumentos: argparse.Namespace) -> int:
    try:
        resultado = ejecutar_desde_configuracion(
            argumentos.configuracion,
            informar=print,
        )
    except ErrorConfiguracion as error:
        raise ErrorCLI("configuración inválida: {}".format(error)) from error

    print("\nEjecuciones: {}".format(len(resultado.ejecuciones)))
    print("Resumen:     {} grupos".format(len(resultado.resumen)))
    print("CSV crudo:   {}".format(resultado.salida_ejecuciones))
    print("CSV resumen: {}".format(resultado.salida_resumen))
    return 0


def _fase_posterior(comando: str) -> int:
    raise ErrorCLI(
        "el comando {} se implementará en la fase 9".format(comando)
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = construir_parser()
    argumentos = parser.parse_args(argv)
    if argumentos.comando is None:
        parser.print_help()
        return 0

    try:
        if argumentos.comando == "resolver":
            return _resolver(argumentos)
        if argumentos.comando == "experimentar":
            return _experimentar(argumentos)
        return _fase_posterior(argumentos.comando)
    except ErrorCLI as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 2


__all__ = [
    "AnalizadorArgumentos",
    "COMANDOS",
    "ErrorCLI",
    "construir_parser",
    "imprimir_estados",
    "imprimir_resultado",
    "main",
]
