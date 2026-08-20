"""Interfaz de linea de comandos del paquete Sokoban."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence


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


def construir_parser() -> AnalizadorArgumentos:
    parser = AnalizadorArgumentos(
        prog="python -m sokoban",
        description="Resolución y análisis de niveles de Sokoban.",
    )
    subcomandos = parser.add_subparsers(dest="comando", metavar="COMANDO")

    subcomandos.add_parser(
        "resolver",
        help="resuelve un nivel (implementación completa en la fase 7)",
        description="Resuelve un nivel de Sokoban.",
    )
    experimentar = subcomandos.add_parser(
        "experimentar",
        help="ejecuta experimentos (implementación completa en la fase 8)",
        description="Ejecuta experimentos reproducibles.",
    )
    experimentar.add_argument(
        "--configuracion",
        default="configuracion.json",
        help="ruta del archivo de configuración",
    )
    graficar = subcomandos.add_parser(
        "graficar",
        help="genera figuras (implementación completa en la fase 9)",
        description="Genera figuras a partir de resultados experimentales.",
    )
    graficar.add_argument(
        "--configuracion",
        default="configuracion.json",
        help="ruta del archivo de configuración",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = construir_parser()
    argumentos = parser.parse_args(argv)
    if argumentos.comando is None:
        parser.print_help()
        return 0

    print(
        "El comando {!r} está disponible como interfaz, pero se implementará "
        "en una fase posterior.".format(argumentos.comando)
    )
    return 0


__all__ = ["AnalizadorArgumentos", "COMANDOS", "construir_parser", "main"]
