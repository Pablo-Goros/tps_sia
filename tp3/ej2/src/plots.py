"""Gráficos del baseline, delegados al análisis compartido sin entrenar."""
import argparse
from pathlib import Path

from tps_sia.tp3.shared.analysis import generate as _generate
from .datos_digitos import EJERCICIO

RESULTS = EJERCICIO / 'results' / 'baseline'


def generate(results_path: Path = RESULTS / 'results.json') -> None:
    _generate(results_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=RESULTS / 'results.json')
    generate(parser.parse_args().results)


if __name__ == '__main__':
    main()
