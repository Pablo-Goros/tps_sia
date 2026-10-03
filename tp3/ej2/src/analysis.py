"""Compara resultados de ej2 y regenera figuras sin cargar datasets."""
import argparse
from pathlib import Path

from tps_sia.tp3.shared.analysis import generate, generate_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.results_dir.glob('*/results.json'))
    if not paths:
        parser.error('No se encontraron corridas en --results-dir')
    rows = generate_comparison(paths, args.output_dir)
    for path in paths:
        generate(path, args.output_dir / path.parent.name)
    print(f'{len(rows)} corridas completas comparadas; tablas y figuras: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
