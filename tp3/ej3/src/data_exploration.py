"""Explore more_digits.csv and freeze the ej3 development split.

Writes data_report.json/.md, split_indices.npz and split_manifest.json to
--output-dir. Every path stored in the outputs is relative to tp3/.

The test-overlap integrity check loads digits_test.csv only to count images
that also appear in the development files. Test images and labels never enter
the split, the report tables or any selection.

    python -m tps_sia.tp3.ej3.src.data_exploration [--output-dir DIR]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from tps_sia.tp3.shared.digit_dataset import cargar, particionar
from tps_sia.tp3.shared.experiments import fingerprint, portable_path, sha256_file, write_json
from tps_sia.tp3.ej3.src.dataset import (class_counts, duplicate_summary, grouped_stratified_split,
                                         hash_overlap_counts, image_hashes, labels_of,
                                         overlap_summary, split_summary)

TP3 = Path(__file__).resolve().parents[2]
DATA = TP3 / 'data'
MORE_DIGITS = DATA / 'more_digits.csv'
DIGITS = DATA / 'digits.csv'
DIGITS_TEST = DATA / 'digits_test.csv'
CACHE = TP3 / 'ej3' / 'cache'
OUTPUT = TP3 / 'ej3' / 'results'
SEED = 42
VALIDATION_FRACTION = 0.2


def csv_columns(path: Path) -> list[str]:
    with path.open(encoding='utf-8-sig', newline='') as source:
        return next(csv.reader(source))


def describe(path: Path, X: np.ndarray, labels: np.ndarray) -> dict:
    side = int(round(np.sqrt(X.shape[1])))
    return {
        'path': portable_path(path, TP3),
        'sha256': sha256_file(path),
        'rows': int(len(X)),
        'columns': csv_columns(path),
        'pixels_per_image': int(X.shape[1]),
        'image_shape': [side, side] if side * side == X.shape[1] else None,
        'pixel_min': float(X.min()),
        'pixel_max': float(X.max()),
        'non_finite_values': int((~np.isfinite(X)).sum()),
        'classes': class_counts(labels),
    }


def integrity_check_test_overlap(test_csv: Path, cache_dir: Path, hashes: dict[str, list[str]]) -> dict:
    """Counts of test images also present in the development files. Counts only."""
    X_test, _ = cargar(test_csv, cache_dir / 'digits_test.npz')
    counts = hash_overlap_counts(image_hashes(X_test), hashes)
    return {'path': portable_path(test_csv, TP3), **counts,
            'purpose': 'Integrity check only; no test image or label is used for the split or for any decision.'}


def build_manifest(more: dict, labels: np.ndarray, split: dict, matches_shared: bool) -> dict:
    summary = split_summary(labels, split)
    return {
        'schema_version': 1,
        'dataset': {'path': more['path'], 'sha256': more['sha256'], 'rows': more['rows']},
        'seed': SEED,
        'validation_fraction': VALIDATION_FRACTION,
        'method': ('Per class, groups of identical images (SHA-256 of float32 pixels) are shuffled '
                   'with numpy default_rng(seed) and moved to validation until round(n * fraction) '
                   'samples; all copies of an image stay on one side.'),
        'indices_file': 'split_indices.npz',
        'index_base': 'zero-based row order of the CSV, header excluded',
        'split_sha256': fingerprint({'train': split['train'].tolist(),
                                     'validation': split['validation'].tolist()}),
        'matches_shared_particionar': matches_shared,
        **summary,
        'excluded': split['excluded'],
    }


def class_table(digits: dict, more: dict, overlap: dict) -> list[str]:
    rows = ['| Dígito | digits.csv | % | more_digits.csv | % | Repetidas de digits | Nuevas |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for d in map(str, range(10)):
        rows.append(f"| {d} | {digits['classes']['counts'][d]} | {100 * digits['classes']['proportions'][d]:.2f} "
                    f"| {more['classes']['counts'][d]} | {100 * more['classes']['proportions'][d]:.2f} "
                    f"| {overlap['by_class'][d]['same_label'] + overlap['by_class'][d]['different_label']} "
                    f"| {overlap['by_class'][d]['new']} |")
    rows.append(f"| **Total** | **{digits['rows']}** | 100 | **{more['rows']}** | 100 "
                f"| **{overlap['rows_with_identical_image']}** | **{overlap['new_rows']}** |")
    return rows


def missing_text(classes: dict) -> str:
    missing = classes['missing_classes']
    return 'ninguno' if not missing else ', '.join(map(str, missing))


def markdown(report: dict) -> str:
    more, digits = report['more_digits'], report['digits']
    dup, overlap, split = report['duplicates'], report['overlap_with_digits'], report['split']
    conflicts = dup['label_conflicts']
    lines = [
        '# Exploración de datos — ejercicio 3', '',
        'Generado por `python -m tps_sia.tp3.ej3.src.data_exploration`. Rutas relativas a `tp3/`.', '',
        '## more_digits.csv', '',
        f"- Archivo: `{more['path']}` (SHA-256 `{more['sha256'][:16]}…`).",
        f"- Muestras: {more['rows']}. Columnas del CSV: {len(more['columns'])} (`{'`, `'.join(more['columns'])}`).",
        f"- Imagen: {more['pixels_per_image']} píxeles ({more['image_shape'][0]}×{more['image_shape'][1]}).",
        f"- Rango de píxeles: [{more['pixel_min']:g}, {more['pixel_max']:g}]. Valores no finitos: {more['non_finite_values']}.",
        f"- Dígitos ausentes: {missing_text(more['classes'])}.", '',
        '## Duplicados y conflictos dentro de more_digits.csv', '',
        'Dos filas son idénticas si todos sus píxeles float32 coinciden (SHA-256 de la imagen).', '',
        f"- Imágenes distintas: {dup['unique_images']} de {more['rows']} filas.",
        f"- Grupos de duplicados: {dup['duplicate_groups']} ({dup['rows_in_duplicate_groups']} filas, "
        f"{dup['extra_copies']} copias extra; grupo más grande: {dup['largest_group']}).",
        f"- Grupos por clase: {dup['duplicate_groups_by_class'] or 'ninguno'}.",
        f"- Conflictos de etiqueta (misma imagen, distinta etiqueta): {conflicts['groups']} grupos, {conflicts['rows']} filas.",
    ]
    lines += [f"  - filas {c['rows']}: etiquetas {c['labels']}" for c in conflicts['details']]
    lines += [
        '', '## Solapamiento con digits.csv', '',
        f"- Filas de more_digits con una imagen idéntica en digits.csv: {overlap['rows_with_identical_image']} "
        f"(misma etiqueta: {overlap['same_label']}; distinta etiqueta: {overlap['different_label']}).",
        f"- Filas nuevas de more_digits: {overlap['new_rows']}.",
        f"- Filas de digits.csv presentes en more_digits: {overlap['reference_rows_found']} de {overlap['reference_rows']}. "
        f"digits.csv {'**es**' if overlap['reference_is_subset'] else '**no es**'} subconjunto de more_digits.csv.",
        f"- Dígitos ausentes en digits.csv: {missing_text(digits['classes'])}.", '',
        '## Distribución de clases', '',
        *class_table(digits, more, overlap), '',
        '## Partición de desarrollo', '',
        f"Estratificada por clase, {100 * (1 - split['validation_fraction']):.0f}/{100 * split['validation_fraction']:.0f}, "
        f"semilla {split['seed']}, con grupos de imágenes idénticas en un único lado. "
        f"Índices en `{split['indices_file']}`; detalle en `split_manifest.json`.", '',
        f"- Train: {split['train_size']}. Validación: {split['validation_size']}. Excluidas: {split['excluded_rows']}.",
        f"- Coincide con `shared.digit_dataset.particionar` (misma semilla y fracción): "
        f"{'sí' if split['matches_shared_particionar'] else 'no'}.", '',
        '| Dígito | Train | % train | Validación | % validación |', '|---|---:|---:|---:|---:|',
    ]
    for d in map(str, range(10)):
        lines.append(f"| {d} | {split['train']['counts'][d]} | {100 * split['train']['proportions'][d]:.2f} "
                     f"| {split['validation']['counts'][d]} | {100 * split['validation']['proportions'][d]:.2f} |")
    integrity = report.get('integrity_check_test_overlap')
    lines += ['', '## Comprobación de integridad: solapamiento con digits_test.csv', '']
    if integrity is None:
        lines.append('No ejecutada (`--skip-test-integrity-check`).')
    else:
        lines += [
            'Sólo conteos. Ninguna imagen ni etiqueta de test entra en la partición ni en decisiones.', '',
            f"- Filas de `{integrity['path']}`: {integrity['rows']}.",
            f"- También en more_digits.csv: {integrity['rows_also_in_more_digits']}.",
            f"- También en digits.csv: {integrity['rows_also_in_digits']}.",
            f"- En ambos: {integrity['rows_also_in_all']}.",
        ]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--cache-dir', type=Path, default=CACHE)
    parser.add_argument('--skip-test-integrity-check', action='store_true')
    args = parser.parse_args(argv)
    output, cache = args.output_dir.resolve(), args.cache_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    X, y = cargar(MORE_DIGITS, cache / 'more_digits.npz')
    X_digits, y_digits = cargar(DIGITS, cache / 'digits.npz')
    labels, labels_digits = labels_of(y), labels_of(y_digits)
    hashes, hashes_digits = image_hashes(X), image_hashes(X_digits)

    more = describe(MORE_DIGITS, X, labels)
    digits = describe(DIGITS, X_digits, labels_digits)
    split = grouped_stratified_split(labels, hashes, SEED, VALIDATION_FRACTION)
    *_, shared_train, shared_validation = particionar(
        X, y, SEED, return_indices=True, validation_fraction=VALIDATION_FRACTION)
    matches = (np.array_equal(shared_train, split['train'])
               and np.array_equal(shared_validation, split['validation']))
    manifest = build_manifest(more, labels, split, matches)

    report = {
        'schema_version': 1,
        'more_digits': more,
        'digits': digits,
        'duplicates': duplicate_summary(hashes, labels),
        'overlap_with_digits': overlap_summary(hashes, labels, hashes_digits, labels_digits),
        'split': {k: v for k, v in manifest.items() if k != 'excluded'},
    }
    if not args.skip_test_integrity_check:
        report['integrity_check_test_overlap'] = integrity_check_test_overlap(
            DIGITS_TEST, cache, {'more_digits': hashes, 'digits': hashes_digits})

    np.savez_compressed(output / 'split_indices.npz', train=split['train'], validation=split['validation'],
                        excluded=np.array([i for e in split['excluded'] for i in e['rows']], dtype=np.int64))
    write_json(output / 'split_manifest.json', manifest)
    write_json(output / 'data_report.json', report)
    (output / 'data_report.md').write_text(markdown(report), encoding='utf-8')
    print(f"more_digits: {more['rows']} rows; train {manifest['train_size']}, "
          f"validation {manifest['validation_size']}, excluded {manifest['excluded_rows']}")
    print(f'Outputs in {output}')


if __name__ == '__main__':
    main()
