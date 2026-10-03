# Exploración de datos — ejercicio 3

Generado por `python -m tps_sia.tp3.ej3.src.data_exploration`. Rutas relativas a `tp3/`.

## more_digits.csv

- Archivo: `data/more_digits.csv` (SHA-256 `e0d401943cdf210c…`).
- Muestras: 15741. Columnas del CSV: 2 (`label`, `image`).
- Imagen: 784 píxeles (28×28).
- Rango de píxeles: [0, 1]. Valores no finitos: 0.
- Dígitos ausentes: ninguno.

## Duplicados y conflictos dentro de more_digits.csv

Dos filas son idénticas si todos sus píxeles float32 coinciden (SHA-256 de la imagen).

- Imágenes distintas: 15741 de 15741 filas.
- Grupos de duplicados: 0 (0 filas, 0 copias extra; grupo más grande: 1).
- Grupos por clase: ninguno.
- Conflictos de etiqueta (misma imagen, distinta etiqueta): 0 grupos, 0 filas.

## Solapamiento con digits.csv

- Filas de more_digits con una imagen idéntica en digits.csv: 3689 (misma etiqueta: 3689; distinta etiqueta: 0).
- Filas nuevas de more_digits: 12052.
- Filas de digits.csv presentes en more_digits: 3689 de 12449. digits.csv **no es** subconjunto de more_digits.csv.
- Dígitos ausentes en digits.csv: 8.

## Distribución de clases

| Dígito | digits.csv | % | more_digits.csv | % | Repetidas de digits | Nuevas |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1480 | 11.89 | 1776 | 11.28 | 461 | 1315 |
| 1 | 1685 | 13.54 | 2022 | 12.85 | 495 | 1527 |
| 2 | 1489 | 11.96 | 1787 | 11.35 | 459 | 1328 |
| 3 | 1532 | 12.31 | 1839 | 11.68 | 476 | 1363 |
| 4 | 1460 | 11.73 | 1752 | 11.13 | 424 | 1328 |
| 5 | 271 | 2.18 | 542 | 3.44 | 28 | 514 |
| 6 | 1479 | 11.88 | 1775 | 11.28 | 429 | 1346 |
| 7 | 1566 | 12.58 | 1879 | 11.94 | 476 | 1403 |
| 8 | 0 | 0.00 | 585 | 3.72 | 0 | 585 |
| 9 | 1487 | 11.94 | 1784 | 11.33 | 441 | 1343 |
| **Total** | **12449** | 100 | **15741** | 100 | **3689** | **12052** |

## Partición de desarrollo

Estratificada por clase, 80/20, semilla 42, con grupos de imágenes idénticas en un único lado. Índices en `split_indices.npz`; detalle en `split_manifest.json`.

- Train: 12594. Validación: 3147. Excluidas: 0.
- Coincide con `shared.digit_dataset.particionar` (misma semilla y fracción): sí.

| Dígito | Train | % train | Validación | % validación |
|---|---:|---:|---:|---:|
| 0 | 1421 | 11.28 | 355 | 11.28 |
| 1 | 1618 | 12.85 | 404 | 12.84 |
| 2 | 1430 | 11.35 | 357 | 11.34 |
| 3 | 1471 | 11.68 | 368 | 11.69 |
| 4 | 1402 | 11.13 | 350 | 11.12 |
| 5 | 434 | 3.45 | 108 | 3.43 |
| 6 | 1420 | 11.28 | 355 | 11.28 |
| 7 | 1503 | 11.93 | 376 | 11.95 |
| 8 | 468 | 3.72 | 117 | 3.72 |
| 9 | 1427 | 11.33 | 357 | 11.34 |

## Comprobación de integridad: solapamiento con digits_test.csv

Sólo conteos. Ninguna imagen ni etiqueta de test entra en la partición ni en decisiones.

- Filas de `data/digits_test.csv`: 2497.
- También en more_digits.csv: 0.
- También en digits.csv: 0.
- En ambos: 0.
