# Ejercicio 3 — Dígitos con `more_digits.csv`

El enunciado en PDF llama al archivo `more_data_digits.csv`; el archivo disponible
es `data/more_digits.csv`. Por ahora está implementada la parte de datos del paso 8
(exploración, comparación con `digits.csv` y partición de desarrollo); el
entrenamiento y la búsqueda están pendientes.

## Exploración de datos y partición

Desde la raíz del repositorio (la carpeta que contiene `tps_sia/`):

```bash
python -m tps_sia.tp3.ej3.src.data_exploration
```

Opciones: `--output-dir` (default `tps_sia/tp3/ej3/results`), `--cache-dir`
(default `tps_sia/tp3/ej3/cache`) y `--skip-test-integrity-check`.

El comando carga los CSV con el loader compartido (`shared/digit_dataset.py`) y un
caché propio en `ej3/cache/`, excluido de Git. No modifica los CSV y genera:

| Archivo | Contenido |
|---|---|
| `data_report.md` / `data_report.json` | Muestras, columnas, dimensión y rango de píxeles, valores no finitos, conteo y proporción por clase con dígitos ausentes, duplicados exactos y conflictos de etiqueta, solapamiento con `digits.csv` y tabla comparativa de clases |
| `split_indices.npz` | Índices `train`, `validation` y `excluded`, referidos al orden de filas de `more_digits.csv` (base 0, sin encabezado) |
| `split_manifest.json` | Semilla, fracción, método, SHA-256 del CSV y de la partición, tamaños, conteos por clase de cada lado y filas excluidas con su motivo |

Dos imágenes son idénticas si todos sus píxeles float32 coinciden (SHA-256 de la
imagen). La partición es estratificada por clase, 80/20, con semilla 42; todas las
copias de una imagen quedan del mismo lado. Las imágenes repetidas con etiquetas
distintas se excluirían de ambos lados porque no tienen un objetivo único; en
`more_digits.csv` no hay ninguna. Sin duplicados, la partición coincide con
`shared.digit_dataset.particionar` con la misma semilla y fracción, que es la que
calcula el runner compartido; el manifiesto lo registra en
`matches_shared_particionar` y guarda `split_sha256` con la misma huella que el runner.
Las rutas guardadas son relativas a `tp3/`, así que las salidas no dependen de la
máquina.

La sección "Comprobación de integridad" del reporte carga `digits_test.csv` sólo
para contar cuántas de sus imágenes aparecen en los archivos de desarrollo. Ninguna
imagen ni etiqueta de test entra en la partición ni en ninguna decisión.

## Tests

```bash
python -m unittest -v tps_sia.tp3.ej3.tests.test_dataset
```

Usan sólo datos sintéticos: grupos de duplicados que no cruzan train/validación,
estratificación aproximada, determinismo con la misma semilla, igualdad con la
partición compartida sin duplicados, detección de conflictos de etiqueta y de clases
ausentes, y salidas del comando sin rutas absolutas.
