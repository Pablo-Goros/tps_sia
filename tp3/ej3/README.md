# Ejercicio 3 — Dígitos con `more_digits.csv`

El archivo de datos es `data/more_digits.csv`. Está implementada la exploración, la partición de desarrollo y el código para
controles, búsqueda por etapas, corridas puntuales, estudio de factores y análisis.
Este flujo no evalúa test, no reentrena para entrega ni implementa extensiones opcionales.

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

## Dependencias

Desde la raíz que contiene `tps_sia/`:

```powershell
python -m pip install -r tps_sia/tp3/requirements.txt
```

NumPy se usa para datos y entrenamiento; Matplotlib, para figuras. El análisis
con `--tables-only` no necesita Matplotlib.

## Protocolo y referencia del ejercicio 2

`configs/search.json` fija dataset, partición, presupuesto, candidatos y semillas.
La referencia predeterminada es `ej2/results/v2/selection.json`, la selección del
ejercicio 2, que está versionada; ej3 sólo lee sus hiperparámetros. Si se decide usar
otra selección, pasar su ruta explícitamente:

```powershell
python -m tps_sia.tp3.ej3.src.experiments --stage controls --reference-selection tps_sia/tp3/<ruta>/selection.json --dry-run
```

Repetir `--reference-selection` en todos los comandos de entrenamiento y estudio
de factores, o cambiar `reference_selection` en el protocolo antes de comenzar.
La selección debe estar dentro de `tp3/` para registrar una ruta portable.
Se importan sus hiperparámetros; las rutas de otra máquina se reemplazan por las
de ej3. Los controles se entrenan desde cero con el mismo presupuesto de ej3,
sin cargar pesos del ejercicio 2.

Antes de cualquier corrida se comprueban el hash del CSV, la cobertura completa
de índices y el hash de la partición guardada. Si el CSV del checkout tiene otros
bytes —por ejemplo, otros fines de línea—, regenerar explícitamente los artefactos
de exploración antes de crear un protocolo de corridas nuevo.

## Búsqueda por etapas

Estos comandos ejecutan modelos **cuando el usuario los invoca**. Desde la raíz:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
python -m tps_sia.tp3.ej3.src.experiments --stage controls --workers 4
python -m tps_sia.tp3.ej3.src.experiments --stage rates --workers 4
python -m tps_sia.tp3.ej3.src.experiments --stage architectures --workers 4
python -m tps_sia.tp3.ej3.src.experiments --stage batches --workers 4
python -m tps_sia.tp3.ej3.src.experiments --stage confirmation --workers 4
```

En bash se pueden prefijar los comandos con
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.

| Etapa | Comparación |
|---|---|
| `controls` | Baseline original y configuración seleccionada de ej2 con los datos nuevos |
| `rates` | Tasas configuradas y tasa del ganador de controles; mantiene el resto de hiperparámetros |
| `architectures` | Arquitecturas configuradas y arquitectura ganadora de tasas |
| `batches` | Lotes configurados y lote anterior; mantiene tasa y arquitectura |
| `confirmation` | Mejores candidatos exploratorios y ambos controles con semillas comunes |

Cada etapa requiere la anterior; confirmación utiliza todas las etapas previas.
La exploración usa semilla 42, y la confirmación usa 42, 0 y 1 por defecto.
La partición permanece fija. El checkpoint de cada corrida minimiza pérdida de
validación; las configuraciones se ordenan por accuracy, pérdida, parámetros e id.
En confirmación se usan medias entre semillas. Un grupo con alguna corrida fallida
no puede ganar; los fallos se conservan como evidencia.

El límite inicial es 100 épocas con parada temprana por pérdida de validación,
paciencia 10 y `min_delta=0`. Agotar épocas se registra como tal; no significa
que la configuración sea mala ni que haya convergido. Las tasas incluyen `1e-5`
y `1e-4`; arquitectura y lotes son configurables. Se usa el optimizador del ganador
anterior. No se incorporan nuevas técnicas al núcleo.

Opciones comunes: `--config`, `--output-dir`, `--reference-selection`, `--workers`,
`--verbose`, `--pause-after` y `--dry-run`. El directorio predeterminado es
`ej3/results/search/`. `--dry-run` carga y valida datos/referencias, guarda la
identidad del protocolo y muestra los trabajos preparados; no entrena. Una etapa
posterior sólo se puede preparar si existen las decisiones previas.

## Corridas puntuales e iteración

`--experiment` acepta un JSON de cambios respecto de la referencia de ej2.
Para variar tasa y presupuesto, por ejemplo:

```json
{
  "optimizer": {"learning_rate": 0.03},
  "epochs": 200,
  "stopping": {"patience": 20}
}
```

```powershell
python -m tps_sia.tp3.ej3.src.experiments --experiment tps_sia/tp3/ej3/configs/experiment.example.json --seed 42
```

Los objetos de optimizador, parada y preprocesamiento se combinan con sus valores
anteriores; los demás campos reemplazan el valor completo. El dataset y los índices
son los de ej3. La semilla se elige con `--seed`. Las corridas puntuales aparecen en
el análisis, pero no se agregan automáticamente a los finalistas de la búsqueda.

Para iterar la búsqueda, copiar `search.json`, cambiar candidatos o presupuesto y
usar otro `--output-dir`. Una tasa ganadora en un extremo de la grilla o curvas que
siguen mejorando al agotar épocas son motivos para que el usuario amplíe la grilla
o el presupuesto. No hay ampliaciones automáticas.

## Reanudación y artefactos

Repetir el mismo comando reutiliza corridas terminadas compatibles y reanuda las
interrumpidas desde el checkpoint. `--pause-after 5` pausa después de cinco épocas
en esa invocación; quitarlo al continuar evita una nueva pausa programada. El
presupuesto de épocas es total, contando las épocas anteriores. Un checkpoint
conserva parámetros, optimizador, RNG, contadores e historia según el runner común.

| Archivo | Contenido |
|---|---|
| `protocol.json` | Identidad del protocolo, referencias, dataset y partición |
| `runs/<config_id>-seed-<seed>/` | Configuración, manifiesto, índices, historia, checkpoint, mejor modelo y resultados |
| `stage_<nombre>.json` | Corridas, ranking y decisión de cada etapa |
| `selection.json` | Configuración seleccionada por validación y evidencia de confirmación |
| `custom-<hash>-seed-<seed>.json` | Referencia a una corrida puntual |

La identidad de una corrida incluye sus índices de entrenamiento, validación y
fuentes. Cambiar protocolo, referencia, dataset o partición requiere otro directorio;
no se mezclan corridas viejas y nuevas. Las rutas guardadas son relativas a `tp3/`.
`validation_target_met` indica si la accuracy **media de validación** alcanza 98 %;
no demuestra el objetivo en generalización y no produce un modelo de entrega.

## Estudio de factores

Después de confirmación, ejecutar partes individuales o todas explícitamente:

```powershell
python -m tps_sia.tp3.ej3.src.factor_study --part dataset --workers 4
python -m tps_sia.tp3.ej3.src.factor_study --part coverage --workers 4
python -m tps_sia.tp3.ej3.src.factor_study --part size --workers 4
python -m tps_sia.tp3.ej3.src.factor_study --part techniques --workers 4
# Alternativa: --part all
```

Opciones: `--config`, `--output-dir`, `--selection`, `--reference-selection`,
`--workers`, `--verbose`, `--pause-after`, `--dry-run`. Si la búsqueda se guardó en
otro directorio, indicar su `selection.json` con `--selection`. El estudio requiere
una selección del mismo protocolo. El directorio predeterminado es
`ej3/results/factors/`.

| Parte | Comparación con validación común de more_digits |
|---|---|
| `dataset` | Configuración de ej2 sobre digits y sobre train de more_digits; se excluyen de digits todas las imágenes idénticas a validación |
| `coverage` | Configuración de ej2 sobre train de more_digits con y sin ejemplos del 8 |
| `size` | Configuración de ej2 sobre subconjuntos estratificados anidados del 25 %, 50 % y 100 % de train |
| `techniques` | Configuraciones de ej2 y ej3 sobre el mismo train completo |

Las semillas y fracciones se configuran en `factor_study`. Cada subconjunto conserva
las clases presentes mediante prefijos de un orden aleatorio por clase; las
proporciones son aproximadas, especialmente en clases pequeñas. Se guardan índices,
conteos, exclusiones y procedencia. El runner rechaza imágenes idénticas en train
y validación, también entre archivos distintos.

La comparación entre datasets cambia también imágenes, cantidad y composición;
no aísla únicamente tamaño. Quitar el 8 reduce además la cantidad de entrenamiento.
Estas limitaciones quedan registradas para interpretar los resultados después.

## Tablas y gráficos sin ejecutar el modelo

```powershell
python -m tps_sia.tp3.ej3.src.analysis
# Sólo tablas, sin Matplotlib:
python -m tps_sia.tp3.ej3.src.analysis --tables-only
```

Opciones: `--results-dir`, `--factor-dir`, `--output-dir` y `--tables-only`.
Se procesan sólo etapas, corridas puntuales y partes del estudio ya guardadas.
El comando no carga CSV, pesos ni checkpoints, no predice y no entrena.

Genera por comparación:

- `comparison.csv`: hiperparámetros, estado, época elegida, tiempo, muestras y métricas.
- `seed_summary.csv` / `summary.json`: medias y desvío muestral entre semillas;
  con una semilla el desvío es `null`.
- `per_class.csv`: precision, recall y F1, con 5 y 8 marcados.
- `paired_differences.csv` para factores: diferencias emparejadas por semilla
  respecto de la primera variante, incluida accuracy sin 8 y recall de 5/8.
- `comparison.md`: tabla descriptiva y condiciones de comparación, sin conclusiones.
- Figuras de accuracy, curvas de train/validación y confusiones crudas/normalizadas
  por corrida; incluye evolución de pesos si se habilitó su registro.

Los resultados de entrenamiento y análisis se guardan en directorios separados.
No se evalúa `digits_test.csv` ni se generan informes finales.

## Chequeos sin entrenamiento

```powershell
python -B -m unittest -v tps_sia.tp3.ej3.tests.test_workflow
```

Usan CSV y mediciones sintéticos en carpetas temporales y bloquean `MLP.entrenar`.
Cubren el flujo de etapas, controles de procedencia, referencias faltantes,
subconjuntos anidados, exclusión de solapamientos, identidad por índices,
despacho de reanudación/reutilización y análisis de tablas/figuras desde JSON.
No comprueban convergencia ni accuracy de un modelo real. El chequeo de figuras
requiere Matplotlib. Los tests existentes de datasets siguen disponibles aparte.
