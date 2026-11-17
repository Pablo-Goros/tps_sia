# TP5: autoencoder básico de caracteres

## Ejecutar desde otra computadora

Clonar el repositorio completo en la rama `feat/tp5` (o hacer `git pull` en esa
rama). TP5 importa `tp3/shared`: no alcanza con copiar solamente `tp5/`.
El MLP compartido incluye BCE, código lineal y scheduler cosenoidal, necesarios
para las configuraciones y modelos publicados.

Desde la raíz del repositorio, en PowerShell:

```powershell
cd tp5
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m unittest discover -s tests -p "test_*.py" -v
.venv\Scripts\python -m src.train evaluate --model results/final/model.npz --output results/final-local-evaluation
.venv\Scripts\python -m src.denoise evaluate --campaign results/denoising --output results/denoising-local-evaluation
```

En Linux/macOS usar `.venv/bin/python` en lugar de `.venv\Scripts\python`.
Las evaluaciones cargan los modelos publicados y no entrenan. Cada salida debe
ser nueva; cambiar su nombre para repetir los comandos. Para crear una campaña
de denoising desde cero, seguir [denoising-data.md](denoising-data.md) usando una
salida nueva, por ejemplo `results/denoising-new`.

Se publican el modelo final de 1.a (`results/final/`), sus 43 corridas
(`results/presentation-campaign-2026-11-10/`), la campaña completa de 1.b
(`results/denoising/`) y los scripts y recursos de presentación (`ppt/`).
Los manifiestos conservan las rutas históricas de la máquina de origen;
los comandos anteriores usan las rutas de la copia local.

El entorno original fue Python 3.14.4, NumPy 2.4.4 y Matplotlib 3.11.1.
`requirements.txt` permite instalar versiones compatibles; para usar las
versiones originales de NumPy y Matplotlib, instalar
`requirements-reproduction.txt` en un entorno con Python 3.14.4.
Git conserva los bytes de los artefactos y usa CRLF para `data/font.h`, como
en los hashes de estas campañas, también en Linux/macOS.

La selección vigente y la comparación confirmatoria están en
[presentation-results.md](presentation-results.md). Configuración reproducible:
[configs/presentation-selected.json](configs/presentation-selected.json).
El modelo usa BCE, Adam 0.002 constante, arquitectura deeper y latente tanh;
la semilla seleccionada reconstruye los 32 caracteres sin errores.

### Opciones de la campaña de la presentación

Las configuraciones de entrenamiento admiten `loss` (`half_squared_error` o
`bce`), `latent_activation` (`tanh` o `linear`) y `lr_scheduler` (`null` o
`{"name": "cosine", "total_epochs": 3000, "min_lr": 0.0}`). Los valores
predeterminados preservan el comportamiento anterior. BCE se calcula desde
logits y conserva la suma sobre píxeles/media sobre muestras. `history.costo`
registra la pérdida optimizada; `metrics.training_loss` conserva el medio error
cuadrático para compatibilidad. El latente lineal se explora dentro del rango
ocupado por los códigos, sin exigir [-1,1].

Para entrenar la configuración seleccionada en una salida nueva:

```bash
python -m src.train train --config configs/presentation-selected.json --output results/presentation-selected-reproduction
```

`src.presentation_campaign --output <directorio-vacio>` reproduce el árbol
completo del [plan](experiment_plan.md), incluidas las referencias desde cero,
con un máximo de 43 corridas y 321000 épocas. Registra las decisiones, congela
el finalista antes de usar semillas nuevas y verifica cada modelo al recargar.
Ese comando genera evidencia y no publica una selección final por sí solo.
Una reproducción consume un nuevo presupuesto de entrenamiento.

El [informe de resultados y pendientes](results/report.md) relaciona los cuatro puntos del apartado 1.a con la evidencia disponible.

La comparación reproducible y su protocolo autorizado se describen en [comparison.md](comparison.md).

Ejercicio 1.a. Todos los comandos siguientes se ejecutan desde `tp5/`, con Python 3.10 o posterior.

## Instalación

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Inspección del dataset

```powershell
python -m src.dataset --output results/dataset
```

Produce `results/dataset/dataset.png`, con los 32 patrones etiquetados, y
`dataset.json`, con identidad SHA256, etiquetas y dimensiones. No requiere una
ventana gráfica. Para leer otra copia del header, agregar `--dataset ruta/font.h`.

`src.dataset.load_dataset(path)` expone `images` `(32,7,5)`, `vectors` flotantes
`(32,35)`, `labels`, `sha256` y `path`. El argumento es opcional y por defecto
apunta al original `data/font.h`, independientemente del directorio actual.
Lee únicamente la matriz `Font3[32][7]` sin ejecutar código ni incorporar los
comentarios. Conserva los códigos `0x60` a `0x7f`, incluido DEL, y convierte cada
fila del bit más significativo al menos significativo, de izquierda a derecha.
El aplanado conserva el orden de las filas de arriba abajo.

El original se conserva sin cambios. SHA256:
`095b431bdfa2d228eeaa337fb45cd963088cc13380c189e01436e446767502f8`.

## Verificación

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Las pruebas del flujo público verifican contenido y orientación asimétrica,
identidad del original, rechazo de datos mal formados y generación de la lámina
con su manifiesto. También cubren un entrenamiento corto y persistencia, sin
exigir convergencia del experimento.

## Entrenamiento y evaluación

```powershell
python -m src.train train --config configs/pilot.json --output results/pilot
python -m src.train evaluate --model results/pilot/model.npz --output results/pilot-evaluation
```

El directorio de entrenamiento debe estar vacío para conservar las corridas
anteriores. Para repetir el piloto, usar por ejemplo `--output results/pilot-repeat`.
La evaluación carga el modelo guardado sin entrenar. Ambos comandos admiten
`--dataset ruta/font.h`; siempre se validan los 32 patrones.

`configs/pilot.json` registra el presupuesto autorizado: arquitectura
`35→20→2→20→35`, tanh interna (incluido el código), salida logística, Adam,
semilla 42, tasa 0.001, lote 32, máximo 10.000 épocas y evaluación cada 100.
Se detiene si en una evaluación todos los patrones tienen como máximo un error.
El umbral binario es `>= 0.5`. La evaluación periódica conserva los pesos, el
generador aleatorio y los momentos de Adam. La interfaz `Autoencoder` encapsula
el MLP de `tp3.shared` sin duplicar su entrenamiento ni retropropagación.

Los campos JSON obligatorios son `architecture`, `seed`, `optimizer`,
`batch_size`, `max_epochs`, `evaluate_every` y `stop_on_tolerance`. La arquitectura
tiene 35 entradas y salidas y dos unidades en la capa central; las capas ocultas
usan tanh. `optimizer` utiliza la configuración pública del TP3, por ejemplo
`{"name": "adam", "learning_rate": 0.001}`. Cambiar una configuración para un
experimento exige acordar su presupuesto antes de ejecutarlo.

Cada corrida conserva `config.json`, `manifest.json` (identidad del dataset y
modelo, versiones, duración, épocas y motivo de parada), `model.npz`,
`history.json`, `metrics.json`, `predictions.npz`, `curves.png` y
`reconstructions.png`. Las rutas absolutas de los manifiestos documentan el
entorno de ejecución original; los artefactos se trasladan juntos y los hashes
permiten comprobar su identidad. La evaluación admite la nueva ruta del modelo.

Las métricas incluyen errores individuales, máximo y promedio, fracción exacta
y fracción dentro de tolerancia. `all_within_tolerance` exige que **cada uno**
de los 32 patrones tenga como máximo un píxel erróneo. `mse_per_pixel` promedia
los 1120 píxeles; `training_loss` es `0.5 * mean(sum(error², axis=1))`, es decir,
17.5 veces el MSE por píxel. Las curvas usan MSE y máximo error por patrón;
las láminas muestran original a la izquierda y reconstrucción binaria a la
derecha con sus errores. Una pérdida pequeña no sustituye la tolerancia.

Los 32 patrones son a la vez entradas y objetivos. Estos resultados miden
reconstrucción del entrenamiento; no estiman generalización a letras nuevas.
El registro del piloto está en [results/pilot/README.md](results/pilot/README.md).
La comparación de arquitecturas y optimizadores requiere revisar esta evidencia
y acordar otro presupuesto. Un piloto fallido no demuestra imposibilidad ni
autoriza reducir el dataset.

## Visualización del espacio latente

```powershell
python -m src.explore --model results/pilot/model.npz --output results/latent
```

El comando carga el modelo guardado y codifica los 32 patrones originales sin
entrenar. Para visualizar el modelo elegido por la comparación, sustituir
`--model` por la ruta de su archivo `model.npz`. Se admite `--dataset ruta/font.h`.

Produce `latent.png` y `latent.json`. El JSON conserva exactamente 32 puntos en
el orden del dataset: `code` numérico (96–127), `label` visible, incluidas las
etiquetas de símbolos y DEL, y `coordinates` con dos valores. Las coordenadas
son la salida de `Autoencoder.encode` del modelo cargado. La figura usa esas
mismas coordenadas y asigna a cada patrón una posición de etiqueta separada,
con una línea hacia su punto, incluso cuando varias coordenadas coinciden.
La figura y el JSON incluyen los SHA256 del modelo y del dataset. Las rutas del
JSON describen el entorno original; para regenerar después de trasladar los
artefactos, pasar sus nuevas rutas al comando y comprobar los hashes.

La prueba `test_explore.py` recarga un modelo, compara los 32 códigos con la
interfaz pública, verifica etiquetas, archivos y hashes y repite la generación.
No exige una distribución particular ni considera la separación visual como
prueba de reconstrucción o generalización. El comando se puede ejecutar sobre
cualquier modelo válido guardado, sin repetir su entrenamiento.

## Interpolación, grilla y revisión visual

```powershell
python -m src.explore --model results/pilot/model.npz --config configs/exploration.json --output results/candidates
```

La configuración especifica `pairs` como pares de códigos decimales 96–127,
`steps` como cantidad de puntos por interpolación (incluye extremos), y
`grid_size` como cantidad de puntos por eje de la grilla. Las dos resoluciones
admiten enteros de 2 a 50. La grilla recorre el rectángulo ocupado por los
códigos de los 32 patrones; todas las coordenadas permanecen en `[-1,1]`.
Esta exploración de un modelo guardado no ejecuta entrenamiento.

Se producen `interpolations.png`, `grid.png` y `candidates.json`. Cada imagen
presenta un identificador y su distancia mínima de Hamming `d` respecto de los
32 originales, tras binarizar con `>= 0.5`. `d=0` identifica una copia;
`d>0` identifica un candidato nuevo. El JSON conserva coordenadas completas,
salidas continuas y binarias, configuración, rango y SHA256 del modelo y dataset.
Si ya existe `candidates.json`, el comando rechaza la generación para conservar
las decisiones visuales previas. Usar un directorio nuevo con `--output` para
regenerar las láminas o explorar otra configuración.
Para regenerar un candidato: cargar ese modelo con `Autoencoder.load`, pasar
sus coordenadas como matriz `(1,2)` a `decode` y aplicar el mismo umbral.

La novedad no implica una letra reconocible. El resultado inicial conserva
`review.status = "pending"`. Después de que el estudiante inspeccione las
láminas, registrar su elección explícita (reemplazar el identificador y nota):

```powershell
python -m src.explore --model results/pilot/model.npz --review results/candidates/candidates.json --candidate 12 --note "El estudiante identifica este candidato como letra tras revisar la lámina"
```

Si no hay una letra adecuada, omitir `--candidate` y explicar el resultado con
`--note`. Se registra `no_suitable_letter` y la demostración permanece pendiente;
otra exploración requiere revisar el resultado y acordar la continuación.
El registro de una selección verifica el hash del modelo, la regeneración y
la novedad, y conserva el identificador, coordenadas, imagen y nota visual.
La selección solamente vale para ese modelo. Si cambia el modelo final,
regenerar las láminas y repetir la revisión con el estudiante.
Después de trasladar los artefactos, `--dataset ruta/font.h` permite indicar
el original disponible para la revisión; se verifica su SHA256 guardado en el
registro, sin depender de la ruta absoluta del entorno anterior.

### Comparación de rho de RMSProp

`configs/comparison-rmsprop-rho.json` compara rho=0.85 y 0.95, con tasa fija
0.002 y semillas 42/43/44 desde cero. Mantiene el resto de la configuración
ganadora: máximo 3000 épocas por corrida, evaluación cada 100 y parada al
cumplir los 32 patrones. Son seis corridas nuevas, hasta 18000 épocas; las
tres referencias originales de rho=0.9 se reutilizan sin entrenamiento.

```bash
python -m src.compare --protocol configs/comparison-rmsprop-rho.json --output results/comparison-rmsprop-rho-reproduction
```

La salida debe estar vacía y los artefactos originales de
`results/comparison-broad/` deben estar disponibles. Repetir el comando consume
un nuevo presupuesto. Las variables BLAS/OMP se mantienen sin fijar, como en
los manifiestos originales efectivos; no comparar velocidades. Los resultados
y las verificaciones se documentan en [comparison.md](comparison.md), sin
reemplazar la selección vigente de `results/final/selection.json`.

### Comparación local de tasas RMSProp

`configs/comparison-rmsprop-local.json` prueba tasas fijas 0.0015 y 0.0025
desde cero con semillas 42/43/44. Mantiene la arquitectura deeper y el resto
del protocolo factorial: lote 32, máximo 3000 épocas, evaluación cada 100 y
parada al cumplir la tolerancia. Son seis entrenamientos nuevos, hasta 18000
épocas; las tres corridas originales de 0.002 se reutilizan sin entrenarlas.

```bash
python -m src.compare --protocol configs/comparison-rmsprop-local.json --output results/comparison-rmsprop-local-reproduction
```

Se necesitan los artefactos originales de `results/comparison-broad/` y una
salida vacía. La ejecución documentada mantuvo las variables BLAS/OMP sin fijar,
como constan en los manifiestos originales; las duraciones no sustentan una
comparación de velocidad. Repetir este comando consume un nuevo presupuesto.
La selección de esta comparación queda en su propio resumen y no reemplaza
`results/final/selection.json`. Resultados y límites en
[comparison.md](comparison.md).

### Continuación controlada de RMSProp

El protocolo aprobado del 2026-11-10 está en
`configs/continuation-rmsprop.json`. Compara las semillas 42 y 43 desde sus
modelos archivados a época 3000, conservando pesos, momentos RMSProp, RNG e
historia: tasa 0.002 contra 0.001, hasta 1000 épocas adicionales por rama.
El límite compartido es 4000 épocas adicionales y 600 segundos; se comprueba
el tiempo antes de cada época y se completa el guardado de evidencia al cerrar.
No ejecuta otras campañas ni cambia la selección final.

```bash
OPENBLAS_NUM_THREADS=1 python -m src.continue_training --protocol configs/continuation-rmsprop.json --output results/continuation-rmsprop-reproduction
```

Usar una salida vacía y disponer de los artefactos originales de
`results/comparison-broad/` (ignorados por Git). El comando valida su hash,
dataset, métricas al recargar y estado persistido antes de entrenar; guarda
protocolo, preflight, historias, modelos, métricas y resumen en la salida nueva.
Las épocas de las historias son acumuladas; `additional_epochs_completed`
identifica el consumo de este experimento. Repetir el comando implica un nuevo
presupuesto de entrenamiento. Los resultados ejecutados y sus límites se
documentan en [comparison.md](comparison.md).

El espacio latente de este autoencoder no tiene estructura generativa
garantizada: las interpolaciones o regiones intermedias pueden producir
imágenes sin interpretación. Ni la distancia ni la apariencia de un candidato
demuestran generalización. Las pruebas comprueban regeneración y novedad,
incluidas una copia y una imagen distinta, sin clasificar automáticamente letras.

## Ejercicio 1.b: preparar la referencia de denoising

El paso B1 del [plan](assignment_plan.md#b1-referencia-a-la-ganadora-y-configuración-de-campaña)
expone la preparación de campaña y sus colecciones fijas, sin entrenar:

```bash
python -m src.denoise prepare --config configs/denoising.json --output results/denoising
```

La salida debe ser un directorio nuevo. `source_selection` se resuelve desde
TP5. Las rutas de modelo y configuración de la selección se resuelven desde
`paths_relative_to`, relativo al TP5 de origen; si falta, se usa `.` por
compatibilidad con las selecciones existentes. Se requiere `manifest.json`
junto a la configuración fuente para cotejar el dataset, modelo y configuración
efectiva. No se usan las rutas absolutas históricas del manifiesto.

`reference/` conserva copias de selección, configuración, manifiesto, modelo y
`font.h`; `manifest.json` registra sus hashes y rutas relativas a la campaña.
Las variantes `winner` y `latent-10` derivan sus capas, optimizador, tasa y pérdida
de esa referencia, cambiando solo el ancho central en la segunda. Los DAE se
entrenarán desde cero en pasos posteriores. La fuente debe usar latente tanh,
salida logística, tasa constante y carecer de regularización adicional.
Cambiar la selección solo afecta preparaciones nuevas. Una evaluación posterior
debe usar las copias de la campaña.

Entrenamiento y evaluación numérica de 1.b:

```bash
python -m src.denoise collections --campaign results/denoising
python -m src.denoise train --campaign results/denoising
python -m src.denoise evaluate --campaign results/denoising
```

`collections` permite completar campañas preparadas antes de B2. `train` ejecuta
una sola vez seis corridas de 3000 épocas. `evaluate` carga los mejores modelos
sin entrenar y produce NPZ/JSON/CSV. No genera gráficos ni informe por pedido
del usuario. Los comandos, esquemas y recetas para generar figuras después
están en [denoising-data.md](denoising-data.md).

## Ejercicio 2: Autoencoder Variacional (VAE)

Los artefactos se generan en `results/vae/` (ignorado por Git, como el resto de
`results/`). Protocolo: [configs/vae.json](configs/vae.json).

```bash
python -m src.vae_train --config configs/vae.json --output results/vae
python -m src.vae_explore --campaign results/vae --output results/vae/figures
python -m unittest tests.test_vae -v
```

`vae_train` saltea las corridas terminadas, así que se puede repetir tras una
interrupción (`--max-runs N` y `--seed S` permiten entrenar por tandas).
`vae_explore` no entrena: carga los modelos guardados y produce figuras y tabla.

### 2.a Dataset nuevo: dígitos del TP3

Imágenes 28×28 en escala de grises, píxeles en [0,1], leídas con el cargador de
`tp3/shared/digit_dataset.py` sin modificar los CSV ([src/digits.py](src/digits.py)).
Se entrenó con dos archivos para comparar:

- `digits.csv`: 12.449 imágenes, **sin ningún 8** y con 271 cincos.
- `more_digits.csv`: 15.741 imágenes, con 585 ochos y 542 cincos.

`digits_test.csv` (2.497 imágenes, los diez dígitos, sin imágenes repetidas en
los dos archivos anteriores) se usa solo para medir y graficar; no interviene
en el entrenamiento ni en ninguna selección.

### 2.b Esquema variacional

[src/vae.py](src/vae.py) reutiliza de `tp3/shared` las activaciones, la regla de
inicialización Xavier y el optimizador Adam. El forward y la retropropagación
son propios porque el MLP del TP3 no tiene nodo estocástico; TP3 no se modificó.

- Encoder `784→256→64` (tanh) con dos salidas lineales de 2 unidades: `μ(x)` y
  `log σ²(x)`. Decoder espejo `2→64→256→784`, salida logística.
- Reparametrización: `z = μ + σ·ε`, `ε ~ N(0,I)`, con `σ = exp(log σ² / 2)`.
- Costo `J = L + λ·R`: `L` es la BCE sumada sobre los 784 píxeles (la pérdida
  ganadora de 1.a) y `R = ½ Σ (exp(log σ²) + μ² − 1 − log σ²)` es la KL contra
  `N(0,I)`; ambos promediados sobre el lote.
- Gradientes: `∂L/∂z` llega desde el decoder; `∂z/∂μ = 1`, `∂z/∂log σ² = ½σε`;
  `∂R/∂μ = μ`, `∂R/∂log σ² = ½(exp(log σ²) − 1)`. `tests/test_vae.py` los
  compara con diferencias finitas en todos los parámetros.
- Entrenamiento: Adam 0.001, lote 64, 50 épocas, semillas 42/43/44 y
  `λ ∈ {0, 0.1, 1}` sobre cada archivo: 18 corridas. Con `λ = 0` no hay
  regularizador y la red se comporta como el autoencoder común (σ tiende a 0).

Resultados sobre `digits_test.csv`, media ± desvío entre las 3 semillas
(reconstrucción determinística por el código medio):

| Entrenamiento | λ | BCE test | KL test | desvío de μ | σ media | BCE del 8 | BCE resto |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| digits | 0 | 151.11 ± 0.90 | 35.84 ± 1.95 | 3.54 ± 0.18 | 0.000 | 246.47 ± 4.16 | 140.83 ± 0.55 |
| digits | 0.1 | 150.44 ± 0.30 | 10.35 ± 0.16 | 1.55 ± 0.04 | 0.017 | 243.17 ± 3.64 | 140.44 ± 0.22 |
| digits | 1 | 149.74 ± 0.52 | 6.48 ± 0.06 | 1.07 ± 0.01 | 0.047 | 235.67 ± 2.88 | 140.47 ± 0.48 |
| more_digits | 0 | 143.04 ± 0.27 | 36.16 ± 2.51 | 3.45 ± 0.30 | 0.000 | 190.65 ± 1.52 | 137.91 ± 0.16 |
| more_digits | 0.1 | 143.20 ± 0.75 | 10.41 ± 0.20 | 1.56 ± 0.07 | 0.017 | 191.11 ± 4.01 | 138.03 ± 0.57 |
| more_digits | 1 | 142.29 ± 0.94 | 6.56 ± 0.11 | 1.10 ± 0.02 | 0.048 | 183.31 ± 1.91 | 137.87 ± 0.89 |

Lectura:

- **El término KL ordena el latente sin costo de reconstrucción.** La BCE de
  test es prácticamente igual para los tres λ, pero el desvío de los códigos
  baja de ~3.5 (λ=0) a ~1.1 (λ=1), es decir, a la escala del prior `N(0,I)`.
  En `latent-*.png`, con λ=0 los códigos se dispersan hasta |z|≈10 y casi todos
  quedan fuera del círculo del 95 % del prior; con λ=1 quedan mayormente dentro.
- **Consecuencia generativa.** Al decodificar los mismos 16 `z ~ N(0,I)`
  (`samples-*.png`), con λ=0 salen casi solo doses y treses (el prior cae en
  una esquina del latente) y algún parche sin forma; con λ=1 aparecen dígitos
  variados. La grilla `grid-*.png` muestra lo mismo sobre `[-2.5, 2.5]²`.
- **El 8.** Entrenando con `digits.csv` la BCE del 8 en test es ~236–246
  contra ~140 del resto, y en la grilla no aparece ningún 8: el modelo no puede
  generar lo que nunca vio. Con los 585 ochos de `more_digits.csv` baja a
  ~183–191 y con λ=1 aparece una región de ochos en la grilla; sigue siendo el
  dígito peor reconstruido, coherente con su baja frecuencia.
- **Límites.** Con solo 2 dimensiones las imágenes salen borrosas y dígitos
  parecidos se confunden (4/9, 3/5/8); `reconstructions.png` usa el primer
  ejemplo de cada dígito de test, sin elegir casos favorables; algunos son
  atípicos (el primer 1 lleva sombrero y base, está en el percentil 99 de BCE
  entre los unos y se reconstruye como 3, aunque el 1 es el dígito mejor
  reconstruido). `reconstructions-extremes-*.png` muestra, por dígito y para
  λ=1, el ejemplo de test mejor, mediano y peor reconstrucción según su BCE. La brecha entre
  BCE de entrenamiento (~126–130) y de test (~142–151) indica sobreajuste
  parcial. No se buscó arquitectura ni tasa: es un único protocolo.

### Dimensión del latente: por qué también 16D

El latente 2D se eligió para poder graficar el espacio latente, pero limita la
reconstrucción. [src/vae_latent.py](src/vae_latent.py) repite el mismo protocolo
(`more_digits.csv`, λ=1, 50 épocas) cambiando solo el ancho del código; los
artefactos quedan en `results/vae-latent/`.

```bash
python -m src.vae_latent train --latent 16 --seed 42   # una corrida por llamada
python -m src.vae_latent report
```

Sobre `digits_test.csv` (media ± desvío si hay 3 semillas; 4, 8 y 32 tienen una
sola semilla, 42, y sirven de orientación):

| Latente | Semillas | BCE test | KL test | BCE + KL | MSE/píxel | Reconocibles | Dimensiones activas |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 3 | 142.3 ± 0.9 | 6.6 ± 0.1 | 148.9 | 0.0396 | 71.5 % | 2, 2, 2 de 2 |
| 4 | 1 | 119.5 | 10.3 | 129.8 | 0.0299 | 80.1 % | 4 de 4 |
| 8 | 1 | 93.8 | 16.7 | 110.5 | 0.0188 | 82.8 % | 8 de 8 |
| 16 | 3 | 80.9 ± 1.0 | 21.6 ± 0.7 | 102.4 | 0.0134 | 83.5 % | 15, 15, 13 de 16 |
| 32 | 1 | 78.4 | 22.6 | 101.0 | 0.0124 | 83.5 % | 16 de 32 |

"Reconocibles": fracción de reconstrucciones cuyo dígito promedio de
entrenamiento más cercano coincide con la etiqueta (las imágenes originales
obtienen 83.1 % con ese mismo clasificador simple). "Dimensiones activas":
coordenadas del código con KL media mayor que 0.01 nats; las demás quedaron
iguales al prior y no transportan información.

- La mejora se agota en 16: de 2 a 16 el costo total (BCE + KL) baja 46 nats;
  de 16 a 32, 1.4. Las reconstrucciones 16D son tan reconocibles como las
  originales según el clasificador por centroides.
- El modelo no usa más: con 32 disponibles activa 16 y con 16 activa 13–15.
  El término KL apaga las coordenadas sobrantes.
- Es el mismo código y las mismas fórmulas que el 2D, lo que descarta que los
  errores de reconstrucción del 2D vengan de la implementación
  (`reconstructions-2d-vs-16d.png`, mismos ejemplos de test).
- Al generar desde `z ~ N(0,I)` (`samples-by-width.png`, `generated-16d.png`)
  las muestras 16D tienen trazo más nítido, pero aparecen más formas
  incompletas o mezcladas que en 2D, donde cada muestra es borrosa pero con
  forma de dígito. El 16D mejora la reconstrucción, no claramente la generación.

### 2.c Generación de muestras nuevas

`generated-*.png` muestra, para el modelo λ=1 (semilla 42), diez muestras
decodificadas desde `z ~ N(0,I)` (generador con semilla fija 2027) y, debajo
de cada una, la imagen de entrenamiento más cercana con su distancia RMS por
píxel (0.05–0.17): son imágenes con forma de dígito que no copian un ejemplo
del dataset. `interpolation-*.png` recorre el segmento entre los códigos de dos
dígitos de test. El juicio de pertenencia es visual y queda a cargo del grupo;
códigos, índices del vecino más cercano y distancias están en
`results/vae/figures/summary.json`.

Otras figuras: `dataset.png` (muestra y conteos de ambos archivos) y
`curves.png` (BCE y KL de test por época, media y rango de las 3 semillas).
Los modelos guardan pesos y configuración para evaluar y generar; no guardan
el estado de Adam ni del generador aleatorio, así que no reanudan un
entrenamiento de forma idéntica.
