# TP5: autoencoder básico de caracteres

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

El espacio latente de este autoencoder no tiene estructura generativa
garantizada: las interpolaciones o regiones intermedias pueden producir
imágenes sin interpretación. Ni la distancia ni la apariencia de un candidato
demuestran generalización. Las pruebas comprueban regeneración y novedad,
incluidas una copia y una imagen distinta, sin clasificar automáticamente letras.
