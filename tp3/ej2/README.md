# Ejercicio 2 — Datos de dígitos y MLP

Desde la raíz del repositorio:

```bash
python -m tps_sia.tp3.ej2.src.datos_digitos
```

La primera ejecución parsea `digits.csv` y genera `cache/digits.npz` dentro
de `ej2/`. Las siguientes cargan el caché; se regenera si cambia la ruta,
el tamaño o la fecha de modificación del CSV, o la versión del formato.
Si se carga otro CSV, recibe un caché propio para evitar reemplazar el de
entrenamiento.

Se conserva la lógica de `data/digit_dataset_loader.py`:
deserializar `image` con `ast.literal_eval` y convertir a `float32`, sin
reescalar los píxeles. `X` tiene forma `(N, 784)` e `y` usa one-hot de
forma `(N, 10)`, también `float32`: la columna `d` corresponde al dígito `d`.

Para usar los defaults del ejercicio desde otros módulos:

```python
from tps_sia.tp3.ej2.src.datos_digitos import cargar, particionar

X, y = cargar()
X_train, y_train, X_val, y_val = particionar(X, y)
```

La partición reserva el 20 % de cada clase para validación con semilla 42.
Los conteos se redondean al entero más próximo y ambas partes conservan
al menos una muestra de cada clase presente. Se mezclan las filas de cada
partición para evitar bloques ordenados por dígito. El módulo imprime las
distribuciones total, de entrenamiento y de validación.

`digits_test.csv` queda reservado para la evaluación final; este comando
no lo lee ni genera su caché.

## Chequeos del paso 1

Desde la raíz del repositorio:

```bash
python -m tps_sia.tp3.ej2.tests.test_datos_digitos
```

Los chequeos verifican formas, tipos, rango de píxeles, one-hot de diez
salidas, creación e invalidación del caché, cargas sin reparseo y partición
completa, disjunta, estratificada y reproducible. A diferencia del comando
de exploración, esta verificación lee también `digits_test.csv` para
comprobar su formato y buscar imágenes idénticas entre ambos archivos.
No entrena modelos ni usa el test para seleccionar hiperparámetros.

## Exploración de `digits.csv`

| Dígito | Total | Entrenamiento | Validación |
|---|---:|---:|---:|
| 0 | 1480 | 1184 | 296 |
| 1 | 1685 | 1348 | 337 |
| 2 | 1489 | 1191 | 298 |
| 3 | 1532 | 1226 | 306 |
| 4 | 1460 | 1168 | 292 |
| 5 | 271 | 217 | 54 |
| 6 | 1479 | 1183 | 296 |
| 7 | 1566 | 1253 | 313 |
| 8 | 0 | 0 | 0 |
| 9 | 1487 | 1190 | 297 |
| **Total** | **12449** | **9960** | **2489** |

El 8 está ausente y el 5 tiene sólo 271 ejemplos, frente a los 1460–1685
de las demás clases presentes. Se mantienen diez salidas para representar
todos los dígitos, incluido el 8 en la evaluación final. Estos conteos se
obtuvieron exclusivamente de `digits.csv`.

## Paso 2 — MLP y SGD

`shared/mlp.py` implementa `MLP` con arquitectura como lista, forward y
backprop matriciales. Reutiliza tanh, ReLU y logística de
`shared/activations.py`. `ej2/src/mlp.py` reexporta la misma clase. La salida por defecto es softmax con
cross-entropy, calculada mediante log-softmax estable; su delta es
`(p - y) / N`. Para una sola salida se debe elegir `salida="logistica"`.

Ejemplo de configuración para el próximo paso de entrenamiento:

```python
from tps_sia.tp3.shared.mlp import MLP

modelo = MLP([784, 128, 10], activacion="relu", eta=0.01,
             tamano_lote=32, inicializacion="auto", semilla=42)
# Con los conjuntos de entrenamiento/validación del paso 1:
# historia = modelo.entrenar(X_train, y_train, epocas=100,
#                           X_val=X_val, y_val=y_val, verbose=10)
# modelo.guardar("modelo.npz")
# modelo = MLP.cargar("modelo.npz")
```

- `predecir`/`forward` devuelve `(N, n_salidas)` y `predecir_clases`, las clases.
- `backprop` devuelve los gradientes sin modificar el modelo: primero los
  pesos de todas las capas y luego sus biases, igual que `parametros`.
- `tamano_lote=1` selecciona online; `None`, batch; un entero positivo,
  mini-batch. Se procesa también el último lote incompleto y cada
  gradiente se promedia por el número real de muestras de ese lote.
- `inicializacion="auto"` usa He normal en ocultas ReLU y Xavier normal
  en las demás capas. Se puede elegir `"xavier"` o `"he"` explícitamente.
  Los biases empiezan en cero. `beta=1` por defecto: tanh aplica
  `tanh(beta*h)` y logística, `sigmoid(2*beta*h)`, como el módulo existente.
- `salida="logistica"` usa costo `0.5 * mean(sum((p-y)**2, axis=1))`.
  La métrica `mse` promedia todos los elementos, así que para K salidas
  ese costo es `K * mse / 2`. Los objetivos deben estar en `[0,1]`;
  softmax requiere además que cada fila sume 1 (one-hot o distribución).
- La `Historia` registra costo, MSE, accuracy, norma media del gradiente
  de los lotes, métricas de validación, épocas y tiempo. `to_dict()`
  permite exportarla a JSON. Accuracy usa umbral 0.5 para una salida y
  argmax para varias, con etiquetas de clasificación.
- `epsilon` detiene por costo de entrenamiento. La validación no modifica
  pesos. Cada llamada a `entrenar` continúa los parámetros y genera una
  nueva historia, accesible también como `modelo.historia`.
- Guardar/cargar conserva parámetros, configuración, optimizador, última historia
  y estado aleatorio para reanudar con el mismo orden de mini-batches.
  El archivo `.npz` se carga con `allow_pickle=False`.

`shared/optimizers.py` separa SGD del MLP; `src/optimizadores.py` conserva
la importación anterior. La interfaz `Optimizador` expone
`paso(parametros, gradientes)` (actualización in-place) y `configuracion()`.
Se puede inyectar `optimizador=SGD(eta=...)`, `Momentum(...)` o `Adam(...)`;
ver la configuración y convenciones del paso 4 más abajo.

## Validación del paso 2

Desde la raíz del repositorio, sin leer ni entrenar con dígitos:

```bash
python -m tps_sia.tp3.ej2.tests.test_validacion
```

Comprueba todos los pesos y biases contra diferencias centradas, en redes
con una y dos capas ocultas, tanh/ReLU y ambas salidas. Para ReLU evita
los puntos no diferenciables. También verifica softmax con logits extremos,
SGD, lotes incompletos, historia, aislamiento de validación, inicialización
y reanudación idéntica tras guardar/cargar.

XOR usa las entradas del enunciado y convierte sus objetivos de −1/+1
a 0/1 para logística. Con semilla 0, tanh, beta 1, SGD con eta 0.3 y
lotes de 2, ambas arquitecturas `[2,2,1]` y `[2,3,2,1]` logran los cuatro
aciertos y costo menor que 0.001. Se comprueba además XOR con dos
salidas softmax.

## Paso 3 — Baseline

Desde la raíz del repositorio (dependencias: NumPy y Matplotlib):

```bash
python -m tps_sia.tp3.ej2.src.baseline
```

Entrena un único MLP `[784, 128, 10]`, ReLU en la capa oculta y softmax
con cross-entropy en la salida. Usa SGD con tasa 0.01, mini-batches de 32,
inicialización automática He/Xavier y 30 épocas. `baseline.json` conserva
la configuración; las semillas del modelo y de la partición son 42.
Los píxeles ya están en `[0, 1]` y se usan sin reescalado.

Se mantiene la partición estratificada 80/20: 9960 muestras de train y
2489 de validación. Sólo train actualiza los pesos; las métricas de ambos
conjuntos se calculan al terminar cada época con los mismos parámetros.
El resultado corresponde a la última época, sin selección de checkpoints
ni parada por validación. No se carga `digits_test.csv`.

El comando guarda en `ej2/results/baseline/`:

- `results.json`: configuración, SHA-256 del CSV, entorno, distribución
  de clases, historia completa, métricas finales y resultado del sanity check.
- `history.csv`: loss y accuracy de train y validación por época.
- `model.npz`: modelo final, historia y estado para reanudar.
- `learning_curves.png`: curvas de loss y accuracy de train vs. validación.
- `confusion_matrix.csv` y `confusion_matrix.png`: conteos en validación;
  filas = dígito real, columnas = predicción, siempre con las diez clases.
  La fila del 8 se marca sin muestras: no permite evaluar su reconocimiento.

La ejecución informa `OK` si la accuracy final de validación alcanza el
90 % y `REVISAR` si queda por debajo. Ese umbral es un chequeo orientativo
del pipeline, no una garantía de desempeño para cada clase o en producción.
El desbalance del 5 y la ausencia del 8 deben considerarse al interpretar
la accuracy global.

Ejecución de referencia con la configuración incluida:

| Métrica final (época 30) | Train | Validación |
|---|---:|---:|
| Cross-entropy | 0.15395 | 0.19498 |
| Accuracy | 95.97 % | 94.38 % |

Supera el sanity check del 90 %. El modelo recargado reproduce la matriz
de confusión y la loss de validación guardadas; los gráficos se regeneran
desde `results.json`. La clase 5 obtiene 44 aciertos de 54 muestras de
validación (81.48 %), por debajo de la accuracy global.

El entrenamiento y el análisis están separados: los gráficos se pueden
regenerar a partir del JSON sin volver a entrenar:

```bash
python -m tps_sia.tp3.ej2.src.plots
```

Para próximas ejecuciones se puede pasar `--config ruta.json` y
`--output-dir ruta` al baseline, y `--results ruta/results.json` a los
gráficos. Cada ejecución empieza un modelo nuevo; usar un directorio
distinto permite conservar cada experimento. Las salidas generadas se
excluyen de Git.

Si OpenBLAS usa demasiados hilos para estos mini-batches, se puede ejecutar
en PowerShell con un hilo (el ajuste afecta al proceso y sus hijos):

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
python -m tps_sia.tp3.ej2.src.baseline
```


## Paso 3a — Núcleo compartido

Las implementaciones reutilizables residen en `tps_sia/tp3/shared/`:
`activations.py`, `mlp.py`, `optimizers.py` y `digit_dataset.py`.
El paquete no importa módulos de los ejercicios. Ej1 consume las activaciones;
ej2 consume el MLP, SGD y loader. Los módulos anteriores de activaciones,
MLP y optimizadores sólo reexportan las mismas clases y funciones; se conservan
sus APIs en español y la lectura de modelos y cachés versión 1.
Desde el paso 4 se guardan modelos versión 2 con estado del optimizador;
el formato del caché no cambia.

El loader común requiere ambas rutas explícitas. Por ejemplo, desde la raíz:

```python
from pathlib import Path
from tps_sia.tp3.shared.digit_dataset import cargar, particionar
from tps_sia.tp3.shared.mlp import MLP
from tps_sia.tp3.shared.optimizers import SGD

exercise = Path("tps_sia/tp3/ej2").resolve()
X, y = cargar(exercise.parent / "data" / "digits.csv",
              exercise / "cache" / "digits.npz")
X_train, y_train, X_val, y_val = particionar(X, y)
modelo = MLP([784, 128, 10], activacion="relu", optimizador=SGD(0.01),
             tamano_lote=32, semilla=42)
```

Cada consumidor elige su dataset y caché. El wrapper
`ej2/src/datos_digitos.py` mantiene `cargar()` sin argumentos, los cachés
separados para otros CSV y el comando de exploración. La invocación histórica
`python tps_sia/tp3/ej2/src/datos_digitos.py` delega al comando de paquete.
Los defaults del baseline siguen apuntando a `ej2/baseline.json`,
`ej2/cache/` y `ej2/results/baseline/`, resueltos desde el módulo.

Chequeos comunes con datos sintéticos, desde la raíz:

```bash
python -m tps_sia.tp3.shared.tests.test_activations
python -m tps_sia.tp3.shared.tests.test_mlp
python -m tps_sia.tp3.shared.tests.test_digit_dataset
python -m tps_sia.tp3.ej2.tests.test_shared_compatibility
```

`ej2.tests.test_validacion` delega a los chequeos comunes del MLP. Los
chequeos propios de los CSV reales permanecen en
`ej2.tests.test_datos_digitos`; los del perceptrón simple permanecen en
`ej1.tests.test_validacion`, que también reutiliza el chequeo de derivadas.

La extracción se verificó cargando el modelo SGD de referencia, sin
reentrenarlo: las salidas de las 2489 muestras de validación coinciden
exactamente con las previas al cambio, al igual que las particiones.
Se reprodujeron la loss y la matriz de confusión guardadas; la configuración
y los archivos de referencia conservaron sus SHA-256.

## Paso 4 — Momentum y Adam

`shared/optimizers.py` ofrece `SGD`, `Momentum` y `Adam`, también
reexportados desde `ej2/src/optimizadores.py`. Todos reciben los gradientes
calculados por `MLP.backprop` y actualizan cada tensor in-place, en el orden
de `MLP.parametros`: todos los pesos y luego todos los biases. El optimizador
no vuelve a sumar ni promediar los gradientes; el MLP ya los promedia por
la cantidad efectiva de muestras del lote.

Convenciones de esta implementación, con `g_t = ∂cost/∂parameter`:

- SGD: `parameter -= eta * g_t`.
- Momentum clásico, con velocidad inicial cero:
  `v_t = momentum * v_(t-1) + g_t`,
  `parameter -= learning_rate * v_t`. La velocidad acumula gradientes;
  el signo negativo y la tasa se aplican al actualizar el parámetro.
- Adam, con momentos iniciales cero:
  `m_t = beta1*m_(t-1) + (1-beta1)*g_t`,
  `v_t = beta2*v_(t-1) + (1-beta2)*g_t**2`;
  `m_hat = m_t/(1-beta1**t)`, `v_hat = v_t/(1-beta2**t)`;
  `parameter -= learning_rate*m_hat/(sqrt(v_hat)+optimizer_epsilon)`.
  `t` cuenta actualizaciones completas de lotes, una vez por llamada a
  `paso`, independientemente de cuántos pesos y biases tenga la red.

Momentum usa por defecto `learning_rate=0.01`, `momentum=0.9`.
Adam usa `learning_rate=0.001`, `beta1=0.9`, `beta2=0.999` y
`optimizer_epsilon=1e-8`. Este último estabiliza la división, fuera de
la raíz; es distinto de `epsilon` de `MLP.entrenar`, que detiene por costo
de entrenamiento. Las tasas y epsilon del optimizador deben ser positivos
y finitos; momentum, beta1 y beta2 deben pertenecer a `[0,1)`.
Estos valores son defaults de implementación, no una selección experimental.

Desde la raíz:

```python
from tps_sia.tp3.shared.mlp import MLP
from tps_sia.tp3.shared.optimizers import Adam, Momentum, construir_optimizador

optimizer = construir_optimizador({
    "name": "adam", "learning_rate": 0.001,
    "beta1": 0.9, "beta2": 0.999, "optimizer_epsilon": 1e-8,
})
# Equivalente: Adam(learning_rate=0.001).
# Alternativa: Momentum(learning_rate=0.01, momentum=0.9).
modelo = MLP([784, 128, 10], activacion="relu",
             optimizador=optimizer, tamano_lote=32, semilla=42)
```

La fábrica acepta claves nuevas en inglés (`name`, `learning_rate`, etc.)
y la configuración histórica de SGD `{"nombre": "sgd", "eta": 0.01}`.
Rechaza nombres, campos e hiperparámetros inválidos. Se validan todas las
formas, tipos y valores finitos de parámetros, gradientes y estado antes de
aplicar un paso; también se comprueban los resultados numéricos preparados.
Una entrada inválida o un overflow deja intactos los parámetros, momentos
y contador del optimizador.

`export_state()` devuelve un mapping con `version`, `config`, `updates`
y copias de los tensores: `velocity` para momentum, `first_moment` y
`second_moment` para Adam, sin tensores para SGD. Antes del primer paso,
las listas están vacías. Para restaurar sobre parámetros ya recuperados:

```python
state = optimizer.export_state()
restored = construir_optimizador(state["config"])
restored.restore_state(state, modelo.parametros)
```

El estado no incluye los pesos: se debe conservar junto a los parámetros
del mismo instante, con su orden y formas. La restauración valida versión,
configuración, contador y todos los tensores antes de cambiar el optimizador.
Los momentos de segundo orden deben ser no negativos. Las copias evitan
que modificar el estado exportado o recibido altere el optimizador.

`MLP.guardar` integra este estado en el modelo NPZ versión 2, sin pickle:
metadata JSON y arrays separados para velocidades/momentos. `MLP.cargar`
restaura parámetros, optimizador, RNG y última historia. Continúa leyendo
modelos SGD versión 1; como éstos no registraban un contador de updates,
ese contador empieza en cero al cargarlos. Guardar/cargar entre dos bloques
de entrenamiento produce exactamente los mismos parámetros que una corrida
continua con la misma semilla y configuración para los tres optimizadores.
El paso 5 amplía este formato a versión 3 con historia acumulativa y control
completo del entrenamiento, como se describe a continuación.

Chequeos del paso 4, desde la raíz:

```bash
python -m tps_sia.tp3.shared.tests.test_optimizers
python -m tps_sia.tp3.ej2.tests.test_validacion
python -m tps_sia.tp3.ej2.tests.test_shared_compatibility
```

Las 13 pruebas nuevas cubren los primeros pasos calculados a mano, todos los
pesos y biases de un MLP, configuración/fábrica, float32, gradientes cero,
rechazo atómico de entradas y estados inválidos, overflow, copias de estado,
reanudación exacta, formato SGD anterior, checkpoints corruptos y XOR con
momentum/Adam en `[2,2,1]` y `[2,3,2,1]`. Los chequeos existentes de gradientes
y XOR con SGD siguen pasando. El baseline continúa usando SGD; este paso
no ejecuta barridos ni selecciona hiperparámetros.


## Paso 5 — Control del entrenamiento y checkpoints

`MLP.entrenar` continúa desde el último estado y devuelve la **historia acumulada**.
`epocas` indica cuántas épocas adicionales ejecutar. `epochs_completed` y
`updates_completed` cuentan épocas terminadas y actualizaciones de lotes.
`Historia` registra índices globales (`epochs`, `updates`), learning rate,
costo, MSE, accuracy y norma media del gradiente. `validation_epochs` permite
ubicar los puntos de validación si se cambia su disponibilidad entre llamadas.
La tasa sigue siendo constante; no se añadió un schedule sin evidencia experimental.

La parada informa `max_epochs`, `training_epsilon`, `early_stopping`,
`interrupted` o `numerical_failure`. `epsilon > 0` detiene cuando el costo de
train es estrictamente menor a epsilon; cero desactiva esa condición.
La parada por validación es opcional (`patience=None` por defecto):
`monitor="validation_loss"` minimiza el costo de validación, que es cross-entropy
con softmax; `validation_accuracy` maximiza accuracy. `patience` cuenta épocas
sin una mejora estrictamente mayor a `min_delta` respecto de la última mejora
significativa. La selección conserva siempre la mejor métrica observada,
aunque su mejora sea menor a `min_delta`; los empates conservan la primera época.
Ni evaluar validación ni seleccionar la mejor época modifica pesos o RNG del
modelo que está entrenándose.

Ejemplo con arrays de train/validación preparados desde datos de desarrollo:

```python
from pathlib import Path
from tps_sia.tp3.shared.mlp import MLP
from tps_sia.tp3.shared.optimizers import Adam

run_dir = Path("tps_sia/tp3/ej2/results/example")
model = MLP([784, 128, 10], activacion="relu", optimizador=Adam(0.001),
            tamano_lote=32, semilla=42)
model.preprocessing = {"name": "identity", "pixel_range": [0.0, 1.0]}
model.entrenar(
    X_train, y_train, epocas=30, X_val=X_val, y_val=y_val,
    patience=5, min_delta=1e-4,
    checkpoint_path=run_dir / "checkpoint.npz", checkpoint_every=2,
    best_model_path=run_dir / "best_model.npz", pause_after=10,
    weight_log_path=run_dir / "weights.jsonl", log_every_updates=20,
    selected_weights=[(0, 0, 0), (1, 0, 0)],
)

continued = MLP.cargar(run_dir / "checkpoint.npz")
continued.entrenar(
    X_train, y_train, epocas=20, X_val=X_val, y_val=y_val,
    checkpoint_path=run_dir / "checkpoint.npz",
    best_model_path=run_dir / "best_model.npz",
)
best = MLP.cargar(run_dir / "best_model.npz")
probabilities = best.predecir(X_val)
```

`checkpoint.npz` guarda el **último estado** para continuar; `best_model.npz`
guarda la **época elegida por validación**, para inferencia/evaluación.
El modelo en memoria permanece en la última época. `guardar_best(path)`
permite exportar posteriormente la selección conservada en un checkpoint.
El archivo de la mejor época también incluye su optimizador, RNG e historia
correspondientes, por lo que puede continuar desde ese instante sin combinar
sus pesos con momentos de otra época. Los dos artefactos deben usar rutas diferentes.

El formato NPZ versión 3 conserva arquitectura, activaciones/beta, loss y
reducción del gradiente, estrategia/lote, shuffle, epsilon, criterio de parada,
semilla y método/escala de inicialización por capa, pesos/biases, estado del
optimizador, RNG, historia/tiempos, selección/paciencia, preprocesamiento y
configuración del registro. `training_calls` registra presupuestos adicionales,
época inicial, pausa y frecuencia/rutas de guardado. Al continuar, las opciones
omitidas de epsilon, shuffle, monitor, patience, min_delta y frecuencia de
registro conservan sus valores anteriores. Es necesario volver a proporcionar
los arrays de validación para mantener la misma selección. Cambiar la política
de entrenamiento reinicia el seguimiento de la mejor época y la paciencia,
conservando parámetros, optimizador y métricas acumuladas.

`preprocessing` describe la transformación aplicada **antes** de entregar los
arrays al MLP: guardarla no transforma automáticamente los datos. Cualquier
estadística adicional debe ajustarse sólo sobre train. El baseline conserva
sus píxeles originales en `[0,1]`, sin reescalado.

El guardado escribe un temporal en el mismo directorio, hace flush/fsync y
reemplaza atómicamente el destino; un fallo de escritura conserva el checkpoint
anterior. Los checkpoints periódicos se guardan al finalizar una época, y se
hace un guardado final con historia/tiempo y motivo de parada. `pause_after`
pausa tras ese número de épocas adicionales. Una interrupción de teclado o
fallo numérico durante una época revierte sus pesos, optimizador y RNG al
inicio de esa época; continuar la repetirá. Una terminación abrupta del proceso
requiere cargar el último checkpoint terminado y repetir las épocas posteriores.
Los fallos de archivo se propagan como errores de I/O.

`weights.jsonl` es independiente del NPZ: registra epoch/update globales,
configuración, pesos seleccionados `(layer, row, column)`, valores y cambios,
y normas de pesos, biases, gradientes y actualizaciones por capa. El registro
está desactivado hasta indicar una ruta; la frecuencia controla su volumen.
Se escribe sólo después de terminar la época. Si se vuelve a un checkpoint
anterior a registros ya escritos después de una terminación abrupta, el análisis
debe descartar registros posteriores al checkpoint o deduplicar por update antes
de concatenar la continuación. La generación de gráficos reutilizables corresponde
al paso 6.

Se mantiene lectura de modelos SGD versión 1 y modelos versión 2. Estos formatos
conservaban sólo la historia de la última llamada y no la semilla inicial:
al migrarlos, la semilla y tasas históricas desconocidas se marcan `null`, y los
contadores históricos no reconstruibles también. Las épocas se numeran desde la
historia disponible. SGD versión 1 inicia el contador de updates en cero; versión 2
conserva el contador del optimizador. Las predicciones permanecen iguales.

Comprobaciones desde la raíz:

```bash
python3 -m unittest tps_sia.tp3.shared.tests.test_training_state tps_sia.tp3.shared.tests.test_optimizers
python3 -m tps_sia.tp3.shared.tests.test_mlp
python3 -m tps_sia.tp3.ej2.tests.test_shared_compatibility
```

Las pruebas de estado comparan parámetros, momentos, RNG, selección e historia
numérica de entrenar N épocas frente a pausar/guardar/cargar tras K y continuar
N−K, para SGD, momentum y Adam; excluyen tiempos. Cubren selección y reanudación
desde la mejor época, criterios de parada, rollback de épocas incompletas,
checkpoints periódicos, escritura atómica, registro e índices acumulados y
migración de formatos anteriores. Los tests de gradientes, XOR y compatibilidad
siguen verificando el núcleo compartido. No se ejecutaron barridos ni se usó
`digits_test.csv` para estas comprobaciones.

## Paso 6 — Experimentos y análisis reutilizables

`shared/experiments.py` ejecuta corridas de desarrollo con configuración validada,
`shared/metrics.py` calcula métricas y `shared/analysis.py` genera tablas y figuras
leyendo resultados guardados. Los comandos de ej2 delegan en estos módulos;
el baseline conserva su comando, configuración y resultados de referencia.
Su importación histórica `baseline.confusion_matrix` sigue disponible.

Desde la raíz, para ejecutar **una** configuración:

```bash
python -m tps_sia.tp3.ej2.src.experiments --config tps_sia/tp3/ej2/configs/search.json --candidate baseline --seed 42 --output-dir tps_sia/tp3/ej2/results/search
python -m tps_sia.tp3.ej2.src.analysis --results-dir tps_sia/tp3/ej2/results/search --output-dir tps_sia/tp3/ej2/results/analysis
```

`search.json` predefine 11 candidatos de tasa/optimizador, incluyendo el baseline,
y las semillas `[42, 0, 1]`. Cada invocación entrena sólo el candidato y la semilla
indicados; `--all` ejecuta explícitamente los 33 pares. La comparación y selección
completas pertenecen al paso 7 y aún están pendientes. La regla predefinida ordena
por accuracy de validación, menor cross-entropy y menor número de parámetros;
los finalistas se confirman con semillas comunes, sin elegir la mejor semilla.

Las rutas `dataset` y `cache` del JSON se resuelven respecto de su directorio.
El runner recibe rutas explícitas y rechaza `digits_test.csv`, incluidos enlaces
simbólicos que lo referencian, antes de cargar datos. Nunca se usa test para
seleccionar hiperparámetros o la mejor época.

La configuración registra arquitectura, activación y `beta`, salida y loss,
optimizador con sus parámetros y tasa constante, estrategia, tamaño de lote,
shuffle, inicialización, semillas, fracción de validación, épocas, stopping,
preprocesamiento y registro de pesos. Se validan sus combinaciones: online usa
lote 1, batch usa `null`, mini-batch usa un entero ≥ 2; softmax corresponde a
cross-entropy y logística al medio error cuadrático. `preprocessing.name`
admite `identity` (píxeles originales del baseline) y `standardize` (media/desvío
ajustados exclusivamente con train, con escala 1 en columnas constantes).
La transformación queda guardada en ambos modelos y en los resultados.

Cada corrida genera `<config_id>-seed-<seed>/`, sin sobrescribir directorios
existentes. La huella de configuración excluye la semilla del modelo y la ruta
del caché para agrupar repeticiones; incluye la procedencia de pesos iniciales.

| Archivo | Contenido |
|---|---|
| `manifest.json` | Identidad necesaria para reanudar: configuración, SHA-256 del CSV, origen de pesos y metadatos de búsqueda |
| `split.npz` | Índices de train/validación en el orden original del CSV |
| `results.json` | Configuración efectiva, semillas, SHA-256 del CSV y partición, conteos por clase, entorno/hilos, tiempo, parámetros, época elegida, motivo de parada y métricas |
| `history.csv` | Epoch global, costo/accuracy de train y validación, tasa, updates y norma de gradiente |
| `checkpoint.npz` | Último estado de entrenamiento para reanudar, incluido optimizador y RNG |
| `best_model.npz` | Mejor época según el monitor de validación configurado, para inferencia |
| `weights.jsonl` | Registro opcional de pesos seleccionados y normas de pesos, gradientes y actualizaciones por capa |

Las métricas en `results.json` corresponden a `best_model.npz`; la historia
conserva todas las épocas ejecutadas y `checkpoint.npz` conserva la última.
`epochs` es el presupuesto **total** de la corrida. Para pausar tras una época
y reanudar con el mismo JSON, semilla y directorio de salida:

```bash
python -m tps_sia.tp3.ej2.src.experiments --config tps_sia/tp3/ej2/configs/search.json --candidate baseline --seed 42 --output-dir tps_sia/tp3/ej2/results/paused --pause-after 1
python -m tps_sia.tp3.ej2.src.experiments --config tps_sia/tp3/ej2/configs/search.json --candidate baseline --seed 42 --output-dir tps_sia/tp3/ej2/results/paused --resume
```

Una corrida completada o con fallo numérico no se reanuda ni sobrescribe.
`--initial-model ruta/modelo.npz` inicia otra corrida desde pesos compatibles,
con historia, optimizador y RNG nuevos; requiere igual preprocesamiento. Para
reanudar esa corrida se conservan `--initial-model` y se agrega `--resume`;
el archivo inicial se verifica por su hash, sin volver a copiar sus pesos.

Las métricas incluyen cross-entropy, accuracy, precision/recall/F1 por clase,
soporte, balanced accuracy, macro-F1 y confusión 10×10. Precision sin predicciones
y recall sin muestras reales se guardan como `null`. F1 se calcula como
`2TP / (support + predictions)`: es cero si una clase observada no se predice,
y `null` si no tiene muestras ni predicciones. Macro-F1 y balanced accuracy
promedian sólo clases con soporte real y enumeran esas clases en el JSON.
Por eso la validación de `digits.csv` no evalúa el reconocimiento del 8.

El análisis escribe `comparison.csv`, `seed_summary.csv` / `.json` y
`run_statuses.json`. Las tablas muestran costo computacional y cantidad de
parámetros; el resumen agrupa por configuración, dataset y partición, con
media y desvío poblacional entre semillas. Una sola semilla tiene desvío cero,
lo que no demuestra estabilidad entre repeticiones. Corridas interrumpidas o
fallidas quedan en el listado de estados y fuera de las comparaciones.
Las figuras incluyen curvas de aprendizaje, época elegida, confusión en
conteos y normalizada por fila (filas ausentes como N/E), y evolución de pesos
cuando se habilitó su registro. Se pueden regenerar sin cargar ningún dataset.
Dependencias: NumPy y Matplotlib; los comandos de entrenamiento no importan
Matplotlib.

### Validación del paso 6

```bash
python -m unittest tps_sia.tp3.shared.tests.test_metrics tps_sia.tp3.shared.tests.test_experiments tps_sia.tp3.ej2.tests.test_experiments
```

Las pruebas usan CSV sintéticos y verifican métricas manuales, clases ausentes,
configuraciones inválidas, aislamiento de test, artefactos y recarga,
preprocesamiento ajustado sólo con train, fallos numéricos, resúmenes por semilla,
compatibilidad del split 80/20, inicialización desde pesos y reanudación exacta
para los tres optimizadores. Con Matplotlib disponible verifican también las
cinco figuras guardadas; esa prueba se omite si falta la dependencia.

## Paso 7 — Comparación y selección de desarrollo

La búsqueda completa tiene un comando propio; `experiments --all` conserva
su significado anterior y ejecuta todas las tasas con todas las semillas.
El comando por etapas evita ese barrido redundante:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m tps_sia.tp3.ej2.src.search --config tps_sia/tp3/ej2/configs/search.json --output-dir tps_sia/tp3/ej2/results --workers 4
python3 -m tps_sia.tp3.ej2.src.search_analysis --results-dir tps_sia/tp3/ej2/results --output-dir tps_sia/tp3/ej2/results/analysis
python3 -m unittest tps_sia.tp3.ej2.tests.test_search
```

En PowerShell, establecer `$env:OPENBLAS_NUM_THREADS="1"` y
`$env:OMP_NUM_THREADS="1"` antes de ejecutar el comando sin esos prefijos.
`--workers` controla corridas simultáneas; su valor por defecto es 1.
NumPy basta para entrenar; el análisis requiere además Matplotlib.

El protocolo se guarda antes de entrenar y se conserva al repetir el comando.
Una corrida de dos épocas estima el costo. Luego se comparan las 11 variantes
de tasa/optimizador con semilla 42 y presupuesto común de 30 épocas. Se
eligen los dos optimizadores mejor ubicados, cada uno con su mejor tasa,
para comparar `[784,64,10]`, `[784,128,10]`, `[784,256,10]` y
`[784,128,64,10]`. La arquitectura de 128 neuronas reutiliza la corrida
anterior. Las tres mejores configuraciones del conjunto explorado y el
baseline se confirman con las semillas comunes 42, 0 y 1; cada corrida ya
terminada se reutiliza. Las decisiones quedan en `rates.json`,
`architectures.json`, `finalists.json` y `confirmation.json`.

Cada checkpoint se elige por la menor cross-entropy de validación. El ranking
de configuraciones usa accuracy media de las tres semillas, cross-entropy
media, cantidad de parámetros y, como desempate determinista, identificador
de configuración. No se selecciona una semilla favorable: el candidato usa
la semilla predefinida 42 y su mejor checkpoint de desarrollo. Se guarda
`results/selection.json` con configuración, semillas, huellas, evidencia,
modelo candidato y mediana de las mejores épocas como presupuesto para un
eventual reentrenamiento del paso 9. Este paso no reentrena con toda la data.

Repetir el mismo comando reutiliza corridas completas y reanuda las
interrumpidas desde su checkpoint; no reemplaza resultados terminados.
Un protocolo distinto requiere otro directorio. Usar un solo proceso
coordinador de `search` por directorio. Con varios workers, una interrupción
del coordinador puede esperar a que terminen las corridas activas.

El análisis usa sólo archivos guardados y genera tablas por etapa,
curvas comparadas, dispersión y costo de confirmación, métricas por clase,
confusiones y curvas individuales. Los tiempos son de pared y dependen de
la concurrencia y del equipo; no constituyen una medición aislada de la
velocidad relativa de los optimizadores. La selección no usa esos tiempos.
Ver el [análisis de desarrollo](development-report.md) para las respuestas
a las preguntas del ejercicio y la interpretación de las comparaciones.


## Paso 7b — Búsqueda v2: un factor a la vez

La búsqueda del paso 7 (26 corridas en `results/runs/`, `results/selection.json`)
se conserva como antecedente **v1**. La v2 rehace la selección con un protocolo de
*coordinate descent* pre-registrado en
[`configs/search_v2.json`](configs/search_v2.json): cada etapa cambia **un** factor
y el resto queda en el mejor valor de la etapa anterior. Está implementada y probada
con un smoke test; **la búsqueda completa todavía no se ejecutó**.

Base fija: `[784,128,10]`, softmax + cross-entropy, lote 32, inicialización `auto`
(He para ReLU, Xavier para tanh), partición estratificada 80/20 con `split_seed` 42,
píxeles sin transformar. Todas las corridas tienen como máximo 100 épocas, parada
temprana por cross-entropy de validación (`patience` 10, `min_delta` 0) y
checkpoint de la mejor época. Una corrida que agota las 100 épocas se informa como
"no convergió", no como mala, y compite con su mejor checkpoint. La semilla del
modelo cambia pesos iniciales y orden de lotes; la partición no cambia.

| Etapa | Factor | Candidatos (semilla 42 salvo indicación) |
|---|---|---|
| 0 | Activación | {tanh, relu} × SGD {0.01, 0.1}; cada activación toma su mejor tasa; diferencia < 0.3 pp → tanh |
| 1 | Tasa por optimizador | SGD {1e-3, 1e-2, 5e-2, 1e-1}; momentum 0.9 {1e-4, 1e-3, 1e-2, 5e-2}; RMSProp y Adam {1e-4, 3e-4, 1e-3, 3e-3}; regla de borde |
| 2 | Optimizador | Mejores de la etapa 1, reutilizados; el top-2 pasa a la etapa 6 |
| 3 | Ancho | {32, 64, 128, 256, 512} |
| 4 | Profundidad | `[h]`, `[h, h/2]`, `[h, h/2, h/4]` con el mejor ancho h |
| 5 | Lote | {16, 32, 64, 128}, sin cambiar la tasa |
| 6 | Cruce | top-2 optimizadores × (tasa ganadora, mejor vecina) × top-2 arquitecturas; semillas 42, 0, 1 |
| 7 | Confirmación | top-3 de la etapa 6 + control (ganador de la etapa 0); semillas 42, 0, 1, 2, 3 |

Reglas pre-registradas:

- **Ranking** (semilla 42): accuracy de validación, luego cross-entropy, parámetros y
  `config_id`. Las corridas con fallo numérico no compiten.
- **Borde** (etapa 1): si la mejor tasa de un optimizador está en un extremo de la
  grilla probada, se agrega el siguiente punto de la secuencia 1-3-10 en esa dirección
  (p. ej. 0.1 → 0.3, 5e-2 → 0.1, 1e-4 → 3e-5) y se repite hasta que quede adentro.
  Una tasa que diverge cuenta como probada y peor. Se registra cada extensión; el
  tope de seguridad es de 4 extensiones por lado y alcanzarlo queda "sin resolver".
- **Empate cercano** (etapas 2–5): si los dos mejores difieren menos de 0.3 pp, ambos
  se repiten con las semillas 0 y 1 y se decide con la media de 3 semillas. La etapa 0
  usa su propia regla de empate (tanh); la etapa 1 elige una tasa por optimizador con
  semilla 42 y los empates de tasa se revisan en la etapa 6 (tasa vecina).
- **Mejora**: con varias semillas, una configuración es mejor sólo si su ventaja media
  supera 2σ, con σ = desvío muestral agrupado `sqrt((s_a² + s_b²)/2)` (ddof=1). Si no,
  se conserva el valor vigente (SGD, ancho 128, una capa, lote 32) cuando está entre los
  dos; si no, decide el ranking de 3 semillas. Las etapas 6 y 7 ordenan por accuracy
  media, cross-entropy media, parámetros y `config_id`, y registran si la ventaja sobre
  el segundo y sobre el control supera 2σ.

RMSProp (`shared/optimizers.py`) usa `s = rho*s + (1-rho)*g²` y
`p -= learning_rate*g/(sqrt(s)+optimizer_epsilon)`, con estado inicial cero y sin
corrección de sesgo; defaults `learning_rate=0.001`, `rho=0.9`,
`optimizer_epsilon=1e-8`. Su estado `second_moment` se guarda y restaura como los
de momentum/Adam, por lo que la reanudación es exacta.

### Comandos

Desde la raíz del repositorio (la carpeta que contiene `tps_sia/`), una etapa por
invocación y en orden; cada etapa lee las decisiones guardadas de las anteriores:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"; $env:OMP_NUM_THREADS = "1"
python -m tps_sia.tp3.ej2.src.staged_search --stage 0 --workers 4
python -m tps_sia.tp3.ej2.src.staged_search --stage 1 --workers 4
# ... etapas 2 a 7
python -m tps_sia.tp3.ej2.src.search_analysis --protocol v2 --results-dir tps_sia/tp3/ej2/results/v2 --output-dir tps_sia/tp3/ej2/results/v2/analysis
```

En bash: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m ...`. Por defecto se usan
`--config tps_sia/tp3/ej2/configs/search_v2.json` y
`--output-dir tps_sia/tp3/ej2/results/v2`. `--workers` corre corridas independientes en
procesos paralelos (default 1). Repetir una etapa ya decidida no reentrena: imprime su
decisión. Una corrida interrumpida se reanuda desde su checkpoint repitiendo el
comando. Si `search_v2.json` cambia, las etapas guardadas se rechazan (su SHA-256 no
coincide): un protocolo distinto requiere otro directorio.

Salidas en `results/v2/`:

| Archivo | Contenido |
|---|---|
| `runs/<config_id>-seed-<semilla>/` | Artefactos de cada corrida (como en el paso 6). Una configuración idéntica pedida por otra etapa se reutiliza |
| `stage_<N>.json` | Corridas de la etapa (accuracy, cross-entropy, mejor época, motivo de parada, convergencia, parámetros y tiempo), decisión, empates cercanos, extensiones de borde y ganador |
| `selection.json` | Etapa 7: configuración congelada, ranking de 5 semillas, comparaciones 2σ, modelo candidato (semilla 42) y épocas de reentrenamiento (mediana de las mejores épocas) |

Portabilidad: en v2, `dataset`, `cache`, `run_directory` y el modelo candidato se guardan
relativos a `tp3/`, y `config_id` excluye rutas dependientes de la máquina, por lo que
la misma configuración tiene el mismo id en cualquier equipo. El SHA-256 del protocolo
se calcula con fines de línea LF (Git con `core.autocrlf` escribe CRLF en Windows).
El SHA-256 de `digits.csv` sí depende de sus bytes: un checkout con otros fines de
línea produce otro hash y no reutiliza corridas. Los resultados v1 conservan sus
rutas absolutas originales y no se migraron.

El análisis v2 genera, por etapa, tabla (`stage_<N>.csv` y `comparison_v2.md`), barras
de accuracy con desvío entre semillas cuando existen, curvas train/validación de las
corridas relevantes (semilla 42), mejor época, motivo de parada, parámetros y tiempo;
la etapa 1 agrega accuracy vs. tasa con las extensiones marcadas. Si existe
`selection.json`, agrega precision/recall/F1 por dígito del ganador (5 resaltado; 8 no
evaluable), curvas y confusión. No carga datasets ni entrena.

### Verificación y costo estimado

```bash
python -m tps_sia.tp3.shared.tests.test_optimizers
python -m tps_sia.tp3.ej2.tests.test_staged_search
```

`test_staged_search` verifica con resultados sintéticos la regla de borde (extiende,
reevalúa, cierra con divergencia y respeta el tope), el empate cercano (agrega semillas
0 y 1), la regla de empate de activación, la regla 2σ, el orden de desempate de la
etapa 7, el rechazo de grupos incompletos y el flujo completo hasta `selection.json`.
También verifica `config_id` idéntico con distintas rutas absolutas, el hash
independiente del fin de línea y el rechazo de `digits_test.csv` (también por enlace;
en Windows sin permisos para symlinks se emula su resolución). El smoke test entrena
de verdad las etapas 0 y 1 con 2 épocas y 256 muestras de train de `digits.csv`, en un
directorio temporal: es una comprobación técnica, no un experimento.

Tiempo por época medido en Windows 11, Python 3.13, NumPy 2.5.3, un hilo de BLAS,
9960 muestras de train y evaluación de validación en cada época:

| Configuración | Sólo entrenamiento | Con checkpoints del runner |
|---|---:|---:|
| SGD `[784,128,10]`, lote 32 | 0.40 s | 0.52 s |
| SGD `[784,512,10]`, lote 32 | 2.3 s | 2.65 s |
| Adam `[784,128,10]`, lote 32 | 0.79 s | 1.00 s |

Con unas 95 corridas nuevas y 30–60 épocas típicas por la parada temprana, la búsqueda
completa se estima en 1.5–3.5 h en serie (la etapa 6, con 24 corridas, es la más
cara); con `--workers 4` y un hilo por proceso, del orden de 0.5–1 h. Cada corrida
ocupa 3–12 MB (checkpoint y mejor modelo, con estado del optimizador).
