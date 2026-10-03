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
La historia acumulativa, parada temprana y controles completos de entrenamiento
corresponden al paso 5; cada llamada todavía genera su propia historia.

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
