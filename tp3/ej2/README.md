# Ejercicio 2 — Datos de dígitos y MLP

Desde la raíz del repositorio:

```bash
python tps_sia/tp3/ej2/src/datos_digitos.py
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

Para usar los datos desde otros módulos de `src`:

```python
from .datos_digitos import cargar, particionar

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

`src/mlp.py` implementa `MLP` con arquitectura como lista, forward y
backprop matriciales. Reutiliza tanh, ReLU y logística de
`ej1/src/activaciones.py`. La salida por defecto es softmax con
cross-entropy, calculada mediante log-softmax estable; su delta es
`(p - y) / N`. Para una sola salida se debe elegir `salida="logistica"`.

Ejemplo de configuración para el próximo paso de entrenamiento:

```python
from tps_sia.tp3.ej2.src.mlp import MLP

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
- Guardar/cargar conserva parámetros, configuración, SGD, última historia
  y estado aleatorio para reanudar con el mismo orden de mini-batches.
  El archivo `.npz` se carga con `allow_pickle=False`.

`src/optimizadores.py` separa SGD del MLP. La interfaz `Optimizador` expone
`paso(parametros, gradientes)` (actualización in-place) y `configuracion()`.
Se puede inyectar `optimizador=SGD(eta=...)`; momentum y Adam quedan para
el paso siguiente.

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
