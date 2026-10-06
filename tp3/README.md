# TP3 — Perceptrón Simple y Multicapa

Sistemas de Inteligencia Artificial — ITBA

## Validación de la implementación

```bash
python -m tps_sia.tp3.ej1.tests.test_validacion
python -m tps_sia.tp3.ej2.tests.test_validacion
python -m tps_sia.tp3.ej2.tests.test_datos_digitos
python -m tps_sia.tp3.ej2.tests.test_shared_compatibility
python3 -m unittest tps_sia.tp3.shared.tests.test_training_state tps_sia.tp3.shared.tests.test_optimizers
python3 -m unittest tps_sia.tp3.shared.tests.test_metrics tps_sia.tp3.shared.tests.test_experiments tps_sia.tp3.ej2.tests.test_experiments
python -m unittest tps_sia.tp3.ej2.tests.test_staged_search tps_sia.tp3.ej2.tests.test_appendix tps_sia.tp3.ej2.tests.test_final_evaluation
```

Ejercicios de validación del enunciado: AND con perceptrón escalón, XOR (no resoluble con un
perceptrón simple), recta con el lineal, `y = tanh(x)` y sigmoide con el no lineal,
`y = max(0, 2x + 1)` con ReLU (opcional), y chequeo numérico de las derivadas.

## Organización del código

Los comandos se ejecutan con `python -m ...` desde la raíz del repositorio.
Dependencias para TP3: NumPy, Pandas y Matplotlib.

| Archivo | Contenido |
|---|---|
| `shared/activations.py` | θ y θ′: escalón, lineal, logística, tanh, ReLU |
| `shared/mlp.py` | MLP matricial, backprop, entrenamiento e historia, guardar/cargar |
| `shared/optimizers.py` | SGD, momentum clásico y Adam; validación y exportación/restauración de estado |
| `shared/digit_dataset.py` | Loader cacheado con rutas explícitas y partición estratificada de dígitos |
| `shared/metrics.py` | Métricas por clase y confusión 10×10 con política de valores indefinidos |
| `shared/experiments.py` | Runner validado, trazabilidad, artefactos y reanudación |
| `shared/analysis.py` | Tablas, dispersión entre semillas y figuras desde resultados guardados |
| `shared/tests/` | Chequeos reutilizables con datos sintéticos |
| `ej1/src/perceptron.py` | Perceptrón simple, online / mini-batch / batch, guardar/cargar |
| `ej1/src/datos.py` | Fraude: carga, exploración, z-score, soluciones analíticas, estratos y k-fold |
| `ej1/src/metricas.py` | MSE, MAE, R²; precision, recall, F1, matriz de confusión |

`shared` no depende de los ejercicios. Las rutas anteriores
`ej1/src/activaciones.py`, `ej2/src/mlp.py` y `ej2/src/optimizadores.py`
reexportan las mismas implementaciones. `ej2/src/datos_digitos.py` conserva
los defaults y el comando de exploración de ej2. Ver las
[importaciones y comprobaciones del paso 3a](ej2/README.md#paso-3a--núcleo-compartido).

Cada ejercicio conserva sus datasets, comandos, configuraciones y resultados.
Los scripts `experimentos_*` entrenan y guardan JSON; los de `graficos_*` /
`analisis_*` sólo leen y grafican.

---

## Ejercicio 1 — TinyModel (Knowledge Distillation)

Perceptrón simple lineal vs. no lineal (logístico) para estimar la probabilidad de fraude de
BigModel (`fraud_dataset.csv`, 7500 transacciones).

### Cómo correrlo

```bash
# Aprendizaje (todas las muestras)
python -m tps_sia.tp3.ej1.src.experimentos_aprendizaje       # ≈ 11 min
python -m tps_sia.tp3.ej1.src.graficos_aprendizaje           # figuras y tabla
python -m tps_sia.tp3.ej1.src.repeticiones_aprendizaje       # 3 corridas en paralelo, ≈ 13 min
python -m tps_sia.tp3.ej1.src.analisis_repeticiones          # análisis de las repeticiones

# Generalización (k-fold)
python -m tps_sia.tp3.ej1.src.experimentos_generalizacion    # ≈ 8 min
python -m tps_sia.tp3.ej1.src.graficos_generalizacion        # figuras y tabla

# Opcional: ReLU
python -m tps_sia.tp3.ej1.src.experimentos_relu              # ≈ 5 min
python -m tps_sia.tp3.ej1.src.graficos_relu                  # figuras y tabla
```

Salidas en `ej1/salidas/aprendizaje/`, `ej1/salidas/generalizacion/` y `ej1/salidas/relu/`,
resueltas desde cada módulo sin depender del directorio de trabajo.

### Datos

* Sin nulos, duplicados ni negativos. Objetivo `big_model_fraud_probability` en `[0,1]`.
* `flagged_fraud` **no se usa para entrenar**. Vale exactamente `BigModel ≥ 0.85`.
* Entradas: 9 features, estandarizados con z-score. Sin normalizar, el lineal diverge y el
  logístico arranca 100 % saturado.


### Aprendizaje (7500 muestras)

| Modelo | η | β | MSE | R² | Salidas fuera de [0,1] |
|---|---|---|---|---|---|
| Lineal | 10⁻⁴ | — | 0.02606 | 0.715 | 5.98 % |
| **Logística** | 5·10⁻³ | 0.5 | **0.01087** | **0.881** | **0 %** |

* **η** (tasa de aprendizaje): cuánto se mueven los pesos en cada actualización,
  Δw = η · (ζ − O) · θ′(h) · x. Muy chica, aprende lento; muy grande, oscila o diverge.
* **β**: controla la pendiente de la logística, θ(h) = 1 / (1 + e^(−2βh)). Con β chico la
  curva es casi lineal; con β grande se parece a un escalón y se satura antes.
* **MSE** (error cuadrático medio) = (1/p) · Σ (ζᵘ − Oᵘ)², donde ζ es la probabilidad de
  BigModel y O la salida del perceptrón. Es el costo de la Clase 10.2 promediado sobre las
  p muestras: cuanto menor, mejor copia el modelo a BigModel.
* **R²** (coeficiente de determinación) = 1 − MSE / Var(ζ). Es la fracción de la variación
  del objetivo que explica el modelo: 1 es perfecto y 0 equivale a predecir siempre el
  promedio.

η y β elegidos por barrido. Repetido 3 veces con semillas distintas: mismos resultados.

#### (a) ¿Observan underfitting?

Sí, en el perceptrón lineal. Converge a un MSE de 0.0261 y deja un 28.5 % de
la varianza sin explicar (R² = 0.715). Ningún valor de η lo mejora: entre 10⁻⁴ y 10⁻² el
error final es siempre el mismo, así que el problema no es de entrenamiento sino del modelo.
Los residuos tienen un patrón sistemático según la zona del objetivo, y un 6 % de las
salidas cae fuera de [0,1]. La logística también tiene
algo de underfitting, porque es una sola neurona, pero mucho menor: R² = 0.881 y sin
salidas fuera de rango.

#### (b) ¿Observan saturación de las capacidades?

Sí, en los dos modelos. El lineal alcanza el 99 % de su mejora en 5 épocas y la logística en
12. Después, multiplicar las épocas hasta 800 no cambia el error: la mejora por época cae
seis órdenes de magnitud y se vuelve ruido numérico. Los dos agotaron lo que pueden aprender
con una sola neurona. En la logística aparece además la saturación de la sigmoide: la
fracción de muestras con derivada casi nula pasa del 4.6 % al 12.6 %. Sin normalizar las
entradas, en cambio, el 100 % arranca saturado y el modelo no aprende.

#### (c) ¿Cuál seleccionarían para el estudio de generalización?

El perceptrón no lineal con activación logística. Tiene mayor potencial de aprendizaje:
reduce el error un 58 % (MSE 0.0109 contra 0.0261) y explica el 88 % de la varianza contra
el 72 % del lineal. Además, su salida está siempre en (0,1), que es exactamente el rango de
una probabilidad, mientras que el lineal entrega valores como −0.4 o 2.1. También es mucho
más preciso en la zona que importa al negocio, la de probabilidad de fraude alta. El costo es
un hiperparámetro más (β) y un entrenamiento algo más lento, pero en inferencia los dos
cuestan lo mismo.


### Generalización (perceptrón logístico)

#### (a) ¿Qué métricas seleccionaron y por qué?

Precision, recall y F1. Descartamos la accuracy porque los errores no pesan lo mismo y las
clases no están balanceadas: con 11.6 % de fraude, un modelo que nunca alerta ya tiene 88.4 %
de accuracy, un número engañoso. El recall es lo más importante, porque mide cuántos fraudes
detectamos, y cada fraude que se escapa es dinero perdido. La precision mide cuántas alertas
eran fraude realmente: queremos evitar falsas alarmas, pero no es la prioridad. El F1 resume
el balance entre ambas en un dataset desbalanceado. Criterio final: recall de al menos 95 % con la mayor precision posible.

#### (b) ¿Qué estrategia utilizaron? ¿Cómo se elige el mejor conjunto de entrenamiento?

Validación cruzada k-fold: se divide el conjunto en 5 partes, se
entrena con 4 y se evalúa con la restante, rotando, y se promedian los resultados. Las
particiones son estratificadas, para que cada parte tenga el mismo 11.6 % de fraude, y se
repite todo 3 veces. El normalizador y el umbral se ajustan sólo con los datos de
entrenamiento de cada fold. Elegimos k = 5 porque con k = 10 los
folds son más ruidosos. El mejor conjunto de entrenamiento es uno representativo,
estratificado como el de producción, y suficiente: desde unas 1500 muestras ya no hay
diferencia entre entrenamiento y prueba.

#### (c) ¿Cuál es el mejor modelo?

**Perceptrón logístico con 6 features, η = 0.001, β = 0.25, 60 épocas y umbral 0.70.**

Detecta el 95 % del fraude con una precision de 0.78 y un F1 de 0.86. Lo elegimos con una
grilla de η, β y conjuntos de features, buscando la mayor precision con recall de al menos
95 %. Sacar los tres features que son ruido (`timestamp`, `time_since_last_login_s` y
`device_screen_resolution`) no empeora el resultado y deja un modelo de sólo 7 pesos. Las
épocas también se validaron: con 60 la precision es máxima, y entrenar más la empeora. No
hay overfitting: entrenamiento y prueba dan el mismo F1 (0.855 y 0.856). El perceptrón
lineal, con el mismo recall, sólo alcanza una precision de 0.62.

| Modelo | Recall | Precision | F1 prueba | F1 entrenamiento | Diapositiva 30 |
|---|---|---|---|---|---|
| **Logística** | **0.951** | **0.779** | **0.856** | 0.855 | Buen modelo |
| Lineal | 0.952 | 0.615 | 0.747 | 0.746 | Underfitting |

| | Predice fraude | Predice legítima |
|---|---|---|
| **Real: fraude** | TP = 827 | FN = 42 |
| **Real: legítima** | FP = 238 | TN = 6393 |

#### Umbral recomendado: 0.70

| Umbral | Recall | Precision | Fraudes no detectados | Falsas alarmas |
|---|---|---|---|---|
| **0.70 (recall ≥ 95 %)** | **0.951** | 0.774 | **43** | 241 |
| 0.82 (F1 máximo) | 0.836 | 0.941 | 143 | 46 |
| 0.85 (el de BigModel) | 0.796 | 0.961 | 177 | 28 |

El modelo final se entrena con las 7500 muestras y se guarda en
`ej1/salidas/generalizacion/modelo_final.json`, con pesos, normalización y umbral.

### Opcional: activación ReLU

θ(h) = max(0, h), con θ′(h) = 1 si h > 0 y 0 si no. Se repitieron los estudios de
aprendizaje y generalización con los mismos protocolos.

| Modelo | MSE | R² | Fuera de [0,1] | Recall | Precision | F1 | Diapositiva 30 |
|---|---|---|---|---|---|---|---|
| Lineal | 0.0261 | 0.715 | 6.0 % | 0.952 | 0.615 | 0.747 | Underfitting |
| **Logística** | **0.0109** | **0.881** | **0 %** | 0.951 | **0.779** | **0.856** | Buen modelo |
| ReLU | 0.0258 | 0.718 | 4.6 % | 0.950 | 0.607 | 0.740 | Underfitting |

#### ¿Qué efecto tiene en las conclusiones anteriores?

Ninguna conclusión cambia: la ReLU se comporta casi igual que el perceptrón lineal. Con un
η adecuado, sólo el 2 % de las muestras queda con h ≤ 0, así que en el resto es la
identidad. Corta las salidas negativas en 0, pero sigue pasándose de 1 en el 4.6 % de los
casos. Mantiene el underfitting del lineal (R² = 0.718) y satura su capacidad igual de
rápido, en 5 épocas. Su saturación propia es otra: con η grande la ReLU "muere", con 94 % de
muestras sin gradiente. En generalización empata con el lineal, así que la logística sigue
siendo la elegida.

---

## Ejercicio 2 — Clasificación de dígitos (perceptrón multicapa)

MLP matricial con backpropagation, activaciones tanh/ReLU/logística, salida softmax con
cross-entropy, entrenamiento online/mini-batch/batch, optimizadores SGD, momentum, Adam y
RMSProp, parada temprana, checkpoints con reanudación exacta y métricas por clase. El
detalle de cada paso está en el [README del ejercicio](ej2/README.md).

### Cómo correrlo

```bash
# Búsqueda por etapas: una invocación por etapa, en orden (0 a 7)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m tps_sia.tp3.ej2.src.staged_search --stage 0 --workers 4
# ... etapas 1 a 7
python -m tps_sia.tp3.ej2.src.search_analysis --protocol v2 --results-dir tps_sia/tp3/ej2/results/v2 --output-dir tps_sia/tp3/ej2/results/v2/analysis

# Apéndice: comprobaciones posteriores que no cambian la selección
python -m tps_sia.tp3.ej2.src.appendix run --part relu --workers 4
python -m tps_sia.tp3.ej2.src.appendix run --part stages_3_5 --workers 4
python -m tps_sia.tp3.ej2.src.appendix report

# Evaluación final sobre digits_test.csv (una sola vez; no repite sin --overwrite)
python -m tps_sia.tp3.ej2.src.final_evaluation
```

En PowerShell, definir antes `$env:OPENBLAS_NUM_THREADS = "1"` y
`$env:OMP_NUM_THREADS = "1"`. Los resultados se guardan en `ej2/results/v2/` y
`ej2/results/v2_appendix/`. Se versionan las decisiones, métricas, figuras y el modelo
candidato; los checkpoints y los demás modelos (unos 550 MB) están en el Drive del equipo.

### Resultados

Las respuestas completas, con las tablas de cada etapa, están en el
[informe de desarrollo](ej2/development-report.md).

* **(a) Evaluación:** partición estratificada 80/20 de `digits.csv` (semilla 42), accuracy
  de validación como criterio, cross-entropy para desempatar y elegir el checkpoint,
  métricas por clase y confusión. `digits_test.csv` se usa una sola vez, al final.
* **(b) Variantes:** búsqueda de un factor a la vez (activación, tasa para cuatro
  optimizadores, optimizador, ancho, profundidad y lote), un cruce con tres semillas y una
  confirmación con cinco. Son 58 corridas en total.
* **Selección:** `[784,256,10]`, tanh, momentum 0.9 con tasa 0.1, lote 32: accuracy media de
  validación **97.24 % ± 0.15** con cinco semillas. Supera al control (SGD 0.1, 96.44 %) por
  encima del umbral 2σ; frente a la misma red con 128 neuronas (97.08 %) la diferencia
  no es distinguible.

| Evaluación final | Muestras | Accuracy |
|---|---:|---:|
| Test, todas las clases | 2497 | **86.70 %** |
| Test, sin el 8 | 2254 | **96.05 %** |
| Validación del mismo modelo | 2489 | 97.07 % |

`digits.csv` no tiene ningún 8, así que la red no puede reconocerlo: los 243 ochos del test
son errores y explican casi toda la caída de la accuracy global.

---

## Ejercicio 3 — Dígitos con `more_digits.csv`

Implementados los datos y el [flujo de desarrollo](ej3/README.md): controles,
búsqueda por tasas/arquitecturas/lotes, confirmación con semillas comunes,
corridas puntuales, reanudación, estudio de factores y análisis desde resultados
guardados. Los experimentos todavía no se ejecutaron. No incluye evaluación
final en test, reentrenamiento para entrega ni técnicas opcionales.
