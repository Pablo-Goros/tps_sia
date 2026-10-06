# Ejercicio 2 — Desarrollo, selección y generalización

Responde las preguntas (a) y (b) del [enunciado](../Enunciado%20TP3.md) con la búsqueda
por etapas definida en [`configs/search_v2.json`](configs/search_v2.json) y resume la
evaluación final sobre `digits_test.csv`. Los artefactos están en `results/v2/` y
`results/v2_appendix/`: se versionan las decisiones de cada etapa, las métricas e
historias de cada corrida, las figuras, `selection.json`, la evaluación final y el modelo
candidato. Los checkpoints y los modelos del resto de las corridas (unos 550 MB) están en
el Drive del equipo. Los comandos para reproducirlos están en el
[README del ejercicio](README.md#paso-7--búsqueda-por-etapas-un-factor-a-la-vez).

## (a) ¿Cómo se evalúa el desempeño del sistema?

**Datos.** `digits.csv` se divide una única vez en una partición estratificada 80/20 con
semilla 42: 9960 imágenes para ajustar pesos y biases y 2489 para comparar
hiperparámetros. Todas las comparaciones usan exactamente las mismas filas; cada corrida
guarda sus índices, el SHA-256 del CSV y la huella de la partición. La semilla del modelo
cambia sólo la inicialización y el orden de los lotes. `digits_test.csv` no participa del
desarrollo: se usa una única vez al final, como el "mundo real" del enunciado.

**Costo optimizado.** La salida es softmax sobre diez neuronas y el costo es la
cross-entropy media por muestra:

\[
L=-\frac{1}{N}\sum_{i=1}^{N}\sum_{k=0}^{9}y_{ik}\log p_{ik},
\qquad p_{ik}=\frac{e^{z_{ik}}}{\sum_j e^{z_{ij}}}.
\]

Con esta combinación el delta de salida es `(p − y)/N`, con N el tamaño real del lote.
En cada capa oculta, el delta es la sensibilidad del costo respecto de la preactivación:
la suma de los deltas de la capa siguiente ponderada por sus pesos, multiplicada por la
derivada local de la activación (`1 − tanh²(h)` para tanh con β = 1). La actualización
resta el gradiente escalado por la tasa; con momentum clásico,
`v ← μ·v + g` y `W ← W − η·v` (μ = 0.9).

**Métricas de evaluación.** La accuracy (argmax de las diez probabilidades) es el
criterio principal, porque es la métrica que pide el enunciado. La cross-entropy de
validación desempata y además elige el checkpoint de cada corrida. Se registran
precision, recall y F1 por dígito, macro-F1, balanced accuracy y la matriz de confusión
(filas: dígito real). La validación tiene 54 ejemplos del 5 y **ningún 8**: el recall del
8 no es evaluable en desarrollo, y ninguna métrica de validación demuestra que la red
reconozca ese dígito.

**Entrenamiento y parada.** Mini-batch con mezcla por época; hasta 100 épocas con parada
temprana por cross-entropy de validación (paciencia 10, `min_delta` 0) y checkpoint de la
mejor época. Agotar las 100 épocas se informa como "no convergió", no como una mala
configuración.

**Comparación entre configuraciones.** Con una semilla se ordena por accuracy de
validación, luego cross-entropy, cantidad de parámetros e identificador. Con varias
semillas se usan medias, y una configuración se considera mejor sólo si su ventaja media
supera 2σ, con σ el desvío muestral agrupado `sqrt((s_a² + s_b²)/2)`. Todas las reglas se
fijaron en `search_v2.json` antes de correr.

**Generalización.** El modelo seleccionado se evalúa una sola vez sobre `digits_test.csv`,
informando juntas la accuracy global y la accuracy sin el 8 (sección final).

## (b) ¿Qué variantes se prueban para encontrar la solución?

Se varía **un factor por etapa**; el resto queda en el mejor valor de la etapa anterior.
Base: `[784,128,10]`, lote 32, inicialización automática (Xavier para tanh y la salida,
He para capas ocultas ReLU), píxeles sin transformar. Las etapas 0 a 5 usan la semilla
42; la 6 y la 7, varias semillas. En total se entrenaron 58 corridas.

### Etapa 0 — Activación (SGD)

| Activación | Tasa | Acc. val. (%) | CE val. | Mejor época | Parada |
|---|---:|---:|---:|---:|---|
| tanh | 0.01 | 95.38 | 0.164 | 100 | agotó 100 épocas |
| **tanh** | **0.1** | **96.63** | 0.135 | 31 | temprana |
| ReLU | 0.01 | 95.90 | 0.146 | 99 | agotó 100 épocas |
| ReLU | 0.1 | 96.67 | 0.124 | 21 | temprana |

La mejor ReLU supera a la mejor tanh por 0.04 pp (una muestra de validación). Por la regla
pre-registrada, una diferencia menor a 0.3 pp es empate y se elige tanh. La decisión se
apoya en esa regla, no en un desempeño mejor: ReLU tiene menor cross-entropy. El apéndice
revisa esta elección con cinco semillas.

### Etapa 1 — Tasa de aprendizaje por optimizador

Accuracy de validación (%), semilla 42. Cuando la mejor tasa caía en un extremo de la
grilla, se agregó el siguiente valor de la secuencia 1-3-10 hasta encontrar uno peor.

| SGD | | Momentum (μ = 0.9) | | RMSProp | | Adam | |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 92.17 | 0.0001 | 92.17 | 0.0001 | 96.26 | 0.0001 | 96.06 |
| 0.01 | 95.38 | 0.001 | 95.38 | 0.0003 | 96.50 | 0.0003 | 96.26 |
| 0.05 | 96.54 | 0.01 | 96.71 | **0.001** | **96.95** | **0.001** | **96.75** |
| 0.1 | 96.63 | 0.05 | 96.83 | 0.003 | 96.22 | 0.003 | 96.67 |
| **0.3** | **96.95** | **0.1** | **97.47** | | | | |
| 1.0 | 96.26 | 0.3 | 90.68 | | | | |

![Accuracy de validación según la tasa, por optimizador](results/v2/analysis/stage_1_rates.png)

- **Extensiones de borde:** SGD 0.1 → 0.3 → 1.0 y momentum 0.05 → 0.1 → 0.3. Todas las
  mejores tasas quedaron dentro de su grilla.
- **Tasas chicas:** con 1e-4 a 1e-2 (SGD) las corridas agotan las 100 épocas con la
  validación todavía mejorando: convergen lento, no divergen. Con tasas altas la parada
  llega antes (mejor época 9 a 15) y aparece inestabilidad: momentum 0.3 cae a 90.68 %
  y su cross-entropy sube a 0.99.
- **Momentum frente a SGD:** momentum con η = 1e-4 da el mismo resultado que SGD con
  η = 1e-3 (92.17 %, CE 0.2717), y momentum 1e-3 coincide con SGD 1e-2. Con tasas
  chicas, momentum equivale aproximadamente a SGD con tasa efectiva η/(1−μ) = 10η.
- **Accuracy frente a cross-entropy:** momentum 0.1 tiene la mejor accuracy pero peor
  cross-entropy que momentum 0.05 (0.152 contra 0.130): más aciertos, pero más confianza
  en los errores.

### Etapa 2 — Optimizador

Se comparan los mejores de la etapa 1 sin reentrenar: **momentum 0.1** (97.47 %) supera
por 0.52 pp a SGD 0.3 y RMSProp 0.001 (96.95 %), sin empate cercano. SGD 0.3 queda
segundo por menor cross-entropy y pasa con momentum a la etapa 6.

### Etapas 3 a 5 — Arquitectura y lote (momentum 0.1, semilla 42)

| Ancho | Acc. (%) | CE | Parámetros | | Profundidad | Acc. (%) | CE | | Lote | Acc. (%) | CE |
|---:|---:|---:|---:|---|---|---:|---:|---|---:|---:|---:|
| 32 | 94.98 | 0.196 | 25 450 | | **[128]** | **97.47** | 0.152 | | 16 | 89.55 | 0.586 |
| 64 | 96.91 | 0.158 | 50 890 | | [128, 64] | 96.71 | 0.165 | | **32** | **97.47** | 0.152 |
| **128** | **97.47** | 0.152 | 101 770 | | [128, 64, 32] | 94.74 | 0.198 | | 64 | 96.87 | 0.132 |
| 256 | 97.07 | 0.197 | 203 530 | | | | | | 128 | 96.50 | 0.133 |
| 512 | 96.67 | 0.277 | 407 050 | | | | | | | | |

Ninguna etapa activó el empate cercano: se mantienen ancho 128, una capa oculta y lote 32.
La tasa se mantiene fija en 0.1 al cambiar estos factores, como fija el protocolo. Por
eso las redes grandes y el lote 16 (mejor época 2, CE 0.59) parecen estar al límite de la
estabilidad con esa tasa: esas comparaciones están confundidas con la tasa.

### Etapa 6 — Cruce con tres semillas (42, 0, 1)

Dos optimizadores × dos tasas (ganadora y mejor vecina) × dos arquitecturas (ganadora de
la etapa 4 y mejor alternativa de las etapas 3–4):

| Configuración | Acc. media (%) | Desvío (pp) | CE media |
|---|---:|---:|---:|
| momentum 0.1, `[784,256,10]` | 97.19 | 0.14 | 0.190 |
| momentum 0.1, `[784,128,10]` | 97.17 | 0.28 | 0.162 |
| momentum 0.05, `[784,256,10]` | 96.97 | 0.20 | 0.143 |
| SGD 0.3, `[784,128,10]` | 96.95 | 0.08 | 0.136 |
| momentum 0.05, `[784,128,10]` | 96.87 | 0.04 | 0.134 |
| SGD 0.1, `[784,128,10]` | 96.57 | 0.25 | 0.136 |
| SGD 0.3, `[784,256,10]` | 96.41 | 0.16 | 0.144 |
| SGD 0.1, `[784,256,10]` | 96.32 | 0.12 | 0.142 |

Con tres semillas la ventaja de 128 sobre 256 que mostraba la semilla 42 desaparece: las
dos primeras difieren 0.01 pp, muy por debajo del umbral 2σ (0.45 pp).

### Etapa 7 — Confirmación con cinco semillas (42, 0, 1, 2, 3)

| Configuración | Acc. media (%) | Desvío (pp) | CE media | Mejores épocas | Tiempo medio (s) |
|---|---:|---:|---:|---|---:|
| **momentum 0.1, `[784,256,10]`** | **97.24** | 0.15 | 0.190 | 12, 16, 13, 14, 28 | 53.8 |
| momentum 0.1, `[784,128,10]` | 97.08 | 0.24 | 0.159 | 11, 13, 11, 10, 11 | 12.6 |
| momentum 0.05, `[784,256,10]` | 96.88 | 0.19 | 0.142 | 13, 11, 9, 10, 14 | 45.6 |
| control: tanh, SGD 0.1, `[784,128,10]` | 96.44 | 0.37 | 0.138 | 28, 22, 19, 30, 31 | 14.9 |

- **Contra el control** (ganador de la etapa 0): +0.80 pp con umbral 2σ de 0.56 pp, una
  mejora que supera el umbral. Momentum con una tasa bien ajustada es la técnica que más
  aporta.
- **Contra el segundo:** +0.16 pp con umbral de 0.39 pp, no distinguible. 256 neuronas
  gana por el orden de la regla, pero 128 es equivalente con la mitad de parámetros y un
  cuarto del tiempo.
- **Cross-entropy:** el ganador tiene la mejor accuracy y la peor cross-entropy de los
  cuatro. La regla prioriza accuracy, que es la métrica del enunciado.

![Accuracy media y desvío de la confirmación](results/v2/analysis/stage_7_accuracy.png)

## Configuración seleccionada

`[784,256,10]`, tanh, salida softmax con cross-entropy, momentum (μ = 0.9) con tasa 0.1,
mini-batch de 32, inicialización automática (Xavier), hasta 100 épocas con parada temprana
(`config_id` `a2fa557ace07f890`). El candidato es el mejor checkpoint de la semilla 42:
época 28, accuracy de validación 97.07 % (CE 0.197) y accuracy de entrenamiento 100 %
(CE 0.0001). La brecha de cross-entropy entre entrenamiento y validación muestra
sobreajuste en confianza, aunque la accuracy de validación se mantiene. Si se reentrenara,
el protocolo fija 14 épocas (mediana de las mejores épocas de la confirmación).

![Curvas de entrenamiento y validación del candidato](results/v2/analysis/winner/learning_curves.png)

La cross-entropy de validación se estanca en torno a 0.197 desde la época 12, mientras la
de entrenamiento tiende a cero: la parada temprana elige la época 28 por mejoras mínimas
dentro de esa meseta.

Recall de validación del candidato por dígito: 0: 98.3 · 1: 98.8 · 2: 98.3 · 3: 93.5 ·
4: 96.6 · 5: 90.7 · 6: 98.3 · 7: 98.1 · 8: no evaluable · 9: 95.6 (macro-F1 96.25 %).

## Comprobaciones posteriores (apéndice)

No pueden cambiar la selección; revisan su solidez con más semillas
([reporte del apéndice](results/v2_appendix/report.md)).

- **ReLU en la configuración final** (cinco semillas): tanh 97.24 ± 0.15 % contra ReLU
  96.95 ± 0.69 %. La diferencia (0.29 pp) no supera el umbral 2σ (1.00 pp). ReLU tiene
  menor cross-entropy (0.138 contra 0.190) pero mucha más variación entre semillas
  (96.10 a 97.67 %): con la semilla 42 sola habría parecido mejor.
- **Etapas 3–5 con tres semillas:** en profundidad y lote se mantienen los ganadores.
  En ancho, 256 (97.19 %) y 128 (97.17 %) quedan empatados, lo que coincide con la
  selección final de 256. El ancho 512 y el lote 16 tienen desvíos de 2 pp o más,
  consistente con la inestabilidad a tasa 0.1.

![Ancho de la capa oculta con tres semillas](results/v2_appendix/analysis/stages_width.png)

## Evaluación final sobre `digits_test.csv`

Se evaluó una sola vez el candidato (SHA-256 verificado contra `selection.json`), sin
reentrenar y sin usar el test para ninguna decisión.

| Conjunto | Muestras | Accuracy (%) | Cross-entropy |
|---|---:|---:|---:|
| Test, todas las clases | 2497 | **86.70** | 1.749 |
| Test, sin el 8 | 2254 | **96.05** | 0.226 |
| Validación, mismo modelo | 2489 | 97.07 | 0.197 |
| Validación, media de 5 semillas | 2489 | 97.24 ± 0.15 | — |

- **El 8 explica la caída global.** `digits.csv` no tiene ningún 8: la red no puede
  predecirlo, y los 243 del test son errores (79 se asignan al 3, 43 al 5 y 41 al 9).
  Esto baja la precision del 3 al 71.6 %.
- **Sobre las clases vistas** el desempeño se sostiene: 96.05 % en test contra 97.07 % en
  validación.
- **El 5**, con sólo 217 ejemplos de entrenamiento, tiene el recall más bajo de las clases
  vistas (86.5 %), con 12 confusiones hacia el 3.

![Matriz de confusión sobre el test](results/v2/confusion_matrix.png)

![Recall por clase sobre el test](results/v2/per_class_recall.png)

Detalle completo: [evaluación final](results/v2/final_evaluation.md).

## Limitaciones

- La validación no contiene el 8 y tiene pocos 5; cada error de un 5 mueve su recall
  unos 1.85 pp.
- Elegir repetidamente sobre la misma validación puede dar una estimación optimista; la
  brecha de 1 pp con el test sin el 8 es coherente con eso.
- Mantener la tasa fija al cambiar arquitectura y lote es parte del diseño "un factor a
  la vez": no explora combinaciones nuevas, salvo las del cruce de la etapa 6.
- Los tiempos son de pared, con corridas en paralelo; sirven como orden de magnitud y no
  intervienen en la selección.
