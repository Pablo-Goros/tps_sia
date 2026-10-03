# Ejercicio 2 — Comparación y selección durante el desarrollo

Este análisis corresponde al paso 7 del [plan](../implementation-plan.md).
Responde las preguntas (a) y (b) del [enunciado](../Enunciado%20TP3.md)
con evidencia de desarrollo. La evaluación sobre `digits_test.csv`
corresponde al paso 9 y sigue reservada.

## (a) ¿Cómo se evalúa el desempeño del sistema?

Se mantiene una partición estratificada 80/20 con semilla de partición 42:
9960 imágenes para ajustar pesos y biases, y 2489 para comparar
hiperparámetros. Todas las comparaciones usan exactamente las mismas filas;
los índices, SHA-256 del CSV y huella de la partición quedan junto a cada
corrida. Cambiar la semilla del modelo cambia la inicialización y el orden
aleatorio de los lotes, conservando la partición.

La accuracy mide la proporción de etiquetas correctas mediante argmax sobre
las diez probabilidades de salida. La cross-entropy mide también la
probabilidad asignada a la etiqueta correcta y es el costo optimizado:

\[
L=-\frac{1}{N}\sum_{i=1}^{N}\sum_{k=0}^{9}y_{ik}\log p_{ik},
\qquad p_{ik}=\frac{e^{z_{ik}}}{\sum_j e^{z_{ij}}}.
\]

La implementación usa log-softmax estable y el delta de salida es
`(p-y)/N`, donde N es el tamaño real del lote. Cada gradiente es una media;
SGD actualiza `W ← W − eta * grad(W)` y análogamente los biases.
Los deltas ocultos propagan la sensibilidad de las capas posteriores y
multiplican por la derivada local de ReLU. El último lote incompleto usa
su propio tamaño, evitando alterar el significado de la tasa.

Se registran precision, recall y F1 por dígito, macro-F1 y balanced accuracy.
Estas dos últimas promedian únicamente las clases con soporte real y
permiten observar el dígito 5 pese a su baja frecuencia. La matriz de
confusión tiene etiquetas reales en filas y predicciones en columnas;
su versión normalizada divide cada fila por su soporte. La validación
contiene 54 ejemplos del 5 y ninguno del 8. Recall del 8 es no evaluable,
no cero: ninguna de estas métricas demuestra que la red reconozca el 8.
Una precision sin predicciones se registra como `null`.

Se comparan curvas de costo y accuracy de train y validación, época del
checkpoint elegido, número de parámetros y duración. Cada corrida tiene
un máximo de 30 épocas, epsilon 0 y ninguna parada por paciencia. El
checkpoint de desarrollo se elige por la menor cross-entropy de validación,
que puede ocurrir antes de la época 30. La confirmación informa media y
desvío poblacional sobre las semillas 42, 0 y 1; tres repeticiones no
constituyen un intervalo de confianza ni eliminan el sesgo de seleccionar
hiperparámetros repetidamente sobre una misma validación.

## (b) ¿Qué variantes se prueban para encontrar la solución?

El protocolo queda guardado antes de entrenar en
[protocol.json](results/protocol.json). Una corrida corta de dos épocas
estima el costo; sus métricas no participan en la selección. No se amplió
la grilla ni el presupuesto usando resultados del test.

1. Tasas y optimizadores: SGD y momentum con `1e-5`, `1e-4`, `1e-3`,
   `1e-2`; Adam con `1e-5`, `1e-4`, `1e-3`. Se mantienen arquitectura
   `[784,128,10]`, ReLU, lote 32 y semilla 42. El baseline es SGD `1e-2`.
2. Arquitecturas: `[784,64,10]`, `[784,128,10]`, `[784,256,10]` y
   `[784,128,64,10]`, con las mejores tasas de los dos optimizadores
   mejor ubicados. Se reutilizan las corridas equivalentes de 128 neuronas.
3. Confirmación: las tres mejores configuraciones del conjunto explorado,
   más el baseline como control, con las mismas tres semillas. Se reutiliza
   la primera semilla. La selección ordena por accuracy media descendente,
   cross-entropy media ascendente y número de parámetros ascendente;
   un identificador resuelve empates exactos de forma determinista.

Momentum usa `v ← 0.9*v − eta*g` y `W ← W+v`: acumula dirección de los
pasos anteriores. Adam usa medias exponenciales de gradientes y cuadrados,
corrección del sesgo inicial y normalización por la raíz del segundo
momento, con beta1 0.9, beta2 0.999 y epsilon del optimizador `1e-8`.
Las tasas se comparan dentro de cada mecanismo: la misma tasa numérica
no representa el mismo tamaño de paso efectivo. El presupuesto común
permite observar las diferencias dentro de 30 épocas; no garantiza que
cada configuración alcance su mejor solución posible ni que use el mismo
tiempo de cómputo.

ReLU permite una derivada no saturada para preactivaciones positivas;
para las negativas su derivada es cero. La salida softmax produce una
distribución compatible con objetivos one-hot y cross-entropy. La
inicialización `auto` usa He normal en las ocultas ReLU y Xavier normal
en la salida, con biases cero: las escalas dependen del tamaño de las
capas y los pesos aleatorios rompen la simetría entre neuronas. Los
píxeles originales están en `[0,1]`, por lo que se conserva el
preprocesamiento identidad. Los mini-batches de 32 promedian gradientes
y se mezclan reproduciblemente cada época; permiten más actualizaciones
que batch sin la variabilidad de actualizar tras cada imagen. Estas son
condiciones controladas del estudio, no conclusiones de comparaciones
que hayan variado activación, inicialización o tamaño de lote.

## Evidencia y selección

Las tablas completas y las figuras se regeneran desde archivos guardados
con el comando `search_analysis` documentado en el [README](README.md).
[comparison.md](results/analysis/comparison.md) reúne las tres etapas,
las métricas por clase del candidato y las curvas de entrenamiento.
Los resultados de cada corrida incluyen configuración, entorno, duración,
historia, pesos recargables y hashes.

Los tiempos de pared incluyen entrenamiento y checkpoints; se registraron
`OPENBLAS_NUM_THREADS=1` y `OMP_NUM_THREADS=1`. Algunas corridas se ejecutaron
simultáneamente, por lo que la contención del equipo impide interpretar
sus duraciones como una comparación aislada de eficiencia. Los tiempos
no intervienen en la regla de selección.

### Tasas pequeñas y evolución del aprendizaje

Con SGD, las tasas `1e-5`, `1e-4`, `1e-3` y `1e-2` alcanzan respectivamente
21,98 %, 74,53 %, 89,59 % y 94,38 % de accuracy en sus checkpoints de
validación. En las cuatro corridas el menor costo se obtiene en la época 30.
Las curvas de las tasas pequeñas todavía descienden: el resultado indica
convergencia lenta bajo este presupuesto, no incapacidad de esa tasa para
aprender con más actualizaciones. No se extendió su presupuesto porque el
objetivo de esta etapa es comparar soluciones dentro de un costo acotado.
Momentum `1e-5` obtiene 74,53 %, cercano a SGD `1e-4`: acumular pasos con
coeficiente 0.9 cambia la escala efectiva de actualización.

Adam `1e-3`, con la arquitectura de referencia, elige la época 9:
accuracy de train 99,73 %, de validación 96,63 %, costo de train 0,0203
frente a 0,1239 de validación. A la época 30 alcanza 100 % en train y
97,11 % en validación, pero su costo de validación sube a 0,1688. Hay
sobreajuste en cross-entropy: pese a clasificar más ejemplos correctamente,
el deterioro de las probabilidades de algunos errores aumenta el costo.
Por eso se distingue seleccionar por costo de elegir la máxima accuracy
observada en una época. El baseline termina con 95,97 % en train y
94,38 % en validación, y costo 0,1539 frente a 0,1950: sigue mejorando con
una brecha menor, aunque su desempeño dentro del presupuesto es inferior.
No basta esa brecha para afirmar que su arquitectura carezca de capacidad;
el estudio posterior separa arquitectura y velocidad de optimización.

Las fórmulas y convenciones descritas corresponden a la implementación de
[MLP](../shared/mlp.py), [optimizadores](../shared/optimizers.py) y
[métricas](../shared/metrics.py). Las configuraciones efectivas, incluyendo
inicialización por capa y reducción de los gradientes, se conservan en
los resultados; las curvas no se reconstruyen a partir de explicaciones.

### Arquitecturas bajo condiciones controladas

| Ocultas | Parámetros | Momentum `1e-2`: accuracy / CE | Adam `1e-3`: accuracy / CE |
|---|---:|---:|---:|
| 64 | 50 890 | 96,30 % / 0,1452 | 96,54 % / 0,1432 |
| 128 | 101 770 | 96,83 % / 0,1258 | 96,63 % / 0,1239 |
| 256 | 203 530 | 96,67 % / 0,1265 | 96,54 % / 0,1213 |
| 128, 64 | 109 386 | 96,54 % / 0,1281 | 96,71 % / 0,1261 |

La tabla usa exclusivamente semilla 42 y el checkpoint elegido por costo.
Duplicar el ancho de 128 a 256 aproximadamente duplica los parámetros y
no mejora la accuracy de ese checkpoint en ninguno de los dos optimizadores.
Con Adam, sí mejora su cross-entropy: arquitectura y criterio de desempeño
no se reducen a una única noción de «mejor». La profundidad `[128,64]`
aumenta los parámetros sólo un 7,5 % respecto de `[128]` y mejora la
accuracy con Adam, pero empeora con momentum. No hay evidencia de que
más capas o neuronas mejoren universalmente el resultado.

Las diferencias entre las tres mejores variantes son pequeñas: se
confirman momentum `[128]`, Adam `[128,64]` y momentum `[256]` con las
mismas semillas antes de decidir. El baseline SGD `[128]` se confirma
como control. La búsqueda no demuestra un óptimo global: las tasas de
arquitectura fueron elegidas primero en la red de referencia y podrían
cambiar al explorar otros presupuestos o arquitecturas.

Se eligió cross-entropy para optimizar la probabilidad de la clase correcta.
El costo cuadrático disponible en el núcleo también permite entrenar, pero
mide diferencias de probabilidades y tiene otra derivada. La combinación
softmax/cross-entropy usada aquí produce directamente el delta `(p-y)/N`;
no se aplica la derivada de una logística independiente a cada salida.
El backprop se calcula con operaciones matriciales explícitas, sin
frameworks de entrenamiento ni diferenciación automática.

### Confirmación y configuración congelada

| Configuración | Accuracy media ± desvío (pp) | CE media | Macro-F1 | Balanced accuracy | Épocas elegidas |
|---|---:|---:|---:|---:|---|
| momentum-0.01-hidden-256 | 96.81 % ± 0.24 | 0.1219 | 0.9624 | 0.9612 | [16, 17, 22] |
| momentum-0.01 | 96.68 % ± 0.11 | 0.1259 | 0.9595 | 0.9589 | [21, 20, 17] |
| adam-0.001-hidden-128-64 | 96.54 % ± 0.13 | 0.1288 | 0.9588 | 0.9554 | [4, 5, 7] |
| baseline | 94.46 % ± 0.09 | 0.1921 | 0.9347 | 0.9341 | [30, 30, 30] |

La configuración elegida es `[784,256,10]`, ReLU, salida softmax,
cross-entropy, momentum 0.9 con tasa 0.01, mini-batch de 32, shuffle,
inicialización automática He/Xavier y preprocesamiento identidad.
Tiene 203 530 parámetros. Su ventaja media sobre momentum con 128 neuronas
es 0,13 puntos porcentuales, con el doble de parámetros y mayor costo de
ejecución. La regla fijada prioriza accuracy y elige esta configuración;
la pequeña diferencia y tres semillas no demuestran superioridad estadística.
La red de 128 sigue siendo una referencia útil del compromiso entre
desempeño y tamaño. Frente al baseline, la mejora media es 2,36 puntos
porcentuales sobre esta misma validación.

El candidato usa la semilla predefinida 42 y el checkpoint de la época 16:
99,88 % de accuracy de train, 96,67 % de validación y costos 0,0187 y
0,1265. La brecha entre costos evidencia sobreajuste; continuar hasta la
época 30 reduce el costo de train a 0,0059 y aumenta el de validación a
0,1333. La selección del checkpoint conserva la mejor época por costo.

El dígito 5 obtiene precision 92,31 %, recall 88,89 % y F1 90,57 %:
48 aciertos de 54 imágenes. Los seis errores se predicen como 1 (uno),
3 (dos) y 9 (tres). Cada error representa 1,85 puntos de recall en esta
clase; el soporte reducido limita la precisión de la estimación.
El dígito 8 tiene soporte cero y métricas no evaluables. Macro-F1 y
balanced accuracy de la confirmación no lo incluyen.

La selección, evidencias, hashes y ruta del modelo se guardan en
[selection.json](results/selection.json). El candidato recargable está en
[best_model.npz](results/runs/0231e346d45648ae-seed-42/best_model.npz). La mediana de las mejores
épocas `[16,17,22]` es **17**: queda registrada como presupuesto de un
eventual reentrenamiento con toda la data de desarrollo en el paso 9.
El candidato preparado en este paso es el checkpoint de desarrollo.

![Comparación de tasas y optimizadores](results/analysis/rates.png)

![Comparación de arquitecturas](results/analysis/architectures.png)

![Confirmación y costo](results/analysis/confirmation.png)

## Comprobaciones y alcance

Se completaron las 26 corridas del protocolo: una corta de dos épocas,
11 de tasas, seis arquitecturas adicionales y ocho repeticiones de
confirmación. No hubo fallos numéricos. Todas conservaron el SHA-256 del
CSV y exactamente los mismos índices de train/validación. Se verificaron
los hashes de los modelos, el ranking de confirmación y la reproducción
exacta de las métricas de train y validación al recargar el candidato.

Las siete pruebas de `ej2/tests/test_search.py` pasan: selección por
media de semillas, grupos incompletos o duplicados, particiones distintas,
comparaciones controladas, ejecución con procesos, reanudación y
detección de modelos alterados. También pasaron las 18 comprobaciones
previas pertinentes de métricas y experimentos de `shared` y ej2.
Los gráficos y tablas se regeneraron desde resultados guardados.

Entorno de entrenamiento: Python 3.14.4, NumPy 2.5.0, Linux/WSL2,
un hilo de BLAS y OMP por corrida. La generación de figuras usó
Matplotlib 3.11.2. La evaluación final de generalización y el objetivo
de 98 % del ejercicio 3 quedan para sus pasos correspondientes.
