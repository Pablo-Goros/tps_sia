# Plan de implementación — TP3, ejercicios 2 y 3

Fecha: 2026-10-03. Alcance: continuar la implementación existente y producir experimentos, evaluación y análisis reproducibles para ambos ejercicios.

## Alcance vigente del ejercicio 3

El usuario acotó esta implementación a dejar disponibles los comandos y las
configuraciones para correr e iterar después. El código de controles, búsqueda por
etapas, corridas puntuales, estudio de factores y análisis está implementado; las
corridas académicas de ej3 no se ejecutaron.

Quedan fuera de este alcance la evaluación final en test, el reentrenamiento para
entrega, las conclusiones del informe y las extensiones opcionales, incluidas L2,
augmentation, ruido e interpretabilidad. Las propuestas históricas de los pasos 8
y 9 que aparecen más abajo deben leerse con esta restricción.

El flujo vigente se documenta en [ej3/README.md](ej3/README.md). La selección v2 de
ej2 debe recuperarse o elegirse otra referencia explícitamente; no se sustituye por
v1 en forma automática. Los chequeos nuevos usan datos y mediciones sintéticos,
con entrenamiento bloqueado, y no son evidencia de rendimiento académico.

## 1. Requisitos y punto de partida

La autoridad de los requisitos es el [enunciado oficial, OFF-010](../../sources/extracted/OFF-010.md), pp. 4–5, junto con su [versión local](Enunciado%20TP3.md). El ejercicio 2 requiere clasificar los diez dígitos y analizar variantes de tasa de aprendizaje, arquitectura y optimización. El ejercicio 3 requiere buscar una accuracy ≥ 98 % con los nuevos datos y explicar tanto las mejoras técnicas como los factores propios del conjunto de datos. `digits_test.csv` se reserva para generalización en ambos ejercicios.

El PDF llama al nuevo archivo `more_data_digits.csv`; el enunciado local y los archivos disponibles usan `data/more_digits.csv`. Implementar con este último nombre y registrar la correspondencia. Las pautas adicionales sobre trazabilidad, tasas pequeñas, checkpoints y evolución de pesos provienen de [AGENTS.md](AGENTS.md); las decisiones experimentales propuestas aquí no son requisitos adicionales del enunciado.

Estado comprobado mediante lectura del código y los resultados existentes:

| Componente | Estado | Acción pendiente |
|---|---|---|
| `ej2/src/datos_digitos.py` | Carga cacheada, one-hot de diez clases, partición estratificada 80/20 | Reutilizar y ampliar trazabilidad de índices y datasets |
| `ej2/src/mlp.py` | Forward/backprop matriciales, ReLU/tanh, softmax + cross-entropy, salida logística + costo cuadrático, online/batch/mini-batch | Ampliar entrenamiento, instrumentación y persistencia |
| `ej2/src/optimizadores.py` | SGD e interfaz independiente del MLP | Implementar momentum y Adam con estado persistente |
| `ej2/tests/` | Chequeos numéricos, XOR, lotes, aislamiento de validación, caché y reanudación SGD | Mantener y extender para las nuevas funcionalidades |
| `ej2/src/baseline.py`, `baseline.json`, `src/plots.py` | Baseline ejecutado y resultados guardados | Conservarlo como referencia; generalizar experimentos y análisis |
| `ej3/` | Sólo `.gitkeep` | Crear configuración, comandos, experimentos y análisis reutilizando el núcleo de `shared` |

El baseline existente usa `[784, 128, 10]`, ReLU, SGD, learning rate 0.01, lotes de 32 y 30 épocas. Sus resultados guardados son 95.97 % de accuracy de entrenamiento y 94.38 % de validación; son antecedentes, no resultados de una ejecución nueva. `digits.csv` contiene 12 449 muestras, ninguna del 8 y sólo 271 del 5, según [ej2/README.md](ej2/README.md). La validación actual no mide reconocimiento del 8 y su accuracy global puede ocultar problemas en el 5. El README principal está desactualizado respecto del baseline y deberá corregirse al cerrar la implementación.

## 2. Decisiones comunes

- Crear `tps_sia/tp3/shared/` como paquete Python para los componentes compartidos por los ejercicios. Migrar allí el MLP de NumPy, optimizadores, activaciones y loader existentes; `ej2` y `ej3` importarán ese núcleo común.
- Mantener las diez salidas y softmax con cross-entropy media por muestra. Documentar su diferencia con el costo cuadrático tradicional y el delta `(p - y) / batch_size`, la propagación hacia capas ocultas y la actualización con signo negativo.
- Los módulos nuevos de `shared` y las claves nuevas de configuración o resultados usarán inglés. Conservar las APIs existentes en español durante la migración y dejar módulos de compatibilidad que reexporten las implementaciones compartidas desde sus rutas anteriores. Mantener lectura del formato anterior o versionar explícitamente cualquier cambio.
- Separar ejecución, mediciones y análisis. Los gráficos deben poder regenerarse sin entrenar ni cargar el conjunto final de test.
- Seleccionar hiperparámetros, épocas y técnicas exclusivamente con datos de desarrollo: `digits.csv` en ej2 y `more_digits.csv` en ej3. Usar el nuevo archivo por sí solo como experimento principal de ej3; una unión con `digits.csv` requeriría un experimento separado, deduplicación y justificación.
- Fijar candidatos, presupuesto y criterio de selección antes de cada barrido. Seleccionar por accuracy de validación; desempatar por menor cross-entropy y luego menor número de parámetros. Confirmar los finalistas con las mismas semillas, inicialmente `[42, 0, 1]`, y comparar media y dispersión sin elegir la mejor semilla retrospectivamente.
- Congelar las decisiones de ambos ejercicios antes de abrir resultados de test. El objetivo de 98 % se verificará sobre la evaluación final de generalización; alcanzarlo en train o validación no basta. Si no se alcanza, registrar el resultado y la brecha sin volver a ajustar usando test.

### Organización del código compartido

`shared/` contendrá funciones, clases y métodos reutilizables, con una única implementación de cada componente. Cada ejercicio conservará sus comandos, configuraciones, datasets seleccionados, experimentos específicos e informes. La estructura objetivo es:

```text
tp3/
├── shared/
│   ├── __init__.py
│   ├── activations.py       # Activaciones y derivadas usadas por los modelos
│   ├── mlp.py               # Red, backprop, entrenamiento e historia
│   ├── optimizers.py        # SGD, momentum y Adam
│   ├── digit_dataset.py     # Carga de dígitos, caché y particiones
│   ├── metrics.py           # Métricas reutilizables
│   ├── experiments.py       # Runner y almacenamiento de resultados
│   ├── analysis.py          # Tablas y gráficos a partir de resultados
│   ├── evaluation.py        # Evaluación de modelos ya seleccionados
│   └── tests/               # Chequeos de los componentes compartidos
├── ej1/                     # Perceptrón simple y estudio de fraude
├── ej2/                     # Comandos, configs, resultados e informe de ej2
└── ej3/                     # Comandos, configs, resultados e informe de ej3
```

Reglas de organización:

- Los ejercicios dependen de `shared`; `shared` no importa código de `ej1`, `ej2` o `ej3`. Por ejemplo, mover las activaciones actualmente ubicadas en `ej1/src/activaciones.py` evita que el MLP común dependa del ejercicio de fraude.
- Usar importaciones de paquete, por ejemplo `from tps_sia.tp3.shared.mlp import MLP`, y comandos `python -m ...` desde la raíz. Evitar modificaciones de `sys.path` o depender del directorio de trabajo para resolver archivos.
- Pasar rutas de dataset, caché, configuración y resultados desde cada comando de ejercicio. El loader compartido no debe guardar automáticamente los datos de ej3 en `ej2/cache/`. Los valores por defecto del baseline se conservan en su comando o módulo de compatibilidad.
- El runner común recibe la configuración y produce resultados; cada ejercicio define sus barridos y comparaciones. Las funciones comunes de análisis reciben resultados guardados; las conclusiones y gráficos específicos permanecen en el ejercicio correspondiente.
- Compartir con ej1 únicamente componentes que ya sean reutilizables, inicialmente las activaciones. Mantener su perceptrón, procesamiento de fraude y análisis específicos en `ej1`; extraer otras utilidades cuando exista un uso concreto compartido.
- Conservar los datos originales y los resultados por ejercicio. `shared/` almacena código y sus pruebas, mientras que caches, checkpoints e informes se guardan en las ubicaciones de cada ejercicio.
- Los módulos antiguos que se conserven por compatibilidad sólo reexportan o delegan; no mantienen copias de la lógica. Mantener nombres de clases, métodos y formatos guardados evita mezclar esta reorganización con cambios matemáticos.

## 3. Secuencia de trabajo

Los pasos 1–3 descritos en `ej2/README.md` ya están implementados. La continuación comienza con la extracción a `shared`, antes del paso 4.

### Paso 3a — Extraer el núcleo común a `shared`

**Archivos:** crear `shared/__init__.py`, `shared/activations.py`, `shared/mlp.py`, `shared/optimizers.py`, `shared/digit_dataset.py` y `shared/tests/__init__.py`. Adaptar importaciones y módulos de compatibilidad en ej1/ej2.

1. Mover la implementación de `ej1/src/activaciones.py` a `shared/activations.py`, y las de `ej2/src/mlp.py`, `optimizadores.py` y `datos_digitos.py` a sus módulos compartidos. Preservar el algoritmo y las APIs públicas existentes.
2. Ajustar los imports internos para que el MLP y optimizadores dependan únicamente del paquete común. Actualizar los consumidores de ej1/ej2 y conservar las rutas anteriores mediante reexportaciones o wrappers delgados.
3. Separar los parámetros de ruta y los defaults específicos de ej2 del loader común. Conservar `cargar()` y el comando de exploración anteriores mediante un wrapper que proporcione los defaults de ej2.
4. Centralizar en `shared/tests/` los chequeos del núcleo que sirven a varios ejercicios, usando nombres de archivo en inglés. Conservar en cada ejercicio los chequeos de sus datasets y pipelines; los comandos antiguos de validación pueden delegar a los chequeos comunes.
5. Actualizar la documentación de importaciones y verificar el baseline sin alterar su configuración ni sus resultados de referencia. No reentrenar el baseline completo sólo por mover archivos.

**Aceptación:** los chequeos existentes siguen pasando, se carga el modelo SGD guardado y sus predicciones se conservan. Los imports nuevos y antiguos resuelven la misma implementación. `shared` no depende de los ejercicios, las rutas de caché/resultados siguen perteneciendo a cada ejercicio y los comandos existentes funcionan desde la raíz.

### Paso 4 — Momentum y Adam

**Archivos:** extender `shared/optimizers.py` y crear `shared/tests/test_optimizers.py`.

1. Implementar momentum clásico, explicitando la convención de velocidad, por ejemplo `v = momentum * v + gradient`, `parameter -= learning_rate * v`.
2. Implementar Adam con momentos de primer y segundo orden, contador de actualizaciones y corrección de sesgo. Usar como punto de partida `beta1=0.9`, `beta2=0.999`, `optimizer_epsilon=1e-8`; distinguir este epsilon del criterio de parada.
3. Extender la fábrica de optimizadores y la interfaz para exportar/restaurar estado. Guardar una velocidad o dos momentos por tensor, en el mismo orden que `MLP.parametros`.
4. Validar formas, configuración y valores finitos antes de modificar parámetros. Mantener SGD como referencia y aceptar su configuración previa.

**Aceptación:** primeros pasos verificables con cálculos manuales, actualización correcta de todos los pesos y biases, rechazo de gradientes inválidos sin modificaciones parciales y construcción/restauración de los tres optimizadores. Los chequeos de gradientes y XOR existentes deben seguir pasando.

### Paso 5 — Control del entrenamiento y checkpoints completos

**Archivos:** extender `shared/mlp.py`; crear `shared/tests/test_training_state.py`.

1. Añadir contadores acumulados de épocas y actualizaciones, historia acumulativa y motivo de parada (`max_epochs`, `training_epsilon`, `early_stopping`, `interrupted` o fallo numérico).
2. Incorporar parada temprana configurable por validación con `patience`, `min_delta` y criterio explícito. Propuesta inicial: vigilar cross-entropy de validación, conservar la mejor época y seleccionarla sólo con validación. Registrar también accuracy; no confundir la mejor época con la última ejecutada.
3. Separar `best_model.npz`, destinado a inferencia/evaluación, de `checkpoint.npz`, destinado a reanudar el último estado. Un checkpoint de la mejor época, si se permite reanudarlo, debe restaurar también el estado correspondiente del optimizador y RNG.
4. Guardar configuración completa, pesos/biases, estado del optimizador, RNG, contadores, historia, estado de parada temprana y preprocesamiento. Versionar el formato y conservar la carga de modelos SGD versión 1.
5. Guardar checkpoints periódicos al terminar una época y permitir pausa/reanudación en ese límite. Usar escritura temporal y reemplazo del checkpoint terminado; documentar que una interrupción a mitad de época puede requerir repetirla.
6. Registrar learning rate por época. Usar inicialmente tasa constante; añadir un schedule sólo si la comparación de validación lo justifica y persistir su estado.
7. Añadir registro configurable de pesos seleccionados y normas por capa de pesos, gradientes y actualizaciones, con índices globales de época/update. Guardarlo en un archivo aparte para evitar historiales excesivos.
8. Hacer explícitos shuffling, inicialización y escala efectiva por capa, función de costo y epsilon de parada. Los píxeles del baseline se usan en `[0,1]`; cualquier transformación adicional se ajusta sólo con train y se guarda junto al modelo.

**Aceptación:** entrenar N épocas de corrido equivale, en parámetros, estado e historia numérica, a entrenar K, guardar/cargar y continuar N−K para SGD, momentum y Adam; excluir tiempos de esa comparación. Validación no altera parámetros ni RNG. La mejor época se restaura correctamente, los motivos de parada son verificables y las historias conservan índices sin reiniciarse.

### Paso 6 — Experimentos y métricas reutilizables

**Archivos nuevos:** `shared/experiments.py`, `shared/metrics.py`, `shared/analysis.py`, `shared/tests/test_metrics.py`, `shared/tests/test_experiments.py`, `ej2/src/experiments.py`, `ej2/src/analysis.py`, `ej2/configs/search.json`, `ej2/tests/test_experiments.py`. Extender `shared/digit_dataset.py` y adaptar gráficos existentes cuando corresponda. Los comandos de ej2 delegan en las funciones compartidas.

1. Extraer del baseline las funciones reutilizables a `shared/experiments.py` sin perder su comando ni configuración de referencia. Llevar la matriz de confusión a `shared/metrics.py`, manteniendo compatibilidad con su importación previa. Centralizar gráficos reutilizables en `shared/analysis.py`; conservar en ej2 la presentación y selección de resultados específicas del ejercicio.
2. Definir configuración validada con dataset, arquitectura, activación y parámetros, salida/loss, optimizador y sus ajustes, learning rate, estrategia/batch size, shuffle, inicialización, semillas, límite de épocas, stopping, preprocesamiento y nivel de registro de pesos.
3. Permitir obtener y guardar índices de train/validación, SHA-256 del CSV, conteos por clase y huella de la partición. Mantener el comportamiento por defecto 80/20 del loader.
4. Crear un directorio por ejecución con identificador estable de configuración y semilla. No sobrescribir resultados terminados; distinguir una reanudación de una nueva corrida inicializada desde pesos existentes.
5. Guardar `results.json`, `history.csv`, modelos/checkpoints y metadatos de entorno, duración, número de parámetros, época elegida y motivo de parada. Registrar los fallos numéricos como corridas fallidas, sin incluirlos entre modelos exitosos.
6. Calcular cross-entropy, accuracy, precision/recall/F1 por clase, macro-F1, balanced accuracy y matriz de confusión 10×10. Incluir soporte por clase y política explícita para denominadores cero: recall sin muestras reales es no evaluable; precision sin predicciones es no definida. Identificar qué clases entran en cada promedio y guardar valores no definidos como `null`, no como `NaN` en JSON.
7. Generar curvas de train/validación, tablas de comparación, confusión en conteos y normalizada por filas, y evolución de pesos/actualizaciones cuando se haya registrado. Mostrar costo computacional y dispersión entre semillas junto al desempeño.

**Aceptación:** una ejecución pequeña genera todos los artefactos, se puede recargar el modelo y reproducir sus métricas, y el análisis funciona desde archivos guardados. El runner de desarrollo rechaza `digits_test.csv`; los tests verifican ese aislamiento y las métricas de clases ausentes con datos sintéticos.

### Paso 7 — Completar el ejercicio 2

**Datos:** train y validación exclusivamente de `digits.csv`, con partición fija durante cada comparación.

Barrido inicial propuesto; confirmar duración con una corrida corta antes de lanzar la serie:

| Etapa | Variantes | Variables controladas |
|---|---|---|
| Tasa y optimizador | SGD y momentum: `1e-5`, `1e-4`, `1e-3`, `1e-2`; Adam: `1e-5`, `1e-4`, `1e-3` | `[784,128,10]`, ReLU, batch 32, mismo split/semilla y presupuesto máximo de 30 épocas; momentum 0.9 |
| Arquitectura | `[784,64,10]`, `[784,128,10]`, `[784,256,10]`, `[784,128,64,10]` | Configuraciones de tasa/optimizador prometedoras de la etapa anterior |
| Confirmación | Mejores 2–3 configuraciones y baseline | Tres semillas comunes; mismo criterio de parada y presupuesto |

El primer barrido contiene 11 combinaciones de tasa/optimizador; el baseline coincide con una de ellas si se conservan todas sus condiciones. No lanzar automáticamente el producto cartesiano de todos los hiperparámetros. Las tasas pequeñas deben analizarse como posibles casos de convergencia lenta, no descartarse sin observar sus curvas. Ampliar el presupuesto o la grilla sólo con evidencia de validación, dejando registrada la decisión.

1. Comparar optimizadores con una tasa razonable para cada uno y condiciones equivalentes; aclarar que igual cantidad de épocas no garantiza igual tiempo ni convergencia.
2. Analizar train vs. validación, sobreajuste/subajuste, velocidad y costo. Justificar arquitectura, activación, inicialización y estrategia de lotes.
3. Evaluar desempeño por clase, especialmente 5 y 8. No afirmar que el sistema reconoce el 8 basándose en esta validación.
4. Guardar `ej2/results/selection.json` con la regla de selección, configuración final, semillas, época o presupuesto de reentrenamiento y evidencia. Preparar el candidato final sin consultar test.

**Aceptación:** hay comparaciones registradas de las tres dimensiones mínimas, resultados reproducibles y respuestas respaldadas a las preguntas (a) y (b). No se exige alcanzar 98 % en este ejercicio.

### Paso 8 — Implementar el ejercicio 3 y explicar la mejora

**Archivos nuevos:** `ej3/README.md`, `ej3/configs/search.json`, `ej3/src/__init__.py`, `ej3/src/experiments.py`, `ej3/src/analysis.py`, `ej3/tests/test_dataset.py`. Los comandos de ej3 reutilizarán el runner, MLP, optimizadores, loader, métricas y análisis de `shared`, sin importar implementaciones internas de ej2.

1. Explorar `more_digits.csv`: cantidad de muestras, clases, rangos, valores no finitos, ejemplos y duplicados/conflictos de etiqueta. No suponer que contiene más muestras, está balanceado o incluye el 8 antes de verificarlo.
2. Compararlo con `digits.csv`, registrar solapamiento y distinguir nuevos ejemplos de ejemplos repetidos. Si hay imágenes idénticas, agruparlas para evitar que crucen train/validación; documentar conflictos y cualquier exclusión sin modificar los CSV originales. La inspección de solapamiento con test será una comprobación de integridad separada, nunca un criterio para ajustar el modelo.
3. Crear una partición de desarrollo estratificada y reproducible, guardar sus índices y mantenerla fija. Pasar al loader compartido las rutas de `more_digits.csv` y de un caché propio dentro de `ej3/cache/`.
4. Reentrenar desde cero, sobre los nuevos datos, el baseline original y la configuración seleccionada en ej2. Separar así el cambio de dataset del cambio de técnica; no empezar exclusivamente desde un modelo previamente entrenado.
5. Buscar mejoras graduales con validación: momentum/Adam, tasas, arquitecturas `[784,128,10]`, `[784,256,10]`, `[784,256,128,10]`, batches 32/64/128 y parada temprana. Empezar con las mejores tasas anteriores, incluir `1e-5`/`1e-4` como referencia y confirmar finalistas con tres semillas comunes.
6. Si persiste una brecha, ampliar primero el presupuesto de épocas cuando las curvas sigan mejorando. Evaluar L2 o augmentation geométrica moderada sólo si los resultados justifican su costo; cada técnica necesita configuración, derivación/backprop cuando aplique y comparación controlada. Augmentation usa únicamente train; validación y test conservan sus imágenes originales.
7. Realizar comparaciones controladas en una validación común de `more_digits.csv`: misma configuración con subconjuntos de train de distintos tamaños, y distintos modelos sobre el mismo train. Mantener cobertura de clases en el estudio de tamaño. Si se compara directamente con `digits.csv`, excluir de su entrenamiento cualquier imagen de esa validación y registrar las muestras elegibles.
8. Comparar cobertura de clases y distribución además del tamaño. Una mejora global entre validaciones distintas no prueba por sí sola que la técnica sea mejor; discutir factores de datos y limitaciones de las comparaciones.
9. Guardar `ej3/results/selection.json` y congelar la configuración final. El resultado ≥ 98 % sigue siendo un objetivo por verificar en generalización, no una garantía del plan.

**Aceptación:** ejecución completa de ej3, evidencia de búsqueda y comparaciones que permitan responder las preguntas (a), (b) y (c), con referencias a resultados y sin seleccionar mediante test.

### Paso 9 — Evaluación final y entrega

**Archivos nuevos:** `shared/evaluation.py`, `ej2/src/evaluate.py`, `ej3/src/evaluate.py`, `ej2/report.md`, `ej3/report.md`. Los comandos de evaluación delegan en el evaluador común con rutas y modelos específicos de cada ejercicio. Actualizar `ej2/README.md`, crear/completar `ej3/README.md` y corregir `tp3/README.md`.

1. Elegir antes de evaluar si se entregará el mejor checkpoint de desarrollo o un modelo reentrenado con todo el dataset permitido. Propuesta principal: reentrenar desde cero con todos los datos de desarrollo, configuración congelada, semilla predefinida 42 y cantidad de épocas determinada por las corridas de validación, por ejemplo la mediana de sus mejores épocas. No consultar test para decidir duración ni semilla.
2. El evaluador carga los modelos finales y su preprocesamiento sin entrenar. Evaluar ambos sobre `digits_test.csv`, con diez clases, y guardar accuracy, cross-entropy, métricas por clase, confusión, tamaño de test, errores/aciertos y huellas de modelos/datos.
3. Para ej3, calcular y registrar `accuracy >= 0.98` usando el valor sin redondear. Si no se cumple, informar el mejor resultado del protocolo y la brecha; no abrir nuevos barridos guiados por esos errores de test.
4. Comparar resultados finales de ambos ejercicios. Los resultados de los modelos de control pueden evaluarse también si fueron predefinidos; no escoger el modelo entregado a partir de su ranking de test.
5. Completar informes en español con tablas y gráficos, respuestas a las preguntas, justificación matemática de costos y actualizaciones, evidencia de selección y limitaciones por distribución/cobertura.
6. Documentar dependencias y comandos desde la raíz para validación, entrenamiento, reanudación, análisis y evaluación. Los comandos nuevos deben especificar configuración y directorio de salida. Revisar que las instrucciones de ej1 no sigan apuntando a rutas anteriores a su reorganización.

**Aceptación:** modelos recargables, evaluación final trazable, respuestas completas y gráficos regenerables desde resultados. Distinguir implementación terminada de objetivo de accuracy alcanzado: el informe debe mostrar ambos estados.

## 4. Verificación y orden de ejecución

1. Antes de modificar el núcleo, ejecutar los chequeos existentes de MLP/XOR y loader como referencia:

   ```powershell
   python -m tps_sia.tp3.ej2.tests.test_validacion
   python -m tps_sia.tp3.ej2.tests.test_datos_digitos
   ```

   El segundo comando existente inspecciona test para formato y solapamiento; esa comprobación no selecciona hiperparámetros. Mantenerla separada del runner de desarrollo.

2. Tras el paso 3a, verificar imports, comandos, carga de modelos y chequeos existentes. Tras los pasos 4–6, ejecutar los nuevos chequeos de `shared/tests/` para optimizadores, reanudación, stopping, métricas e aislamiento del test, además de los chequeos específicos de cada ejercicio; repetir los existentes que cubren el núcleo modificado.
3. Hacer una corrida breve con datos sintéticos y otra de pocas épocas con datos de desarrollo; comprobar artefactos, recarga y regeneración de gráficos antes de gastar tiempo en barridos.
4. Ejecutar paso 7, luego paso 8, congelar las selecciones y terminar con paso 9. Registrar tiempos para dimensionar los barridos; si OpenBLAS necesita un límite de hilos, registrar el ajuste usado.

Entregar cada paso con código, comprobaciones pertinentes y documentación actualizada. Los pasos 3a, 4, 5, 6 y 7 están verificados; el siguiente bloque concreto es el paso 8. No volver a implementar los pasos 1–3 ni lanzar los barridos antes de que los checkpoints y el control de entrenamiento estén verificados.

## 5. Extensiones posteriores

Una vez completos los requisitos y su análisis, considerar el opcional de ruido gaussiano con modelo congelado: varias desviaciones estándar, media, semilla, escala de píxeles y clipping documentados, curvas de accuracy por nivel y degradación por clase. La interpretabilidad por atribuciones es otra extensión independiente. Ninguna de estas extensiones reemplaza las comparaciones y respuestas obligatorias.

## 6. Seguimiento

- [x] Inspeccionar implementación y resultados existentes de ej2.
- [x] Identificar requisitos, estado de ej3 y restricciones de evaluación.
- [x] Paso 3a: crear `shared` y migrar el núcleo reutilizable preservando compatibilidad.
- [x] Paso 4: momentum y Adam.
- [x] Paso 5: control del entrenamiento y checkpoints completos.
- [x] Paso 6: runner, métricas y análisis reutilizables.
- [x] Paso 7: comparaciones y selección del ejercicio 2.
- [x] Paso 7b: protocolo v2 de un factor a la vez para el ejercicio 2 — implementado y ejecutado (58 corridas; selección en `ej2/results/v2/selection.json`).
- [x] Paso 7c: apéndice de la búsqueda v2 (ReLU en la configuración final; etapas 3–5 con tres semillas) y evaluación final del ejercicio 2 sobre `digits_test.csv`.
- [x] Código obligatorio del paso 8: datos, controles, búsqueda, selección de desarrollo y estudio de factores implementados. Corridas académicas pendientes de ejecución por el usuario; extensiones opcionales excluidas.
- [ ] Paso 9: evaluación final e informes — fuera del alcance vigente de esta implementación.
- [ ] Extensiones opcionales, después de completar lo obligatorio.

Verificación del paso 3a (2026-10-03): chequeos de MLP/XOR, derivadas,
loader y compatibilidad aprobados; también los chequeos específicos de
ej1 y de los CSV de ej2. El modelo SGD versión 1 conserva exactamente
sus predicciones sobre las 2489 muestras de validación, junto con las
particiones, loss y confusión originales. Los SHA-256 del baseline y su
configuración permanecen iguales. Una ejecución de una época, con
configuración y salidas temporales, verificó el comando de entrenamiento,
los artefactos, su recarga y la regeneración de gráficos; no reemplaza
la ejecución de referencia ni constituye un nuevo experimento de selección.

Verificación del paso 4 (2026-10-03): momentum clásico y Adam implementados
en `shared/optimizers.py`, con fábrica y exportación/restauración de estado.
Las 13 pruebas de `shared/tests/test_optimizers.py` verifican cálculos manuales,
todos los pesos/biases, validación atómica, momentos y contador, reanudación
exacta, compatibilidad y XOR. Los chequeos anteriores de gradientes, XOR y
compatibilidad siguen pasando. `MLP` guarda formato versión 2 para preservar
el estado de los tres optimizadores y conserva la lectura de SGD versión 1;
la última historia sigue siendo por llamada, a completar en el paso 5.
El modelo SGD de referencia conserva su loss y confusión de validación y los
archivos del baseline permanecen intactos; no se reentrenó ni se hicieron
barridos o selección usando test.


Verificación del paso 5 (2026-10-03): `shared/mlp.py` incorpora historia y
contadores acumulados, tasa constante registrada por época, motivos de parada,
selección/parada por validación, checkpoint del último estado y exportación de
la mejor época con su propio optimizador/RNG. El formato NPZ versión 3 guarda
configuración, preprocesamiento, estado y registro opcional de pesos en JSONL;
conserva lectura y migración de versiones 1/2. Guardado temporal y reemplazo
atómico, checkpoints periódicos, pausa por época y rollback de épocas incompletas
verificados. Las 9 pruebas de `test_training_state.py` y las 13 de optimizadores
pasan, junto a gradientes/XOR del MLP y compatibilidad de ej2. La equivalencia
N frente a K + guardado/carga + N−K es exacta para parámetros, momentos, RNG,
selección e historia numérica de SGD, momentum y Adam, excluyendo tiempos.
Una corrida de tres épocas con 256 muestras de train y 96 de validación de
`digits.csv` verificó artefactos, recarga y lectura de la historia guardada;
el baseline SGD versión 1 conserva loss/confusión y todos sus archivos de
referencia mantienen SHA-256. No se ejecutaron barridos ni se abrió test.
Los gráficos del smoke no se regeneraron porque Matplotlib no está instalado
en este entorno; el análisis reutilizable sigue pendiente del paso 6.


Verificación del paso 6 (2026-10-03): `shared/experiments.py`, `metrics.py`
y `analysis.py` implementan configuración validada, corridas identificadas por
configuración/semilla, separación entre reanudación e inicialización desde pesos,
SHA-256 del CSV y partición, índices guardados, conteos, modelos/checkpoints,
historia CSV, metadatos de entorno y selección por validación. El loader conserva
exactamente su partición 80/20 y puede devolver índices y otra fracción explícita.
El runner rechaza `digits_test.csv`, también por symlink, antes de cargarlo;
registra fallos numéricos y excluye corridas fallidas/interrumpidas del ranking.
Las métricas incluyen políticas explícitas de denominadores cero y promedios
sobre clases con soporte, con `null` en JSON. El análisis genera curvas,
confusión en conteos/por filas, evolución opcional de pesos y tablas con tiempo,
parámetros y dispersión por semilla, sin cargar datos ni entrenar.

Los comandos de ej2 delegan en shared y `configs/search.json` predefine los
11 candidatos de tasa/optimizador y tres semillas; una invocación ejecuta sólo
un candidato/semilla salvo `--all` explícito. Se conserva la búsqueda completa,
su regla y su hash en los metadatos. El baseline reutiliza construcción del
modelo, hash, exportación de historia, confusión y gráficos comunes, conservando
su comando, configuración de referencia e importaciones anteriores.

Las 18 pruebas de métricas/runner/comandos, más las 22 de checkpoints y
optimizadores, pasan (40 en total), incluidas recarga y métricas exactas,
reanudación para SGD/momentum/Adam, clases ausentes, fallos y las cinco figuras.
También pasan gradientes/XOR, activaciones, loader sintético y compatibilidad.
Una corrida de dos épocas sobre las 9960 muestras de train y 2489 de validación
de `digits.csv`, pausada tras la primera y reanudada, verificó artefactos,
recarga y análisis por API y comando. Duración de entrenamiento ≈ 3.65 s con
OpenBLAS limitado a un hilo; accuracy de validación 87.91 %, usada sólo como
smoke técnico, sin selección. Todos los archivos de referencia del baseline
conservan SHA-256. Matplotlib se instaló en `/tmp/tp3-step6-deps` para comprobar
figuras, sin cambiar dependencias globales. No se ejecutaron barridos ni se
leyó `digits_test.csv`; pasos 7–9 permanecen pendientes.

Verificación del paso 7 (2026-10-03): `ej2/src/search.py` ejecuta el protocolo
por etapas y `search_analysis.py` regenera las comparaciones desde artefactos.
Se completaron 26 corridas: piloto de dos épocas, 11 combinaciones de
tasa/optimizador, seis arquitecturas adicionales y ocho confirmaciones.
Se mantuvieron los mismos índices y huellas de `digits.csv`, sin leer test.
Se seleccionó `[784,256,10]`, ReLU y momentum 0.9 con tasa 0.01, con
accuracy de validación media 96.81 % y desvío poblacional 0.24 puntos
porcentuales entre semillas 42, 0 y 1. `ej2/results/selection.json` congela
configuración, evidencias y hashes; el candidato de semilla 42 usa el mejor
checkpoint de la época 16, y la mediana `[16,17,22]` fija 17 épocas para
un eventual reentrenamiento del paso 9. Recarga y métricas de train y
validación reproducidas exactamente. Las siete pruebas del flujo de búsqueda
pasan, además de las 18 comprobaciones pertinentes de métricas/experimentos.
Tablas, curvas, confusión y dispersión regeneradas. El
[informe de desarrollo](ej2/development-report.md) responde (a) y (b),
analiza convergencia lenta y sobreajuste, compara capacidad y costo y
explicita soporte del 5 y ausencia del 8. Pasos 8 y 9 pendientes.

Paso 7b (2026-10-03, implementado, sin ejecutar): la búsqueda del paso 7 queda
como antecedente v1, sin modificar sus 26 corridas ni su `selection.json`. La v2
rehace la selección del ejercicio 2 con un protocolo de un factor a la vez
pre-registrado en `ej2/configs/search_v2.json` (SHA-256 en los metadatos de cada
corrida, calculado con fines de línea LF): activación, tasa por optimizador con
regla de borde 1-3-10, optimizador, ancho, profundidad, lote, cruce chico con tres
semillas y confirmación con cinco semillas contra el control de la etapa 0. Todas
las corridas tienen hasta 100 épocas con parada temprana por cross-entropy de
validación (patience 10) y checkpoint de la mejor época; agotar el presupuesto se
informa como "no convergió". Reglas de empate (activación → tanh; empate cercano
< 0.3 pp → semillas 0 y 1) y de mejora (ventaja > 2σ agrupado entre semillas)
fijadas antes de correr.

`shared/optimizers.py` agrega RMSProp con estado atómico y persistente; los tests
verifican dos pasos calculados a mano con una fórmula independiente, todos los pesos
y biases, gradientes inválidos, ida y vuelta de configuración, reanudación exacta y
entrenamiento de un MLP. `ej2/src/staged_search.py` ejecuta una etapa por invocación,
lee las decisiones previas, reutiliza corridas idénticas y admite procesos paralelos.
`shared/experiments.py` acepta `path_root`: en v2, dataset, caché y directorio de
corrida se guardan relativos a `tp3/` y `config_id` no depende de rutas absolutas
(sin `path_root` se conserva el id histórico). `search_analysis.py --protocol v2`
genera tablas, barras con desvío entre semillas, curvas, extensiones de borde y
métricas por clase del ganador desde artefactos guardados.

`ej2/tests/test_staged_search.py` cubre las reglas con resultados sintéticos, el flujo
completo hasta `selection.json`, la portabilidad del id, el rechazo de `digits_test.csv`
(también por enlace) y un smoke test real de las etapas 0 y 1 (2 épocas, 256 muestras
de train) en un directorio temporal. Tiempo medido por época con un hilo: 0.52 s para
SGD `[784,128,10]` y 2.65 s para `[784,512,10]`, lote 32, incluyendo checkpoints;
estimación de la búsqueda completa: 1.5–3.5 h en serie, 0.5–1 h con cuatro procesos.
No se leyó `digits_test.csv`.

Paso 8, parte de datos (2026-10-03, puntos 1–3; sin entrenar ni buscar):
`ej3/src/data_exploration.py` genera `ej3/results/data_report.{json,md}`,
`split_indices.npz` y `split_manifest.json` usando el loader compartido con caché
en `ej3/cache/` (excluido de Git). `more_digits.csv` tiene 15 741 muestras, las
diez clases (585 del 8 y 542 del 5), píxeles en [0, 1] sin valores no finitos, sin
duplicados internos ni conflictos de etiqueta. 3689 filas son idénticas a imágenes
de `digits.csv`, todas con la misma etiqueta; 12 052 son nuevas y `digits.csv` no
es subconjunto. La partición agrupa imágenes idénticas, es estratificada 80/20 con
semilla 42 (train 12 594, validación 3147, sin exclusiones) y, al no haber
duplicados, coincide con `shared.digit_dataset.particionar`, que usa el runner.
La comprobación de integridad cuenta 0 imágenes de `digits_test.csv` en
`more_digits.csv` y 0 en `digits.csv`; sólo se registran conteos.
`ej3/tests/test_dataset.py` cubre la partición y las salidas con datos sintéticos.

Paso 7c (2026-10-03): apéndice y evaluación final del ejercicio 2. `ej2/src/appendix.py`
ejecuta dos comprobaciones posteriores que no pueden cambiar la selección y reutilizan por
`config_id` las corridas de `results/v2/runs` sin escribir allí: (A) ReLU en la
configuración seleccionada con cinco semillas, comparada por la regla de la etapa 7, y
(B) los factores de las etapas 3–5 con semillas 0, 1 y 42. `ej2/src/final_evaluation.py`
evaluó una sola vez el checkpoint candidato (SHA-256 verificado) sobre `digits_test.csv`,
informando juntas la accuracy global y la accuracy sin el 8; se niega a repetir la
evaluación sin `--overwrite`. Ningún archivo de la búsqueda v2 se modificó. El paso 9
(informes y entrega de ambos ejercicios) sigue abierto.

Implementación del flujo obligatorio de ej3 (2026-10-03):
`shared/staged_search.py` reúne reglas, materialización y ejecución de corridas,
con reexports que preservan el comando de ej2. El runner admite índices explícitos
y validación de otro archivo, verifica solapamiento de imágenes y registra fuentes.
`ej3/configs/search.json`, `src/protocol.py` y `src/experiments.py` preparan controles,
tasas, arquitecturas, lotes, confirmación y corridas puntuales. `src/factor_study.py`
prepara comparaciones de dataset, cobertura del 8, tamaño y configuración sobre
una validación común. `src/analysis.py` produce tablas y figuras desde JSON guardados,
sin cargar datasets ni modelos. Se verificaron reglas y compatibilidad de ej2,
y el flujo de ej3 con artefactos sintéticos y entrenamiento bloqueado. No se
entrenó ningún modelo real ni se leyó test en esta implementación.
