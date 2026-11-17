# TP5 · Guion narrativo del punto 1.a

Referencia inicial: **`C:\Users\Pablo\Downloads\TP5 (1).pptx`**, slides originales 1–11. Última revisión de estilo: **`TP5 (2).pptx`**. Esta guía reemplaza la propuesta anterior; no modifica el PowerPoint. La propuesta ahora ocupa **13 diapositivas**: el todos contra todos tiene dos slides propias y el cambio de RMSProp a Adam tiene otra antes de la configuración final.

Hilo conductor: **búsqueda amplia → candidato poco consistente → ajustar LR no alcanza → revisar la señal de aprendizaje → BCE permite cumplir → configuración final → reconstrucciones y latente.**

## 1–4 · Presentación del problema

**1. Portada:** conservar.

**2. Separador Dataset:** conservar; quitar «El CEITBA como organización social» y «03 · 9», restos de la plantilla.

**3. Nuestro dataset inicial: font.h:** conservar la imagen del dataset. Texto:

> 32 caracteres binarios
>
> 7 filas × 5 columnas → 35 entradas
>
> Objetivo: comprimir a 2 dimensiones y reconstruir cada carácter con **como máximo 1 píxel incorrecto**.

Para decir: “Convertimos cada imagen en un vector de ceros y unos. El autoencoder busca reconstruir ese mismo vector pasando por un cuello de botella de dos neuronas. Queremos aprender los 32 patrones, sin reducir el dataset.”

Corregir «menos de 1 píxel»: exigiría cero errores. El requisito es **≤ 1 por carácter**. Son 26 minúsculas y seis símbolos, no 32 letras.

**4. Separador Autoencoder básico:** conservar, con tilde en «básico» y sin restos de CEITBA. Transición: “Primero buscamos qué combinación de arquitectura y optimizador podía resolver el problema.”

## 5 · El todos contra todos: qué probamos

**Correcciones detectadas en `TP5 (2).pptx`:** la segunda arquitectura debe terminar en **35**, no 45; y en la cuenta final debe decir **3 tasas**, no “3 tazas”.

**Imagen nueva:** [story-search-design.png](assets/story-search-design.png), también disponible en SVG. Es un esquema del experimento: a la izquierda las cinco arquitecturas y a la derecha los cuatro optimizadores con sus tres tasas. Mostrarlo grande; reemplaza el gráfico de resultados en esta slide.

**Texto para copiar:**

> 5 arquitecturas × 4 optimizadores × 3 tasas × 3 semillas
>
> **60 configuraciones · 180 corridas**
>
> Mismos datos, mismo objetivo y mismo presupuesto por corrida.

**Contenido del gráfico, si preferís armarlo con elementos editables:**

| Arquitectura | Capas completas |
| --- | --- |
| Base | 35 → 20 → 2 → 20 → 35 |
| Más ancha | 35 → 40 → 2 → 40 → 35 |
| Todavía más ancha | 35 → 60 → 2 → 60 → 35 |
| Más profunda | 35 → 30 → 15 → 2 → 15 → 30 → 35 |
| Mayor profundidad | 35 → 40 → 20 → 10 → 2 → 10 → 20 → 40 → 35 |

| Optimizador | Tasas de aprendizaje probadas |
| --- | --- |
| Adam | 0,0005 · 0,001 · 0,002 |
| RMSProp | 0,0005 · 0,001 · 0,002 |
| Momentum | 0,02 · 0,05 · 0,1 |
| SGD | 0,01 · 0,03 · 0,05 |

**Guion oral:** “En lugar de elegir una arquitectura de entrada, armamos un todos contra todos. Partimos de una red con una capa intermedia de veinte neuronas. Primero probamos darle más ancho, con cuarenta y sesenta; después agregamos profundidad, usando dos y tres capas por lado. En todas mantuvimos las treinta y cinco entradas, el cuello de botella de dos y las treinta y cinco salidas. Cada arquitectura la cruzamos con SGD, Momentum, RMSProp y Adam. Para cada optimizador usamos tres tasas: estas son las que aparecen en la tabla. Eso nos dio sesenta configuraciones, y repetimos cada una con las semillas cuarenta y dos, cuarenta y tres y cuarenta y cuatro: ciento ochenta corridas en total.”

**Continuación oral:** “La comparación usó siempre los mismos treinta y dos patrones, pérdida cuadrática, tanh en las capas internas y sigmoide en la salida. Cada corrida tenía hasta tres mil épocas y podía detenerse antes si todos los caracteres quedaban con como máximo un píxel incorrecto. Las tres semillas nos permitían ver si una combinación funcionaba con distintas inicializaciones.”

**Cómo mostrarlo:** primero señalar las arquitecturas, luego los optimizadores y sus tasas, y cerrar con la cuenta 5 × 4 × 3 × 3. Si usás animación, revelar esas partes en ese orden; no hace falta leer todos los tamaños en voz alta.

**Footer recomendado, en dos líneas cortas, alineadas a la izquierda y con etiquetas en negrita:**

> **Pérdida:** cuadrática · **Activación interna y latente:** tanh · **Salida:** sigmoide
>
> **Entrenamiento:** lote 32 · máximo 3.000 épocas · **Tolerancia:** ≤ 1 píxel por carácter

La primera línea reúne las decisiones del modelo; la segunda, las condiciones de entrenamiento. Usar un tamaño menor que el cuerpo, pero legible. Podés decir oralmente que se evaluó cada 100 épocas y se detuvo al cumplir, sin sumar más texto al footer. Evitar «tanh + sigmoide» sin aclarar dónde se aplica cada una. Si insertás el esquema completo como imagen, ya incluye su pie: no duplicarlo con otro footer.

**Para preguntas:** cada arquitectura se cruzó con las doce opciones de optimizador/tasa, pero las tasas numéricas no son iguales para todos los optimizadores. Se comparan configuraciones con sus tasas ensayadas; no es una comparación a un LR universal. Las semillas cambian la inicialización, no crean un conjunto de test. Momentum usa coeficiente 0,9; RMSProp rho 0,9; Adam beta1 0,9 y beta2 0,999; epsilon 1e−8 en RMSProp y Adam. Esos detalles pueden quedar fuera de la slide principal.

**Transición:** “Con esa búsqueda hecha, veamos cuáles consiguieron cumplir el objetivo.”

## 6 · Qué dejó la búsqueda: una solución todavía poco consistente

**Imagen nueva:** [story-search-results.png](assets/story-search-results.png). Conserva la comparación de arquitecturas, pero agrega dentro de cada barra el optimizador y la tasa de su mejor combinación. Reemplaza al gráfico anterior de arquitecturas para este relato.

**Texto para copiar:**

> **Solo 2 de las 180 corridas cumplieron la tolerancia.**
>
> Ambas usaron la arquitectura de mayor profundidad y RMSProp.
>
> Elegimos **RMSProp 0,002**, que cumplió en **1 de 3 semillas**.

**Pie del gráfico:** “Media del error máximo por carácter (3 semillas)”. Todas las barras usan el mismo azul del dataset. La aclaración “Mejor combinación de optimizador y tasa de cada arquitectura” va como texto editable en la PPT, fuera de la imagen.

**Guion oral:** “Cada barra muestra la mejor combinación de optimizador y tasa de esa arquitectura, promediada sobre sus tres semillas. Las redes más profundas tuvieron menor error agregado dentro de esta búsqueda. Aun así, de las ciento ochenta corridas, solo dos cumplieron el requisito: ambas con la red más profunda y RMSProp. Para elegir no nos quedamos con la mejor corrida aislada: comparamos las combinaciones sobre sus tres semillas, priorizando la media del máximo error. Así elegimos RMSProp con tasa cero coma cero cero dos. En esa combinación, las semillas cuarenta y dos y cuarenta y tres quedaron con un máximo de dos píxeles incorrectos; la cuarenta y cuatro llegó a uno. Ya teníamos una corrida que cumplía, pero queríamos que también funcionara con las otras inicializaciones.”

**Imagen de la tabla, sin footer:** [story-selected-seeds-table.png](assets/story-selected-seeds-table.png), también disponible en SVG. Respeta estos títulos y valores:

| Semilla | Máximo error por carácter | ¿Cumple? |
| --- | ---: | --- |
| 1 | 2 píxeles | No |
| 2 | 2 píxeles | No |
| 3 | 1 píxel | Sí |

En la tabla, 1/2/3 identifican las tres repeticiones; sus semillas reales son 42/43/44. Para hablar en los mismos términos que la imagen, decir “primera, segunda y tercera inicialización”. Los registros conservan los identificadores originales.

**Cómo leer el gráfico:** el máximo se calcula dentro de cada corrida sobre sus 32 caracteres; luego se promedia ese máximo entre las tres semillas. Un valor medio de 1,67 no significa que un carácter tenga un error fraccionario. Para cumplir, cada corrida debe dejar todos sus caracteres con error máximo ≤ 1. No se usa un umbral de error medio como criterio de éxito.

**Precisión:** las dos corridas exitosas fueron deeper/RMSProp 0,002/semilla 44 y deeper/RMSProp 0,0005/semilla 43. Por eso no decir «solo una de las 180 cumplió». La combinación seleccionada cumplió en 1/3 semillas. Las barras comparan las mejores combinaciones por arquitectura; no aíslan exclusivamente la profundidad.

**Transición:** “Como en dos semillas estábamos a un píxel de cumplir, el siguiente intento fue ajustar la tasa de aprendizaje.”

## 7 · ¿Alcanzaba con ajustar la tasa de aprendizaje?

**Cambio respecto del PowerPoint original:** reemplazar el diagrama de arquitectura de la antigua slide 6 por [story-06-learning-rate.png](assets/story-06-learning-rate.png). El diagrama pasa a la nueva slide 11. Los nombres de archivos se conservan aunque cambie la numeración.

**Texto:**

> Conservamos la red y RMSProp.
>
> Probamos tasas cercanas: **0,0015 · 0,002 · 0,0025**.
>
> **El ajuste de LR no mejoró la consistencia.**

**Guion oral:** “Como estábamos cerca de cumplir, pensamos que el problema podía ser el tamaño de los pasos. Probamos tasas cercanas desde cero y también continuar las semillas pendientes con una tasa menor. La tasa 0,0015 no produjo éxitos; 0,0025 volvió a cumplir en una de tres semillas. Las continuaciones tampoco resolvieron los patrones pendientes. Ajustar cuánto se movían los pesos no alcanzó, así que revisamos qué señal de error estábamos usando.”

**Pie:** “Misma arquitectura, pérdida cuadrática y RMSProp; semillas 42–44; hasta 3.000 épocas. Éxito: todos los caracteres con ≤ 1 píxel incorrecto.”

**Para preguntas:** las cuatro continuaciones de semillas 42/43, con LR 0,002 o 0,001 hasta época 4.000, tuvieron 0/4 éxitos. Son cuatro ramas, no cuatro inicializaciones independientes. Hubo mejoras secundarias del error medio; «no mejoró» se refiere a cumplir el requisito con más consistencia.

## 8 · ¿Por qué cambiar la función de pérdida?

**Cambio:** quitar el gráfico de resultados duplicado. Explicar el mecanismo antes de mostrar los resultados.

**Imagen:** [story-07-output-gradient.png](assets/story-07-output-gradient.png). Es una ilustración matemática; no son gradientes medidos durante nuestras corridas.

**Texto:**

> Un píxel puede estar muy equivocado y recibir una señal de corrección pequeña.
>
> **Cuadrática + sigmoide:** la saturación atenúa el gradiente de salida.
>
> **BCE + sigmoide:** elimina ese factor de atenuación.

**Ejemplo destacado:**

> Píxel esperado: **1** · Predicción: **0,01**
>
> Magnitud del gradiente de salida: **0,0098 con cuadrática → 0,99 con BCE**

**Guion oral:** “La tasa controla el tamaño del paso, pero ese paso parte de un gradiente. La sigmoide se aplana cerca de cero y de uno. Con la pérdida cuadrática, ese aplanamiento multiplica la señal de corrección: un píxel que debería estar encendido, pero que la red predice casi apagado, puede aportar un gradiente muy pequeño. Con entropía cruzada binaria, la derivada de la pérdida cancela ese factor de la sigmoide. Así, una predicción saturada y equivocada conserva una señal de corrección fuerte en la salida. Eso nos dio un motivo para cambiar la pérdida.”

**Fórmulas opcionales para el slide:** por píxel, con `p = sigmoid(u)`:

\[
L_Q=\tfrac12(p-y)^2,\qquad
\frac{\partial L_Q}{\partial u}=(p-y)\,p(1-p)
\]

\[
L_{BCE}=-y\log p-(1-y)\log(1-p),\qquad
\frac{\partial L_{BCE}}{\partial u}=p-y
\]

**Para preguntas:** en el ejemplo los gradientes son negativos, −0,009801 y −0,99; mostramos magnitudes. La segunda es unas 101 veces mayor en ese punto. Esto no implica entrenar 101 veces más rápido ni que Adam actualice los pesos 101 veces más.

Nuestra implementación usa `u=2βh`, con `β=1`. Respecto de `h`, ambas fórmulas incorporan el mismo factor `2β`; arriba lo absorbemos en el logit `u`. Se suman los píxeles y promedian las muestras. El código calcula BCE desde logits mediante una expresión estable.

**Qué demuestra y qué no:** es un mecanismo matemático que justifica el cambio. Las corridas muestran que ayudó en esta tarea; no medimos saturación durante el entrenamiento para identificarla como causa única. La cancelación corresponde a la salida, no elimina posibles gradientes pequeños en las capas ocultas tanh. No decir «el gradiente de BCE nunca se anula»: disminuye cuando la predicción es correcta.

## 9 · Con BCE cumplimos en las tres semillas

**Imagen:** [story-08-loss-result.png](assets/story-08-loss-result.png). Solo compara cuadrática y BCE con tanh. La comparación de latente lineal/tanh queda fuera del hilo principal.

**Texto:**

> Misma arquitectura · RMSProp 0,002 · latente tanh
>
> **Cuadrática: 1/3 → BCE: 3/3**
>
> Todos los caracteres dentro de tolerancia en las tres inicializaciones.

**Guion oral:** “Para comprobar el cambio, comparamos las dos pérdidas manteniendo la arquitectura, RMSProp, la tasa y las mismas semillas. Con cuadrática cumplió una de tres; con BCE cumplieron las tres. El error binario medio también bajó de aproximadamente 0,458 a 0,052 píxeles por carácter. Este fue el cambio que nos permitió cumplir de forma consistente en las inicializaciones exploradas.”

**Pie:** “Comparación de pérdidas en el mismo entorno; semillas 42–44; tolerancia ≤ 1 píxel por carácter.”

**Transición:** “Con BCE ya cumplíamos usando RMSProp. Quisimos ver si cambiar el optimizador podía reducir el error restante.”

**Trazabilidad:** esta comparación usa la referencia reconstruida de la campaña posterior; no mezcla métricas históricas y nuevas como si fueran una misma ejecución.

## 10 · De RMSProp a Adam: mismo cumplimiento, menor error

**Imagen:** [story-optimizer-comparison.png](assets/story-optimizer-comparison.png), disponible también en SVG. Dos paneles: a la izquierda el cumplimiento de tolerancia; a la derecha el error restante. Todas las barras usan el azul del dataset. No tiene footer: las condiciones van como texto editable en la PPT.

**Texto para copiar:**

> Misma arquitectura · BCE · latente tanh · LR 0,002
>
> Ambos cumplen en **3/3 semillas**.
>
> Con Adam, la media del error máximo baja de **1,00 a 0,33 píxeles**.
>
> **Elegimos Adam para la configuración final.**

**Guion oral:** “El cambio de pérdida ya había permitido cumplir con RMSProp. Mantuvimos BCE, la arquitectura y la tasa cero coma cero cero dos, y comparamos RMSProp con Adam usando las mismas tres semillas. El panel izquierdo muestra que ambos cumplieron las tres veces. La diferencia está en el derecho: con RMSProp el máximo error era un píxel en cada corrida; con Adam, dos corridas reconstruyeron todo exactamente y una quedó con un máximo de un píxel. Al promediar esos máximos, pasamos de uno a cero coma treinta y tres. Por eso elegimos Adam.”

**Cómo leerlo:** 0,33 es el promedio de los máximos de tres corridas, no un error fraccionario en un carácter. Los paneles tienen unidades distintas: el izquierdo cuenta corridas y el derecho mide píxeles. No comparar alturas entre paneles.

**Para preguntas:** RMSProp conserva 3/3 éxitos; Adam mejora el error dentro de la tolerancia. Esta comparación es sobre las semillas exploratorias 42–44. Las cinco semillas nuevas se usaron después para confirmar la configuración final. Las corridas se detienen al alcanzar tolerancia, de modo que se comparan resultados bajo esa regla de parada, no mínimos de error tras entrenar indefinidamente.

**Transición:** “Con esto cerramos las decisiones principales: la arquitectura de la búsqueda inicial, BCE como pérdida y Adam como optimizador.”

## 11 · Configuración final

**Imagen:** [03-final-architecture.png](assets/03-final-architecture.png), que estaba en la 6.

El diagrama no tiene título superior. Su footer dice: “tanh en capas internas y latente, sigmoide en la salida”.

**Texto:**

> **35 → 40 → 20 → 10 → 2 → 10 → 20 → 40 → 35**
>
> Capas internas y latente: **tanh** · Salida: **sigmoide**
>
> **BCE · Adam · LR 0,002 · lote de 32**
>
> Confirmación: **5/5 semillas nuevas dentro de tolerancia**

**Guion oral:** “Esta es la configuración que conservamos. La elegimos antes de probar cinco semillas nuevas y las cinco cumplieron el requisito sobre los mismos 32 caracteres. Evaluamos cada cien épocas y detenemos al alcanzar la tolerancia, con un máximo de tres mil.”

**Para preguntas:** con BCE, RMSProp y Adam cumplieron 3/3; Adam redujo la media del máximo error de 1 a 0,33 píxeles. BCE produjo el salto de consistencia antes del cambio de optimizador. Las cinco semillas nuevas prueban sensibilidad a la inicialización, no generalización a caracteres no vistos.

Omitir del relato las pruebas posteriores de tasa/scheduler, arquitectura angosta y presupuesto.

## 12 · Reconstrucción del dataset completo

**Imagen:** conservar [07-reconstructions.png](assets/07-reconstructions.png), grande.

**Texto:**

> **32/32 caracteres reconstruidos exactamente**
>
> Modelo seleccionado: semilla 49 · época 600
>
> Reconstrucciones binarizadas con umbral 0,5

**Guion oral:** “Cada par muestra el original a la izquierda y la reconstrucción a la derecha. En el modelo seleccionado no quedó ningún píxel incorrecto después del umbral. Aprendimos el dataset completo. Las otras cuatro semillas nuevas también cumplieron la tolerancia, aunque conservaron algún error de un píxel.”

No decir que todas las corridas tuvieron cero errores ni que las salidas continuas son idénticas al target.

## 13 · Los caracteres en el espacio latente

**Imagen:** conservar [08-latent-space.png](assets/08-latent-space.png), cuadrada y grande.

**Texto:**

> Cada punto representa un carácter.
>
> **35 píxeles → (z₁, z₂) → reconstrucción**
>
> Coordenadas reales del encoder entrenado.

**Guion oral:** “Estas son las dos salidas del encoder para cada carácter. El decoder usa esas coordenadas para reconstruir las imágenes que acabamos de mostrar. Como el latente usa tanh, los puntos están entre menos uno y uno. No asignamos un significado particular a los ejes. Queda explorar otros puntos para buscar una letra nueva.”

La generación sigue pendiente. El gráfico no demuestra que puntos intermedios produzcan letras válidas.

## Respaldo y reproducción

- Requisito: `sources/extracted/PA-004.md`, p. 1.
- Campaña histórica y LR: `tps_sia/tp5/comparison.md`. Los artefactos de esas corridas no están disponibles en este checkout; la figura de LR transcribe el registro, no recalcula modelos históricos.
- Comparación de pérdida: `tps_sia/tp5/results/presentation-campaign-2026-11-10/01-loss-latent/summary.json`.
- Adam y confirmación: `tps_sia/tp5/presentation-results.md`, resúmenes de etapas 02 y 06.
- Gradientes: derivación por regla de la cadena consistente con `tps_sia/tp3/shared/mlp.py`, métodos `_costo` y `_gradientes`. `wiki/topics/retropropagacion.md` respalda la regla de la cadena con OFF-004, pp. 39–50; no atribuirle la comparación específica BCE sin verificarla.
- Modelo y evaluación: `tps_sia/tp5/results/final/`.

Las figuras tienen PNG/SVG con prefijo `story-`. Para regenerar las figuras del todos contra todos y la tabla: `python tps_sia/tp5/ppt/generate_search.py`; `assets/search-provenance.json` registra la procedencia de la búsqueda. El diseño se obtiene de `configs/comparison-broad.json` y verifica que estén las 60 combinaciones completas. Las métricas históricas se transcriben de `comparison.md`, sin entrenamientos nuevos. Para regenerar LR, gradientes, BCE y comparación de optimizadores: `python tps_sia/tp5/ppt/generate_story.py`; su procedencia está en `assets/story-provenance.json` y `assets/optimizer-provenance.json`. Las figuras anteriores se conservan; `generate.py` las regenera.
