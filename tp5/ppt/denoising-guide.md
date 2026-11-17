# TP5 · Guion narrativo del punto 1.b (Denoising Autoencoder)

Deck: **`TP5-con-1b.pptx`**, copia de `TP5.pptx` con la sección 1.b insertada entre las portadas «3 · Autoencoder de Eliminación de Ruido» (diapositiva 15) y «4 · Autoencoder Variacional» (ahora la 25). Son **7 diapositivas** (16–22) y **2 de respaldo** (23–24). El guion oral de cada una está copiado en sus notas del orador.

Hilo conductor: **ruido → entrenar con ruido → ¿qué ancho de código? → comparación → ejemplos → límites.**

Nombres de modelo, siempre iguales: «Entrada ruidosa», «AE de 1.a (sin ruido al entrenar)», «DAE código 2» y «DAE código 10». Etiqueta de sección en todas: «3. Autoencoder de eliminación de ruido».

## 15 · Separador de sección

**Cambio:** se quitaron «El CEITBA como organización social» y «03 · 9», restos de otra plantilla. El resto queda igual.

## 16 · ¿Cómo entrenamos un DAE?

**Imagen:** [denoising-01-noise-levels.png](assets/denoising-01-noise-levels.png), también disponible en SVG. Texto alternativo: “El carácter x original y sus versiones con p = 0,05; 0,10; 0,20 y 0,30: tienen 3, 4, 7 y 12 píxeles invertidos. El nivel 0,30 se usa solo para evaluar.”

**Texto para copiar:**

> **Ruido al entrenar**
>
> • Cada píxel se invierte con probabilidad p
>
> • Entrada ruidosa, objetivo limpio
>
> • Cada época sortea p por carácter: 0; 0,05; 0,10 o 0,20
>
> **p = 0,30:** solo para evaluar, fuera del rango entrenado.

**Guion oral:** “Para el punto 1.b usamos el mismo dataset de treinta y dos caracteres. El ruido que elegimos es invertir píxeles: cada píxel pasa de cero a uno, o al revés, con probabilidad p. La red recibe la imagen ruidosa, pero el objetivo es la imagen limpia. En cada época, a cada carácter le sorteamos un nivel entre cero, cinco, diez y veinte por ciento, así que el ruido es nuevo cada vez. El treinta por ciento no se usa para entrenar: lo reservamos para evaluar fuera del rango entrenado.”

**Pie:** Sin pie en la diapositiva. El ejemplo es el carácter «x»: primera realización de cada nivel en la colección de evaluación (índices fijos 24, 2432, 5632, 8832 y 12032 de `collections/final.npz`).

**Para preguntas:** Con probabilidad p se invierten en promedio 35 · p píxeles: 1,75; 3,5; 7 y 10,5. En el ejemplo son 3, 4, 7 y 12 porque el número exacto varía en cada sorteo. Se incluyó p = 0 en el sorteo (un cuarto de las imágenes queda limpia) para que la red no pierda la capacidad de reconstruir caracteres sin ruido. Como el lote es de 32 y hay 32 caracteres, cada época es una actualización de pesos. El ruido es el único que estudiamos: inversión independiente de píxeles, no ruido estructurado.

**Cuidado:** No presentar p = 0,30 como parte del entrenamiento.

**Transición:** “Con el ruido definido, faltaba decidir qué red entrenar.”

## 17 · Dos arquitecturas, una sola diferencia

**Imagen:** [denoising-02-architectures.png](assets/denoising-02-architectures.png), también disponible en SVG. Texto alternativo: “Dos diagramas de bloques apilados. DAE código 2, la red de 1.a: 35, 40, 20, 10, 2, 10, 20, 40, 35. DAE código 10: 35, 40, 20, 10, 10, 10, 20, 40, 35. Solo cambia la capa central, resaltada.”

**Texto para copiar:**

> **¿Por qué probar 10?**
>
> • Con 2 números ya costó separar 32 caracteres sin ruido
>
> • Hipótesis: con 10 queda margen para absorber el ruido
>
> **Igual en ambas:** BCE · Adam 0,002 · lote 32 · tanh interna · salida sigmoide.

**Guion oral:** “Partimos de la red que ganó en 1.a y comparamos dos variantes que solo difieren en la capa central: código de dos, que es la misma red de 1.a, y código de diez. Todo lo demás es igual: entropía cruzada binaria, Adam con tasa cero coma cero cero dos, lote de treinta y dos, tanh en las capas internas y sigmoide en la salida. Los pesos se entrenan desde cero. ¿Por qué diez? En 1.a ya costó que dos números alcanzaran para separar los treinta y dos caracteres sin ruido. Un DAE, además, tiene que llevar muchas versiones estropeadas al mismo carácter limpio. Nuestra hipótesis era que con diez dimensiones queda margen para absorber esas perturbaciones.”

**Pie:** Sin pie en la diapositiva.

**Para preguntas:** Solo se compararon los anchos 2 y 10; no se probaron 4, 6 ni 16. El optimizador y la tasa se heredaron de 1.a, donde se ajustaron para un autoencoder sin ruido de código 2, y no se reoptimizaron para el DAE. En 1.a hicieron falta más de 200 corridas de búsqueda (180 de la campaña factorial y 43 posteriores) para que el código de 2 alcanzara. Al cambiar solo el ancho del código, el efecto se puede atribuir a ese cambio.

**Cuidado:** No decir que 10 es el ancho óptimo: es una hipótesis que se contrasta contra 2.

**Transición:** “Para que la comparación fuera justa, fijamos el protocolo antes de entrenar.”

## 18 · Cómo medimos y elegimos

**Imagen:** ninguna; lista a lo ancho con el formato de «Configuración final».

**Texto para copiar:**

> **Corridas:** 2 variantes × 3 semillas · 3000 épocas cada una
>
> **Validación:** 12.832 imágenes fijas con p = 0,05; 0,10 y 0,20
>
> **Checkpoint:** menor error de píxeles medio en validación
>
> **Evaluación:** otras 12.832 imágenes, no usadas para decidir
>
> **Referencias:** Entrada ruidosa y AE de 1.a (sin ruido al entrenar)
>
> **Ganó el código 10** (representante: semilla 44)

**Guion oral:** “Entrenamos seis corridas: las dos variantes con las semillas cuarenta y dos, cuarenta y tres y cuarenta y cuatro, tres mil épocas cada una. Para elegir el mejor checkpoint de cada corrida usamos una colección fija de validación, de doce mil ochocientas treinta y dos imágenes, y nos quedamos con el de menor error de píxeles medio con cinco, diez y veinte por ciento de ruido. Los resultados que vamos a mostrar salen de otra colección independiente, del mismo tamaño y generada con otra semilla, que no se usó para decidir nada. Comparamos siempre contra dos referencias: la entrada ruidosa sin procesar y el autoencoder de 1.a, que nunca vio ruido. En validación ganó el código diez: uno coma veintitrés píxeles erróneos de media contra dos coma veintiocho. Como representante usamos la semilla cuarenta y cuatro.”

**Pie:** Sin pie: el recuadro coral cierra la diapositiva.

**Para preguntas:** Cada colección tiene 100 versiones por carácter para cada nivel con ruido (0,05; 0,10; 0,20 y 0,30) y una versión limpia por carácter: 4 × 3.200 + 32 = 12.832. La validación se midió al inicio, cada 100 épocas y al final; los empates se resuelven por MSE y después por la época más temprana. Los niveles 0 y 0,30 no intervinieron en la selección. Las variantes se comparan por el promedio de los mejores puntajes de sus tres semillas: código 10 = 1,23 píxeles; código 2 = 2,28. La corrida seleccionada es `latent-10-seed-44`. No se eliminaron coincidencias casuales entre colecciones: su independencia es estadística.

**Cuidado:** No decir que la evaluación mide generalización: son versiones ruidosas de los mismos 32 caracteres.

**Transición:** “Veamos primero si entrenar con ruido sirve.”

## 19 · Un autoencoder común empeora el ruido; el DAE lo reduce

**Imagen:** [denoising-03-error-by-level.png](assets/denoising-03-error-by-level.png), también disponible en SVG. Texto alternativo: “Gráfico de líneas de píxeles erróneos por imagen según p. Con p = 0,10: Entrada ruidosa 3,47; AE de 1.a (sin ruido al entrenar) 9,26; DAE código 10 0,50. El DAE código 2 queda entre la entrada y el DAE código 10. La zona de p = 0,30 está sombreada como fuera del rango entrenado.”

**Texto para copiar:**

> **Con p = 0,10**
>
> • Entrada ruidosa: 3,47 píxeles erróneos
>
> • AE de 1.a (sin ruido al entrenar): 9,26
>
> • DAE código 10: 0,50
>
> **Resultado:** el AE de 1.a empeora la entrada; el DAE la mejora.

**Guion oral:** “El eje horizontal es la probabilidad de invertir cada píxel y el vertical, cuántos píxeles quedan mal por imagen: más abajo es mejor. La línea punteada gris es la entrada ruidosa, es decir, lo que pasa si no hacemos nada. Con diez por ciento de ruido, la entrada trae tres coma cuarenta y siete píxeles erróneos. El autoencoder de 1.a, que reconstruye perfecto las imágenes limpias, deja nueve coma veintiséis: empeora el problema. El DAE de código diez deja cero coma cincuenta. La zona sombreada es el treinta por ciento, que está fuera del rango entrenado: ahí el DAE todavía baja el error, de diez coma seis a siete coma dos, pero mucho menos.”

**Pie:** **Evaluación:** 3.200 imágenes por nivel con ruido · **DAE:** media de 3 semillas, banda mínimo–máximo

| Píxeles erróneos por imagen | p = 0 | p = 0,05 | p = 0,10 | p = 0,20 | p = 0,30 (fuera de rango) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Entrada ruidosa | 0 | 1,767 | 3,466 | 6,893 | 10,568 |
| AE de 1.a (sin ruido al entrenar) | 0 | 6,463 | 9,260 | 12,606 | 15,032 |
| DAE código 2 | 0,260 | 0,553 [0,326–0,713] | 1,327 [1,141–1,423] | 4,654 [4,465–4,869] | 9,032 [8,811–9,161] |
| DAE código 10 | 0 | 0,111 [0,094–0,120] | 0,500 [0,471–0,547] | 2,935 [2,886–2,966] | 7,183 [7,098–7,304] |

| Fracción de imágenes exactas | p = 0 | p = 0,05 | p = 0,10 | p = 0,20 | p = 0,30 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Entrada ruidosa | 1,000 | 0,166 | 0,029 | 0,001 | 0,000 |
| AE de 1.a (sin ruido al entrenar) | 1,000 | 0,349 | 0,188 | 0,084 | 0,045 |
| DAE código 2 | 0,844 | 0,767 | 0,656 | 0,355 | 0,132 |
| DAE código 10 | 1,000 | 0,955 | 0,852 | 0,496 | 0,173 |

Colección de evaluación: 3.200 imágenes por nivel con ruido y 32 limpias. Para los DAE, media de 3 semillas y rango mínimo–máximo entre corchetes. Estas tablas sirven para las diapositivas 19 y 20.

**Para preguntas:** El AE de 1.a empeora el 73 % de los ejemplos con p = 0,10 y el 80 % con p = 0,20. El DAE código 10 empeora el 0,9 %, 2,7 %, 11,1 % y 19,3 % con p = 0,05; 0,10; 0,20 y 0,30. La comparación con el AE de 1.a es casi controlada: además del ruido al entrenar, difieren la semilla (49 contra 42–44) y la duración (el AE de 1.a se detuvo a las 600 épocas, al cumplir la tolerancia; los DAE entrenaron 3000). El AE de 1.a es el modelo final de 1.a (semilla 49, BCE, Adam 0,002, SHA256 e39ab072…). Con p = 0 hay solo 32 imágenes limpias.

**Cuidado:** No decir que el DAE «elimina el ruido» en general: lo reduce, y cada vez menos al subir p.

**Transición:** “Las dos variantes del DAE reducen el ruido, pero no por igual.”

## 20 · Código 10 vs código 2

**Imagen:** [denoising-04-code-comparison.png](assets/denoising-04-code-comparison.png), también disponible en SVG. Texto alternativo: “Dos paneles. Izquierda: imágenes recuperadas exactas según p; DAE código 10: 100, 96, 85, 50 y 17 %; DAE código 2: 84, 77, 66, 36 y 13 %. Derecha: píxeles erróneos con p = 0; DAE código 2: 0,26 de media; DAE código 10: 0.”

**Texto para copiar:**

> **Gana el código 10**
>
> • En todos los niveles y en las 3 semillas
>
> • Los rangos de error entre semillas no se solapan
>
> • El DAE código 2 daña imágenes limpias: 0,26 píxeles de media
>
> **Límite:** solo comparamos 2 y 10; no sabemos si 10 es el óptimo.

**Guion oral:** “A la izquierda, la fracción de imágenes que se recuperan sin ningún píxel mal. El código diez queda arriba en todos los niveles: con diez por ciento de ruido recupera exactas el ochenta y cinco por ciento de las imágenes, y el código dos, el sesenta y seis. Pasa lo mismo con las tres semillas: en píxeles erróneos, los rangos entre semillas no se solapan en ningún nivel. A la derecha, lo que pasa con las imágenes limpias: el DAE de código dos les introduce errores, cero coma veintiséis píxeles de media, y el de código diez las conserva todas. Esto apoya la hipótesis, con una aclaración: solo comparamos dos anchos, así que no podemos decir que diez sea el óptimo.”

**Pie:** **Evaluación:** 3.200 imágenes por nivel con ruido y 32 limpias · **Bandas y barra de error:** mínimo–máximo entre 3 semillas

**Para preguntas:** Con p = 0,20, la peor semilla del código 10 deja 2,966 píxeles erróneos y la mejor del código 2, 4,465. Con p = 0, el DAE código 2 deja entre 0,031 y 0,469 píxeles erróneos según la semilla y solo el 84 % de las imágenes limpias queda exacto. Los porcentajes del gráfico están redondeados (95,5 % se muestra como 96 %). Tres semillas son pocas: alcanzan para ver que el orden no depende de una inicialización, no para estimar la variabilidad con precisión.

**Cuidado:** No decir que 10 es el ancho óptimo ni que el resultado vale para otros anchos.

**Transición:** “Para verlo en imágenes concretas, estos son seis ejemplos con veinte por ciento de ruido.”

## 21 · Ejemplos con p = 0,20

**Imagen:** [denoising-05-examples-p020.png](assets/denoising-05-examples-p020.png), también disponible en SVG. Texto alternativo: “Lámina de seis caracteres (acento grave, a, h, p, x y DEL) con p = 0,20. Filas: Original, Entrada ruidosa, AE de 1.a (sin ruido al entrenar), DAE código 2 y DAE código 10. Errores del DAE código 10: 3, 0, 3, 0, 0 y 0; DAE código 2: 7, 0, 10, 8, 9 y 0; AE de 1.a: 13, 22, 22, 8, 17 y 0.”

**Texto para copiar:**

> Solo el título; la imagen ocupa la diapositiva (fondo oscuro).

**Guion oral:** “Cada columna es un carácter. Arriba está el original, después la entrada con veinte por ciento de ruido y abajo lo que devuelve cada modelo. El número indica cuántos píxeles quedaron mal, en rojo si es más de uno. El autoencoder de 1.a deja entre ocho y veintidós píxeles mal en cinco de los seis: en general devuelve otra figura. El DAE de código dos acierta dos y falla en los otros cuatro. El DAE de código diez recupera cuatro de los seis sin errores y deja tres píxeles mal en los otros dos. Estos ejemplos no los elegimos por favorables: los índices se fijaron antes de ver los resultados.”

**Pie:** **Ejemplos:** primera realización de 6 caracteres, fijados antes de ver resultados; no se eligieron por favorables

**Modelos:** DAE código 2, semilla 43 · DAE código 10, semilla 44 (mejores de cada variante en validación)

**Para preguntas:** Caracteres 0, 1, 8, 16, 24 y 31 (`` ` ``, a, h, p, x y DEL), primera realización; índices de `evaluation/metrics.json → example_indices`. Píxeles erróneos: entrada 7, 6, 8, 4, 7 y 8; AE de 1.a 13, 22, 22, 8, 17 y 0; DAE código 2 (`winner-seed-43`, la mejor semilla de esa variante por validación) 7, 0, 10, 8, 9 y 0; DAE código 10 (`latent-10-seed-44`, la seleccionada) 3, 0, 3, 0, 0 y 0. `DEL` tiene todos los píxeles encendidos y es el patrón que más se distingue del resto.

**Cuidado:** No reemplazar estos ejemplos por otros más lucidos.

**Transición:** “Cerramos con lo que aprendimos y hasta dónde llega.”

## 22 · Qué aprendimos y límites

**Imagen:** ninguna; lista a lo ancho con el formato de «Configuración final».

**Texto para copiar:**

> **Con 10 % de ruido:** 85 % de imágenes exactas (DAE código 10)
>
> **Con 20 % de ruido:** 50 % de imágenes exactas
>
> **Fuera de rango (p = 0,30):** error de 10,6 a 7,2 píxeles; 17 % exactas
>
> **Límites:** 32 caracteres conocidos · 3 semillas · anchos 2 y 10
>
> **Presupuesto:** 3000 épocas quedaron cortas
>
> **Entrenar con ruido lo reduce;** un AE común lo amplifica

**Guion oral:** “Con diez por ciento de ruido, el DAE de código diez recupera de forma exacta el ochenta y cinco por ciento de las imágenes; con veinte por ciento, el cincuenta. Fuera del rango entrenado, con treinta por ciento, baja el error de diez coma seis a siete coma dos píxeles, pero solo el diecisiete por ciento queda exacto. Los límites: todas las imágenes evaluadas son versiones ruidosas de los mismos treinta y dos caracteres, así que medimos recuperación de caracteres conocidos, no qué pasa con letras nuevas. Usamos tres semillas, que son pocas, y solo dos anchos de código. Y el presupuesto de tres mil épocas quedó corto: cuatro de las seis corridas eligieron la última época. La idea principal: entrenar con ruido es lo que permite reducirlo, mientras que un autoencoder común lo amplifica.”

**Pie:** Sin pie: el recuadro coral cierra la diapositiva.

**Para preguntas:** La validación todavía bajaba al final (por ejemplo, de 1,427 a 1,254 entre las épocas 2500 y 3000 en `latent-10-seed-42`); es posible que más épocas mejoren ambas variantes, pero el protocolo no autorizaba ampliar el presupuesto. Solo se estudió un tipo de ruido (inversión independiente de píxeles). El optimizador y la tasa no se reoptimizaron para el DAE. Una posible explicación, no verificada, de la caída con p = 0,30 es que con tantos píxeles invertidos la imagen ruidosa se parezca más a otro carácter que al original.

**Cuidado:** No decir que el DAE «generaliza» o «funciona con letras nuevas», ni que 10 es el ancho óptimo.

**Transición:** “Con esto cerramos el punto 1.b y pasamos al autoencoder variacional.”

## 23 · Respaldo · Ejemplos con p = 0,10

**Imagen:** [denoising-06-examples-p010.png](assets/denoising-06-examples-p010.png), también disponible en SVG. Texto alternativo: “Lámina de los mismos seis caracteres con p = 0,10. Errores del DAE código 10: 0, 3, 0, 0, 0 y 0; DAE código 2: 1, 0, 0, 8, 0 y 0; AE de 1.a: 5, 0, 10, 9, 13 y 0; Entrada ruidosa: 2, 6, 2, 3, 4 y 2.”

**Texto para copiar:**

> Solo el título; la imagen ocupa la diapositiva (fondo oscuro).

**Guion oral:** “Diapositiva de respaldo. Son los mismos seis caracteres con diez por ciento de ruido. El DAE de código diez recupera cinco de los seis sin errores y deja tres píxeles mal en la a. El de código dos falla sobre todo en la p, con ocho píxeles mal.”

**Pie:** **Ejemplos:** los mismos 6 caracteres e índices fijos, con p = 0,10; no se eligieron por favorables

**Modelos:** DAE código 2, semilla 43 · DAE código 10, semilla 44 (mejores de cada variante en validación)

**Para preguntas:** Píxeles erróneos: entrada 2, 6, 2, 3, 4 y 2; AE de 1.a 5, 0, 10, 9, 13 y 0; DAE código 2 1, 0, 0, 8, 0 y 0; DAE código 10 0, 3, 0, 0, 0 y 0. Acá se ve un caso en que el DAE código 10 no es el mejor: en la «a», el AE de 1.a y el DAE código 2 quedan en 0 y el código 10 en 3.

**Cuidado:** Mostrarla solo si preguntan por otro nivel de ruido.

**Transición:** Volver a la diapositiva de cierre.

## 24 · Respaldo · Error por carácter

**Imagen:** [denoising-07-by-character.png](assets/denoising-07-by-character.png), también disponible en SVG. Texto alternativo: “Mapa de calor de los píxeles erróneos por imagen del DAE código 10, media de 3 semillas, para los 32 caracteres y p = 0,05; 0,10; 0,20 y 0,30. DEL es el más fácil con ruido bajo; el acento grave, el más difícil con p = 0,05 y 0,10.”

**Texto para copiar:**

> Solo el título; la imagen ocupa la diapositiva (fondo oscuro).

**Guion oral:** “Diapositiva de respaldo. Cada columna es un carácter y cada fila un nivel de ruido; el número es cuántos píxeles deja mal en promedio el DAE de código diez. El color es relativo al máximo de cada fila. DEL, que tiene todos los píxeles encendidos, es el más fácil con ruido bajo. Como análisis exploratorio, con ruido bajo los caracteres que más se parecen a otro tienden a ser los más difíciles; con treinta por ciento esa relación desaparece. Son solo treinta y dos puntos, así que es una hipótesis, no una conclusión.”

**Pie:** **Exploratorio (solo 32 puntos):** con ruido bajo, los caracteres más parecidos a otro tienden a ser los más difíciles. No está demostrado.

**Para preguntas:** `report.md` informa una correlación de Spearman de −0,83 entre el error por carácter y la distancia de Hamming mínima al resto con p = 0,05 (−0,56 con p = 0,10 y 0,00 con p = 0,30). Al recalcularla desde `by-character.csv` (media de 3 semillas, rangos promedio en los empates), `generate_denoising.py` obtiene −0,81; −0,56; −0,25 y 0,03: misma conclusión, pero el primer valor no coincide con el informe. Por eso la diapositiva no muestra el número. Los pares más cercanos son `l`/`|` y `r`/`n`.

**Cuidado:** No dar por demostrada la explicación por carácter: es exploratoria.

**Transición:** Volver a la diapositiva de cierre.

## Qué no afirmar en toda la sección

- Que el DAE «elimina el ruido» en general, que «generaliza» o que «funciona con letras nuevas»: todas las imágenes de evaluación son versiones ruidosas de los mismos 32 caracteres.
- Que 10 es el ancho óptimo: solo se compararon 2 y 10.
- Que p = 0,30 formó parte del entrenamiento: es extrapolación.
- Que las explicaciones por carácter están demostradas: son exploratorias.
- Aclarar que 3 semillas son pocas, que el presupuesto de 3000 épocas quedó corto (4 de 6 corridas eligieron la última época) y que la comparación con el AE de 1.a es casi controlada (difieren la semilla y la duración).

## Respaldo y reproducción

- Informe completo y limitaciones: `tps_sia/tp5/results/denoising/report.md`. Protocolo y esquemas de archivos: `tps_sia/tp5/denoising-data.md`.
- Para regenerar las figuras: `python ppt/generate_denoising.py` desde `tps_sia/tp5/`. Lee solo `results/denoising/` (CSV de `evaluation/`, `metrics.json`, `selection.json`, `collections/final.npz` y las predicciones guardadas de los modelos mostrados); no entrena ni carga modelos.
- El script verifica con `assert` los números de las diapositivas contra `by-level.csv` y `seed-aggregates.csv` (tablas de arriba, porcentajes de ejemplos empeorados, validación 1,23 y 2,28, corrida seleccionada, índices fijos y errores de los ejemplos) y guarda los SHA256 de las entradas en `assets/denoising-provenance.json`.
- Las figuras tienen PNG (dpi 220) y SVG con prefijo `denoising-`. Las de `results/denoising/figures/` son de análisis, con otra paleta, y quedan como respaldo.
- Diferencia con el informe: la correlación de Spearman con p = 0,05 da −0,81 al recalcularla, no −0,83 (ver diapositiva 24).
