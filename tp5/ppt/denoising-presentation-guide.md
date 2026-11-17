# TP5 · Guion del punto 1.b (Denoising Autoencoder)

Guía única para la sección 1.b. Reemplaza a `denoising-guide.md` y `denoising-extension-guide.md`, y describe el deck **`TP5-con-1b-anexo.pptx` tal como quedó después de la edición del grupo** (10 diapositivas).

Hilo conductor: **ruido → entrenar con ruido → ¿qué ancho de código? → cómo elegimos → comparación → ejemplos → límites → ¿y con más épocas?**

Nombres de modelo, siempre iguales: «Entrada ruidosa», «AE de 1.a (sin ruido al entrenar)», «DAE código 2» y «DAE código 10». Etiqueta de sección en todas: «3. Autoencoder de eliminación de ruido».

| # | Diapositiva | Tipo |
| ---: | --- | --- |
| 1 | Portada · 3 · Autoencoder de Eliminación de Ruido | Oscura |
| 2 | ¿Cómo entrenamos un DAE? | Clara |
| 3 | Dos arquitecturas, una sola diferencia | Clara |
| 4 | Cómo medimos y elegimos | Clara |
| 5 | Un autoencoder común empeora el ruido; el DAE lo reduce | Clara |
| 6 | Código 10 vs código 2 | Clara |
| 7 | Ejemplos con p = 0,20 | Oscura |
| 8 | Límites del análisis | Clara |
| 9 | ¿Y si probamos con más épocas? | Clara |
| 10 | Validación hasta 6000 épocas | Oscura |

**Cambios respecto de las guías anteriores:**

- Se quitaron «Qué aprendimos y límites» y los dos respaldos (ejemplos con p = 0,10 y error por carácter). Su información útil para preguntas quedó en la sección [Material quitado](#material-quitado-por-si-preguntan).
- «Límites del análisis» y el análisis de 6000 épocas pasaron al flujo principal; ya no se presentan como anexo.
- La diapositiva 4 ahora tiene un gráfico de validación por corrida y lista más corta.
- La diapositiva 5 suma las etiquetas 10,6 y 7,2 en p = 0,30.
- La diapositiva 7 reemplaza el pie por tres textos laterales (Lectura, Selección de ejemplos, Destacado).

Los **guiones orales** de esta guía parten de las notas actuales del deck. Donde las notas tenían un error, acá figura la versión corregida y se marca con **[corregido]**; las notas del PowerPoint todavía tienen el texto anterior.

## 1 · Portada de sección

**Texto:** «3 · Autoencoder de Eliminación de Ruido». Sin guion.

**Transición:** “Pasamos al punto 1.b: un autoencoder que tiene que limpiar ruido.”

## 2 · ¿Cómo entrenamos un DAE?

**Imagen:** [denoising-01-noise-levels.png](assets/denoising-01-noise-levels.png). El carácter «x» limpio y sus versiones con p = 0,05; 0,10; 0,20 y 0,30.

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

**Pie:** sin pie. El ejemplo es la primera realización de cada nivel en la colección de evaluación (índices fijos 24, 2432, 5632, 8832 y 12032 de `collections/final.npz`).

**Guion oral:** “Para el punto 1.b usamos el mismo dataset de 32 caracteres. El ruido que elegimos es invertir píxeles: cada píxel pasa de cero a uno, o al revés, con probabilidad p. La red recibe la imagen ruidosa, pero el objetivo es la imagen limpia. En cada época, a cada carácter le sorteamos un nivel entre cero, cinco, diez y veinte por ciento, así que el ruido es nuevo cada vez. El treinta por ciento no se usa para entrenar: lo reservamos para evaluar fuera del rango entrenado.”

**Para preguntas:** Con probabilidad p se invierten en promedio 35 · p píxeles: 1,75; 3,5; 7 y 10,5. En el ejemplo son 3, 4, 7 y 12 porque el número exacto varía en cada sorteo. Se incluyó p = 0 en el sorteo (un cuarto de las imágenes queda limpio) para que la red no pierda la capacidad de reconstruir caracteres sin ruido. Como el lote es de 32 y hay 32 caracteres, cada época es una actualización de pesos.

**Cuidado:** no presentar p = 0,30 como parte del entrenamiento.

**Transición:** “Con el ruido definido, faltaba decidir qué red entrenar.”

## 3 · Dos arquitecturas, una sola diferencia

**Imagen:** [denoising-02-architectures.png](assets/denoising-02-architectures.png). Dos diagramas de bloques: 35→40→20→10→**2**→10→20→40→35 y 35→40→20→10→**10**→10→20→40→35.

**Texto para copiar:**

> **¿Por qué probar 10?**
>
> • Con 2 números ya costó separar 32 caracteres sin ruido
>
> • Hipótesis: con 10 queda margen para absorber el ruido
>
> **Igual en ambas:** BCE · Adam 0,002 · lote 32 · tanh interna · salida sigmoide.

**Pie:** sin pie.

**Guion oral:** “Partimos de la red que ganó en 1.a y comparamos dos variantes que solo difieren en la capa central: código de dos, que es la misma red de 1.a, y código de diez. Todo lo demás se mantiene igual: la entropía cruzada binaria, Adam con tasa cero coma cero cero dos, el lote de treinta y dos, la tanh en las capas internas y la sigmoide en la salida. Los pesos se entrenan desde cero. ¿Por qué diez? En 1.a ya costó que dos números alcanzaran para separar los treinta y dos caracteres sin ruido. Un DAE, además, tiene que llevar muchas versiones estropeadas al mismo carácter limpio. Nuestra hipótesis era que con diez dimensiones queda margen para absorber esas perturbaciones.”

**Para preguntas:** Solo se compararon los anchos 2 y 10; no se probaron 4, 6 ni 16. El optimizador y la tasa se heredaron de 1.a, donde se ajustaron para un autoencoder sin ruido de código 2, y no se reoptimizaron para el DAE. En 1.a hicieron falta más de 200 corridas de búsqueda (180 de la campaña factorial y 43 posteriores) para que el código de 2 alcanzara. Al cambiar solo el ancho del código, el efecto se puede atribuir a ese cambio.

**Cuidado:** no decir que 10 es el ancho óptimo: es una hipótesis que se contrasta contra 2.

**Transición:** “Para que la comparación fuera justa, fijamos el protocolo antes de entrenar.”

## 4 · Cómo medimos y elegimos

**Imagen:** [denoising-08-validation-by-run.png](assets/denoising-08-validation-by-run.png). Barras con el error de validación del mejor checkpoint de cada corrida: DAE código 2, 2,42; 2,13 y 2,30 (media 2,28); DAE código 10, 1,25; 1,23 y 1,20 (media 1,23). La semilla 44 del código 10 está marcada como «seleccionada». **En el deck la imagen no tiene texto alternativo.**

**Texto para copiar:**

> **Corridas:** 2 variantes × 3 semillas
> 3000 épocas cada una
>
> **Validación:** 9.600 imágenes fijas con p = 0,05; 0,10 y 0,20 **[corregido: el deck dice 9.632]**
>
> **Checkpoint:** menor error de píxeles medio en validación
>
> **Evaluación:** otras 12.832 imágenes, no usadas para decidir
>
> **Ganó el código 10**
> (representante: semilla 44)

**Pie:** sin pie.

**Guion oral [corregido]:** “Entrenamos seis corridas: las dos variantes con las semillas 42, 43 y 44, tres mil épocas cada una. Para elegir el mejor checkpoint de cada corrida usamos una colección fija de validación y nos quedamos con el de menor error de píxeles medio sobre las 9.600 imágenes con 5, 10 y 20 por ciento de ruido. Los resultados que vamos a mostrar salen de otra colección independiente, de 12.832 imágenes y generada con otra semilla, que no se usó para decidir nada. Comparamos siempre contra dos referencias: la entrada ruidosa sin procesar y el autoencoder de 1.a, que nunca vio ruido. En el gráfico, cada barra es el mejor checkpoint de una corrida: ganó el código 10, con 1,23 píxeles erróneos de media contra 2,28. Como representante usamos la semilla 44, la de menor error.”

*Qué cambia respecto de las notas actuales:* las notas dicen que la validación tiene 12.832 imágenes. La colección tiene ese tamaño, pero el puntaje promedia solo los tres niveles de selección: 3 × 3.200 = 9.600. También se agregó una frase para leer el gráfico.

**Para preguntas:** Cada colección tiene 100 versiones por carácter para cada nivel con ruido (0,05; 0,10; 0,20 y 0,30) y una versión limpia por carácter: 4 × 3.200 + 32 = 12.832. En validación, el nivel 0 (32 imágenes) y el 0,30 (3.200) no entran en el puntaje. La validación se midió al inicio, cada 100 épocas y al final; los empates se resuelven por MSE y después por la época más temprana. La corrida seleccionada es `latent-10-seed-44` (1,20). Mejores épocas: código 2, 3000, 3000 y 2800; código 10, 3000, 2600 y 3000 (semillas 42, 43 y 44). No se eliminaron coincidencias casuales entre colecciones: su independencia es estadística.

**Cuidado:** no decir que la evaluación mide generalización: son versiones ruidosas de los mismos 32 caracteres.

**Transición:** “Veamos primero si entrenar con ruido sirve.”

## 5 · Un autoencoder común empeora el ruido; el DAE lo reduce

**Imagen:** [denoising-03-error-by-level.png](assets/denoising-03-error-by-level.png), con las etiquetas **10,6** (Entrada ruidosa) y **7,2** (DAE código 10) agregadas en p = 0,30.

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

**Pie:** **Evaluación:** 3.200 imágenes por nivel con ruido · **DAE:** media de 3 semillas, banda mínimo–máximo

**Guion oral:** “El eje horizontal es la probabilidad de invertir cada píxel y el vertical, cuántos píxeles quedan mal por imagen: cuanto más abajo, mejor. La línea punteada gris es la entrada ruidosa, es decir, lo que pasa si no hacemos nada. Con 10 % de ruido, la entrada trae 3,47 píxeles erróneos. El autoencoder de 1.a, que reconstruye perfecto las imágenes limpias, deja 9,26: o sea, empeora el problema. El DAE de código 10 deja 0,50. La zona sombreada es el 30 %, que está fuera del rango entrenado: ahí el DAE de código 10 todavía baja el error, de 10,6 a 7,2, pero mucho menos.”

**Para preguntas:**

| Píxeles erróneos por imagen | p = 0 | p = 0,05 | p = 0,10 | p = 0,20 | p = 0,30 (fuera de rango) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Entrada ruidosa | 0 | 1,767 | 3,466 | 6,893 | 10,568 |
| AE de 1.a (sin ruido al entrenar) | 0 | 6,463 | 9,260 | 12,606 | 15,032 |
| DAE código 2 | 0,260 | 0,553 [0,326–0,713] | 1,327 [1,141–1,423] | 4,654 [4,465–4,869] | 9,032 [8,811–9,161] |
| DAE código 10 | 0 | 0,111 [0,094–0,120] | 0,500 [0,471–0,547] | 2,935 [2,886–2,966] | 7,183 [7,098–7,304] |

DAE: media de 3 semillas y rango mínimo–máximo. El AE de 1.a empeora el 73 % de los ejemplos con p = 0,10 y el 80 % con p = 0,20. El DAE código 10 empeora el 0,9 %, 2,7 %, 11,1 % y 19,3 % con p = 0,05; 0,10; 0,20 y 0,30. Con p = 0,30, el DAE código 10 reduce el error un 32 %, pero solo el 17 % de las imágenes queda exacto. La comparación con el AE de 1.a es casi controlada: además del ruido al entrenar, difieren la semilla (49 contra 42–44) y la duración (el AE de 1.a se detuvo a las 600 épocas, al cumplir la tolerancia; los DAE entrenaron 3000).

**Cuidado:** no decir que el DAE «elimina el ruido» en general: lo reduce, y cada vez menos al subir p.

**Transición:** “Las dos variantes del DAE reducen el ruido, pero no por igual.”

## 6 · Código 10 vs código 2

**Imagen:** [denoising-04-code-comparison.png](assets/denoising-04-code-comparison.png). Izquierda: imágenes exactas según p. Derecha: píxeles erróneos con imágenes limpias.

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

**Pie:** **Evaluación:** 3.200 imágenes por nivel con ruido y 32 limpias · **Bandas y barra de error:** mínimo–máximo entre 3 semillas

**Guion oral [corregido]:** “A la izquierda, la fracción de imágenes que se recuperan sin ningún píxel mal. El código 10 queda arriba en todos los niveles: con p = 0,10, o sea 10 % de ruido, recupera exactas el 85 % de las imágenes, y el código 2, el 66 %. Pasa lo mismo con las 3 semillas: en píxeles erróneos, los rangos entre semillas no se solapan en ningún nivel. A la derecha, lo que pasa con las imágenes limpias: el DAE de código 2 les introduce errores, 0,26 píxeles de media, y el de código 10 las conserva todas. Esto apoya la hipótesis de que el código 10 es mejor, aunque, como solo comparamos 10 con 2, no podemos decir que 10 sea el óptimo.”

*Qué cambia respecto de las notas actuales:* las notas dicen «con p=10»; el nivel es p = 0,10. También se reescribieron las abreviaturas «c10», «c2», «q» y «osea».

**Para preguntas:**

| Imágenes exactas | p = 0 | p = 0,05 | p = 0,10 | p = 0,20 | p = 0,30 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Entrada ruidosa | 100 % | 16,6 % | 2,9 % | 0,1 % | 0 % |
| AE de 1.a (sin ruido al entrenar) | 100 % | 34,9 % | 18,8 % | 8,4 % | 4,5 % |
| DAE código 2 | 84,4 % | 76,7 % | 65,6 % | 35,5 % | 13,2 % |
| DAE código 10 | 100 % | 95,5 % | 85,2 % | 49,6 % | 17,3 % |

Con p = 0,20, la peor semilla del código 10 deja 2,966 píxeles erróneos y la mejor del código 2, 4,465. Con p = 0, el DAE código 2 deja entre 0,031 y 0,469 píxeles erróneos según la semilla. Los porcentajes del gráfico están redondeados (95,5 % se muestra como 96 %). Tres semillas son pocas: alcanzan para ver que el orden no depende de una inicialización, no para estimar la variabilidad con precisión.

**Cuidado:** no decir que 10 es el ancho óptimo ni que el resultado vale para otros anchos.

**Transición:** “Para verlo en imágenes concretas, estos son seis ejemplos con veinte por ciento de ruido.”

## 7 · Ejemplos con p = 0,20

**Imagen:** [denoising-05-examples-p020.png](assets/denoising-05-examples-p020.png). Seis caracteres (`` ` ``, a, h, p, x y DEL); filas Original, Entrada ruidosa, AE de 1.a, DAE código 2 y DAE código 10, con «N err» debajo. A la derecha, tres textos y una flecha que apunta a la fila del DAE código 10.

**Texto para copiar:**

> **Lectura:** N err contabiliza píxeles distintos del original (rojo: más de 1)
>
> **Selección de ejemplos:** fueron fijados antes de ver resultados; no se eligieron por favorables
>
> **Destacado:** con 20 % de ruido, el DAE código 10 recupera 4 de 6 caracteres exactos; los otros 2 quedan con solo 3 píxeles mal

**Pie:** sin pie; los tres textos laterales lo reemplazan.

**Guion oral:** “Cada columna es un carácter. Arriba está el original, después la entrada con veinte por ciento de ruido y abajo lo que devuelve cada modelo. El número indica cuántos píxeles quedaron mal, en rojo si es más de uno. El autoencoder de 1.a deja entre ocho y veintidós píxeles mal en cinco de los seis: en general devuelve otra figura. El DAE de código dos acierta dos y falla en los otros cuatro. El DAE de código diez recupera cuatro de los seis sin errores y deja tres píxeles mal en los otros dos. Estos ejemplos no los elegimos por favorables: los índices se fijaron antes de ver los resultados.”

**Para preguntas:** Son los caracteres 0, 1, 8, 16, 24 y 31, en su primera realización (índices de `evaluation/metrics.json → example_indices`). Píxeles erróneos:

- **Entrada ruidosa:** 7, 6, 8, 4, 7 y 8.
- **AE de 1.a:** 13, 22, 22, 8, 17 y 0.
- **DAE código 2:** 7, 0, 10, 8, 9 y 0. Es `winner-seed-43`, la mejor semilla de esa variante por validación.
- **DAE código 10:** 3, 0, 3, 0, 0 y 0. Es `latent-10-seed-44`, la seleccionada.

La diapositiva ya no dice qué semillas se muestran: si preguntan, son esas dos. `DEL` tiene todos los píxeles encendidos y es el patrón que más se distingue del resto.

**Cuidado:** no reemplazar estos ejemplos por otros más lucidos.

**Transición:** “Antes de cerrar, veamos qué límites tiene este análisis.”

## 8 · Límites del análisis

**Imagen:** [denoising-10-validation-curves.png](assets/denoising-10-validation-curves.png). Error de validación de las 6 corridas hasta la época 3000, con un punto en el mejor checkpoint de cada una y las últimas 500 épocas sombreadas.

**Texto para copiar:**

> **Límites**
>
> • Solo comparamos 2 y 10 como ancho del código
>
> • Solo caracteres conocidos: no medimos letras nuevas
>
> • 3 semillas y un solo tipo de ruido
>
> • 3000 épocas: 4 de 6 corridas eligen la última
>
> **¿Y con más épocas? Lo probamos**

**Pie:** **Validación:** error de píxeles medio con p = 0,05; 0,10 y 0,20 · **●** mejor checkpoint de cada corrida · **Zona sombreada:** últimas 500 épocas del protocolo

**Guion oral [corregido]:** “Antes de cerrar, veamos qué límites tiene este análisis, porque condicionan lo que podemos afirmar. Primero, solo comparamos dos anchos de código, dos y diez: no sabemos si diez es el óptimo. Segundo, todas las imágenes evaluadas son versiones ruidosas de los mismos treinta y dos caracteres, así que medimos recuperación de caracteres conocidos, no qué pasa con letras nuevas. Tercero, usamos tres semillas, que son pocas, y un solo tipo de ruido: inversión de píxeles. Y cuarto, el presupuesto de entrenamiento. Este gráfico muestra el error de validación de las seis corridas, y los puntos marcan el mejor checkpoint de cada una: cuatro de las seis lo tienen justo en la última época, y casi todas las curvas todavía bajaban. Eso sugiere que tres mil épocas pudieron quedar cortas. De todos estos límites, este era el que podíamos probar con un cambio mínimo, así que lo hicimos: extendimos el entrenamiento, y es lo que mostramos a continuación. Los demás quedan como trabajo futuro.”

*Qué cambia respecto de las notas actuales:* las notas dicen «lo mostramos en el anexo»; ahora el análisis de 6000 épocas está en el flujo principal.

**Para preguntas:**

- **¿Por qué solo 2 y 10?** El protocolo acordado fijó esas dos variantes; probar otros anchos queda como trabajo futuro.
- **¿Cómo evaluarían la generalización?** Con caracteres que no estén en el entrenamiento, algo que no se hizo.
- **¿Más épocas cambian las conclusiones?** No: con 6000 épocas ambos DAE mejoran y el código 10 sigue ganando.
- **Datos del gráfico:** según `results/denoising/selection.json`, el mejor checkpoint está en la época 3000 en `winner-seed-42`, `winner-seed-43`, `latent-10-seed-42` y `latent-10-seed-44`. En `winner-seed-44` está en la 2800 y en `latent-10-seed-43`, en la 2600. Entre las épocas 2500 y 3000, la validación bajó en 5 de las 6 corridas; la excepción es `latent-10-seed-43` (de 1,277 a 1,291). Por eso se dice «casi todas».

**Cuidado:** no decir que «3000 épocas quedaron cortas» como hecho: esta evidencia lo sugiere y la diapositiva siguiente lo muestra. No prometer que se probarán los otros límites: son trabajo futuro.

**Transición:** “Veamos entonces qué pasa con más épocas.”

## 9 · ¿Y si probamos con más épocas?

**Imagen:** [extension-exact-by-level.png](../results/denoising-extended-6000/figures/extension-exact-by-level.png). Imágenes exactas según p con 3000 épocas (punteada clara) y 6000 (sólida), para ambos DAE.

**Texto para copiar:**

> **Análisis adicional**
>
> • Duplicamos el entrenamiento: de 3000 a 6000 épocas
>
> • Ambos DAE mejoran; el código 10 sigue ganando
>
> • Con p = 0,10: 85 % → 91 % de imágenes exactas (código 10)
>
> **El presupuesto de 3000 épocas limitaba el resultado**

**Pie:** **Corridas:** las 6 continuaron desde la época 3000 · **Resultados:** media de 3 semillas · **Evaluación:** una sola vez, sobre la colección final

**Guion oral:** “Como en las últimas épocas la validación seguía bajando, hicimos un análisis adicional: retomamos las seis corridas desde su estado guardado en la época tres mil y las llevamos hasta seis mil. Con diez por ciento de ruido, el DAE de código diez pasa de recuperar de forma exacta el ochenta y cinco por ciento de las imágenes al noventa y uno, y el de código dos, del sesenta y seis al setenta y nueve. El código diez sigue siendo mejor en todos los niveles con ruido, así que la conclusión no cambia, pero las tres mil épocas limitaban los números. Es un análisis exploratorio, fuera del protocolo acordado: los resultados principales siguen siendo los de tres mil épocas.”

**Importante:** la diapositiva ya no dice en ningún lado que el análisis es exploratorio y fuera del protocolo; solo lo dice el guion. Hay que decirlo en voz alta.

**Para preguntas:**

| Píxeles erróneos (3000 → 6000 épocas) | p = 0,05 | p = 0,10 | p = 0,20 | p = 0,30 (fuera de rango) |
| --- | ---: | ---: | ---: | ---: |
| DAE código 2 | 0,553 → 0,290 | 1,327 → 0,852 | 4,654 → 3,957 | 9,032 → 8,598 |
| DAE código 10 | 0,111 → 0,068 | 0,500 → 0,326 | 2,935 → 2,510 | 7,183 → 6,883 |

| Imágenes exactas (3000 → 6000 épocas) | p = 0,05 | p = 0,10 | p = 0,20 | p = 0,30 |
| --- | ---: | ---: | ---: | ---: |
| DAE código 2 | 77 % → 87 % | 66 % → 79 % | 35 % → 47 % | 13 % → 18 % |
| DAE código 10 | 96 % → 97 % | 85 % → 91 % | 50 % → 57 % | 17 % → 21 % |

Es una extensión: no reemplaza a los resultados de 3000 épocas. La regla se fijó antes de ejecutar (`results/denoising-extended-6000/PROTOCOL.md`): el mejor checkpoint de cada corrida es el de menor error de validación en todo el rango 0–6000, y la variante ganadora se elige igual que antes. La colección de evaluación se usó una sola vez, después de fijar la selección.

Corrió con Python 3.11 y NumPy 2.3.5 (la campaña original, con 3.14.4 y 2.4.4). El punto de partida es el mismo estado guardado, pero la numérica de las épocas nuevas podría diferir levemente. Se verificó por SHA256 que los archivos de `results/denoising/` no cambiaron.

La brecha con p = 0,10 se achica de casi 20 puntos (66 % contra 85 %) a 12 (79 % contra 91 %). Con imágenes limpias, el DAE código 2 pasa de 0,26 a 0,11 píxeles erróneos de media; el DAE código 10 sigue en 0. Los rangos entre semillas siguen sin solaparse en todos los niveles con ruido.

**Cuidado:** no decir que el DAE «converge» ni que «más épocas siempre mejoran». No presentarlo como parte del protocolo.

**Transición:** “Para ver cómo evolucionó el entrenamiento época por época, miremos la validación.”

## 10 · Validación hasta 6000 épocas

**Imagen:** [extension-validation-curves.png](../results/denoising-extended-6000/figures/extension-validation-curves.png). Error de validación de 0 a 6000 épocas, media de 3 semillas con banda mínimo–máximo por variante, y una línea vertical en la época 3000 (protocolo original).

**Texto para copiar:** solo el título; la imagen ocupa la diapositiva (fondo oscuro).

**Pie:** **Error de validación:** promedio de p = 0,05; 0,10 y 0,20 · **Banda:** mínimo–máximo entre 3 semillas
**Sin meseta clara:** 2 de las 3 corridas del DAE código 10 tienen su mejor checkpoint exactamente en la época 6000

**Guion oral:** “Cada curva es el error de validación, es decir, cuántos píxeles quedan mal en promedio, a lo largo del entrenamiento; la banda es el rango entre las tres semillas. La línea vertical marca donde terminaba el protocolo original. Las dos variantes siguen bajando después de las tres mil épocas: tomando el mejor checkpoint de cada semilla, el código dos pasa de dos coma veintiocho a uno coma setenta y ocho píxeles erróneos, y el código diez, de uno coma veintitrés a uno coma cero dos. Dos de las tres corridas del código diez tienen su mejor punto justo en la última época, así que no vimos una meseta: no sabemos cuánto más podría mejorar.”

**Cierre sugerido (agregado):** como se quitó «Qué aprendimos y límites», esta es la última diapositiva de la sección. Para no terminar en un detalle técnico, conviene sumar al final del guion: “En resumen: entrenar con ruido es lo que permite reducirlo, mientras que un autoencoder común lo amplifica. Con 10 % de ruido, el DAE de código 10 recupera exactas el 85 % de las imágenes con el protocolo, y el 91 % con más épocas.”

**Para preguntas:** Cuánto seguiría mejorando es desconocido: no se probó más allá de 6000 épocas. La validación es la colección que se usa para elegir el mejor checkpoint; la evaluación es otra colección, que no interviene en la selección. Mejores épocas a 6000: DAE código 2, 5600, 5000 y 5700 (semillas 42, 43 y 44); DAE código 10, 6000, 6000 y 5700. La corrida seleccionada sigue siendo `latent-10-seed-44`. Los valores 2,28 → 1,78 y 1,23 → 1,02 promedian el mejor checkpoint de cada semilla. La curva, en cambio, muestra la media en cada época, por eso justo en la época 3000 puede quedar levemente por encima de 2,28 y 1,23.

**Cuidado:** no decir que la validación mide el resultado final (eso es la evaluación) ni que el entrenamiento «convergió».

**Transición:** “Con esto cerramos el punto 1.b y pasamos al autoencoder variacional.”

## Material quitado (por si preguntan)

Estas diapositivas ya no están en el deck. Sus figuras siguen en `assets/` por si hace falta mostrarlas.

- **Qué aprendimos y límites.** Con 10 % de ruido, el DAE código 10 recupera exactas el 85 % de las imágenes; con 20 %, el 50 %. Con p = 0,30 (fuera de rango), el error baja de 10,6 a 7,2 píxeles, pero solo el 17 % queda exacto.
- **Ejemplos con p = 0,10** ([denoising-06-examples-p010.png](assets/denoising-06-examples-p010.png)). Mismos 6 caracteres. Píxeles erróneos: entrada 2, 6, 2, 3, 4 y 2; AE de 1.a 5, 0, 10, 9, 13 y 0; DAE código 2 1, 0, 0, 8, 0 y 0; DAE código 10 0, 3, 0, 0, 0 y 0. En la «a», el DAE código 10 queda peor que los otros dos modelos.
- **Error por carácter** ([denoising-07-by-character.png](assets/denoising-07-by-character.png)). `DEL` es el más fácil con ruido bajo. Exploratorio, con solo 32 puntos: con ruido bajo, los caracteres más parecidos a otro tienden a ser los más difíciles. `report.md` informa una correlación de Spearman de −0,83 con p = 0,05, pero al recalcularla desde `by-character.csv` da −0,81: misma conclusión, distinto valor.

## Qué no afirmar en toda la sección

- Que el DAE «elimina el ruido» en general, que «generaliza» o que «funciona con letras nuevas»: todas las imágenes de evaluación son versiones ruidosas de los mismos 32 caracteres.
- Que 10 es el ancho óptimo: solo se compararon 2 y 10.
- Que p = 0,30 formó parte del entrenamiento: es extrapolación.
- Que el análisis de 6000 épocas es parte del protocolo, que el DAE «converge» o que «más épocas siempre mejoran».
- Siempre aclarar que 3 semillas son pocas y que la comparación con el AE de 1.a es casi controlada.

## Respaldo y reproducción

- Informe completo: `results/denoising/report.md` (la sección 6 trata las 6000 épocas). Protocolos: `denoising-data.md` y `results/denoising-extended-6000/PROTOCOL.md`.
- Figuras de 3000 épocas: `python ppt/generate_denoising.py` y `python ppt/generate_denoising_extra.py` desde `tp5/`. Leen solo `results/denoising/`; no entrenan ni cargan modelos.
- Figuras de 6000 épocas: `python -m src.denoise_extension figures --extension results/denoising-extended-6000 --output <carpeta nueva>`.
- Números de 6000 épocas: `results/denoising-extended-6000/comparison-3000-vs-6000.csv` y `selection-6000.json`.
