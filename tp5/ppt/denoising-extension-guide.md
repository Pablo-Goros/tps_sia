# TP5 · Guion del anexo de 1.b: de 3000 a 6000 épocas

Deck: **`TP5-con-1b-anexo.pptx`**, copia de `TP5-con-1b.pptx` con **2 diapositivas de anexo al final** (11 y 12), después de los dos respaldos. Las 10 diapositivas existentes no cambian. El guion oral de cada una está copiado en sus notas del orador.

Es un **análisis adicional exploratorio, fuera del protocolo acordado**: los resultados principales de la sección siguen siendo los de 3000 épocas. Fuente: sección 6 de `results/denoising/report.md` y la carpeta `results/denoising-extended-6000/`.

Nombres de modelo, iguales al resto del deck: «DAE código 2» y «DAE código 10». Etiqueta de sección: «3. Autoencoder de eliminación de ruido».

## 11 · Anexo · ¿Y con más épocas?

**Imagen:** [extension-exact-by-level.png](../results/denoising-extended-6000/figures/extension-exact-by-level.png), también disponible en SVG. Se usa tal como está en el repo. Texto alternativo: “Imágenes recuperadas exactas según p, con 3000 épocas (línea punteada clara) y 6000 (línea sólida). DAE código 10: con p = 0,10 pasa de 85 % a 91 % y con p = 0,20, de 50 % a 57 %. DAE código 2: de 66 % a 79 % y de 35 % a 47 %. La zona de p = 0,30 está sombreada como fuera del rango entrenado.”

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

**Guion oral:** “Como en las últimas épocas la validación seguía bajando, hicimos un análisis adicional: retomamos las seis corridas desde su estado guardado en la época tres mil y las llevamos hasta seis mil. Con diez por ciento de ruido, el DAE de código diez pasa de recuperar de forma exacta el ochenta y cinco por ciento de las imágenes al noventa y uno, y el de código dos, del sesenta y seis al setenta y nueve. El código diez sigue siendo mejor en todos los niveles con ruido, así que la conclusión no cambia, pero las tres mil épocas limitaban los números. Es un análisis exploratorio, fuera del protocolo acordado: los resultados principales siguen siendo los de tres mil épocas.”

**Pie:** **Análisis adicional exploratorio,** fuera del protocolo acordado · **Corridas:** las 6 continuaron desde la época 3000

**Resultados:** media de 3 semillas · **Evaluación:** una sola vez, sobre la colección final

| Píxeles erróneos por imagen (3000 → 6000 épocas) | p = 0,05 | p = 0,10 | p = 0,20 | p = 0,30 (fuera de rango) |
| --- | ---: | ---: | ---: | ---: |
| DAE código 2 | 0,553 → 0,290 | 1,327 → 0,852 | 4,654 → 3,957 | 9,032 → 8,598 |
| DAE código 10 | 0,111 → 0,068 | 0,500 → 0,326 | 2,935 → 2,510 | 7,183 → 6,883 |

| Imágenes exactas (3000 → 6000 épocas) | p = 0,05 | p = 0,10 | p = 0,20 | p = 0,30 |
| --- | ---: | ---: | ---: | ---: |
| DAE código 2 | 77 % → 87 % | 66 % → 79 % | 35 % → 47 % | 13 % → 18 % |
| DAE código 10 | 96 % → 97 % | 85 % → 91 % | 50 % → 57 % | 17 % → 21 % |

Media de 3 semillas sobre la colección de evaluación (3.200 imágenes por nivel con ruido), de `comparison-3000-vs-6000.csv`.

**Para preguntas:** Es una extensión: no reemplaza a los resultados de 3000 épocas, que siguen siendo los del protocolo. La regla se fijó antes de ejecutar (`results/denoising-extended-6000/PROTOCOL.md`): el mejor checkpoint de cada corrida es el de menor error de validación en todo el rango 0–6000 y la variante ganadora se elige igual que antes. La colección de evaluación se usó una sola vez, después de fijar la selección. Corrió con Python 3.11 y NumPy 2.3.5 (la campaña original, con 3.14.4 y 2.4.4): el punto de partida es el mismo estado guardado, pero la numérica de las épocas nuevas podría diferir levemente. Se verificó por SHA256 que los archivos de `results/denoising/` no cambiaron. La brecha con p = 0,10 se achica de casi 20 puntos (66 % contra 85 %) a 12 (79 % contra 91 %). Con imágenes limpias, el DAE código 2 pasa de 0,26 a 0,11 píxeles erróneos de media; el DAE código 10 sigue en 0. No se probó más allá de 6000 épocas y no hay una meseta clara.

**Cuidado:** No decir que el DAE «converge» ni que «más épocas siempre mejoran», ni presentar este análisis como parte del protocolo.

**Transición:** “Para ver cómo evolucionó el entrenamiento época por época, miremos la validación.”

## 12 · Anexo · Validación hasta 6000 épocas

**Imagen:** [extension-validation-curves.png](../results/denoising-extended-6000/figures/extension-validation-curves.png), también disponible en SVG. Se usa tal como está en el repo. Texto alternativo: “Error de validación, en píxeles erróneos por imagen y escala logarítmica, de la época 0 a la 6000 para DAE código 2 y DAE código 10: media de 3 semillas con banda mínimo–máximo y una línea vertical en la época 3000, donde terminaba el protocolo original. Las dos curvas siguen bajando después de la época 3000 y la del DAE código 10 queda por debajo.”

**Texto para copiar:**

> Solo el título; la imagen ocupa la diapositiva (fondo oscuro).

**Guion oral:** “Cada curva es el error de validación, es decir, cuántos píxeles quedan mal en promedio, a lo largo del entrenamiento; la banda es el rango entre las tres semillas. La línea vertical marca donde terminaba el protocolo original. Las dos variantes siguen bajando después de las tres mil épocas: tomando el mejor checkpoint de cada semilla, el código dos pasa de dos coma veintiocho a uno coma setenta y ocho píxeles erróneos, y el código diez, de uno coma veintitrés a uno coma cero dos. Dos de las tres corridas del código diez tienen su mejor punto justo en la última época, así que no vimos una meseta: no sabemos cuánto más podría mejorar.”

**Pie:** **Error de validación:** promedio de p = 0,05; 0,10 y 0,20 · **Banda:** mínimo–máximo entre 3 semillas

**Sin meseta clara:** 2 de las 3 corridas del DAE código 10 tienen su mejor checkpoint exactamente en la época 6000

**Para preguntas:** Cuánto seguiría mejorando es desconocido: no se probó más allá de 6000 épocas. La validación es la colección que se usa para elegir el mejor checkpoint (el puntaje promedia 9.600 imágenes, 3.200 por nivel con p = 0,05; 0,10 y 0,20); la evaluación es otra colección, que no interviene en la selección. Mejores épocas a 6000: DAE código 2, 5600, 5000 y 5700 (semillas 42, 43 y 44); DAE código 10, 6000, 6000 y 5700. La corrida seleccionada sigue siendo `latent-10-seed-44`. Los valores 2,28 → 1,78 y 1,23 → 1,02 promedian el mejor checkpoint de cada semilla; la curva muestra la media en cada época, por eso justo en la época 3000 puede quedar levemente por encima de 2,28 y 1,23.

**Cuidado:** No decir que la validación mide el resultado final (eso es la evaluación) ni que el entrenamiento «convergió».

**Transición:** “Con esto cerramos el anexo; volvemos a las preguntas.”

## Qué no afirmar en el anexo

- Que el DAE «converge» o que «más épocas siempre mejoran»: no se probó más allá de 6000 épocas y no hay meseta clara.
- Que esto es parte del protocolo acordado: es exploratorio; los resultados principales son los de 3000 épocas.
- Que el resultado generaliza a letras nuevas: todas las imágenes de evaluación son versiones ruidosas de los mismos 32 caracteres.
- Que 10 es el ancho óptimo: solo se compararon 2 y 10.
- Siempre aclarar que la colección de evaluación se usó una sola vez, después de fijar la selección.

## Respaldo y reproducción

- Método, resultados y limitaciones: sección 6 de `results/denoising/report.md`; regla fijada antes de ejecutar: `results/denoising-extended-6000/PROTOCOL.md`.
- Números: `comparison-3000-vs-6000.csv` (media, mínimo y máximo entre semillas, a 3000 y 6000 épocas) y `selection-6000.json` (mejor checkpoint por corrida y selección).
- Gráficos: los de `results/denoising-extended-6000/figures/`, sin cambios. Para regenerarlos en una carpeta nueva: `python -m src.denoise_extension figures --extension results/denoising-extended-6000 --output <carpeta nueva>`.

## 9 · Límites del análisis (diapositiva agregada)

Va **inmediatamente después de la diapositiva 8** («Qué aprendimos y límites»), que no se modificó: queda para decidir cuál de las dos conservar. Con esta inserción, los respaldos pasan a ser la 10 y la 11, y las diapositivas del anexo descritas arriba, la 12 y la 13.

**Imagen:** [denoising-10-validation-curves.png](assets/denoising-10-validation-curves.png), también disponible en SVG. Se usa tal como está en el repo. Muestra el error de validación de las 6 corridas, con un punto en el mejor checkpoint de cada una y las últimas 500 épocas sombreadas. Texto alternativo: “Curvas de validación de las seis corridas con el mejor checkpoint marcado; cuatro de las seis lo tienen en la época 3000”

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

**Guion oral:** “Antes de cerrar, veamos qué límites tiene este análisis, porque condicionan lo que podemos afirmar. Primero, solo comparamos dos anchos de código, dos y diez: no sabemos si diez es el óptimo. Segundo, todas las imágenes evaluadas son versiones ruidosas de los mismos treinta y dos caracteres, así que medimos recuperación de caracteres conocidos, no qué pasa con letras nuevas. Tercero, usamos tres semillas, que son pocas, y un solo tipo de ruido: inversión de píxeles. Y cuarto, el presupuesto de entrenamiento. Este gráfico muestra el error de validación de las seis corridas, y los puntos marcan el mejor checkpoint de cada una: cuatro de las seis lo tienen justo en la última época, y casi todas las curvas todavía bajaban. Eso sugiere que tres mil épocas pudieron quedar cortas. De todos estos límites, este era el que podíamos probar con un cambio mínimo, así que lo hicimos: extendimos el entrenamiento y lo mostramos en el anexo. Los demás quedan como trabajo futuro.”

**Para preguntas:**

- **¿Por qué solo 2 y 10?** El protocolo acordado fijó esas dos variantes; probar otros anchos queda como trabajo futuro.
- **¿Cómo evaluarían la generalización?** Con caracteres que no estén en el entrenamiento, algo que no se hizo.
- **¿El anexo cambia las conclusiones?** No: con 6000 épocas ambos DAE mejoran y el código 10 sigue ganando.
- **Datos del gráfico:** según `results/denoising/selection.json`, el mejor checkpoint está en la época 3000 en `winner-seed-42`, `winner-seed-43`, `latent-10-seed-42` y `latent-10-seed-44`; en `winner-seed-44` está en la 2800 y en `latent-10-seed-43`, en la 2600. Según los `history.json`, entre las épocas 2500 y 3000 la validación bajó en 5 de las 6 corridas; la excepción es `latent-10-seed-43` (de 1,277 a 1,291). Por eso el guion dice «casi todas».

**Cuidado:** No decir que «3000 épocas quedaron cortas» como hecho: la evidencia lo sugiere y el anexo lo muestra. No prometer que se probarán los otros límites: son trabajo futuro. No repetir acá resultados ya mostrados (85 %, 50 %, 17 %, 10,6 → 7,2). No decir que 10 es el óptimo ni que el resultado generaliza a letras nuevas.

**Transición:** “Veamos entonces qué pasa con más épocas.”
