# TP4 – Aprendizaje No Supervisado: decisiones tomadas y por qué

Este documento explica **qué** se decidió en cada punto del TP, **por qué** (con referencia a la
teórica correspondiente) y **qué evidencia** lo respalda (archivo en `resultados/`). Todo está
implementado desde cero con `numpy` siguiendo los algoritmos tal como se dieron en clase; no se
usa `scikit-learn` ni `minisom`.

---

## 0. Decisiones generales

| Decisión | Por qué |
|---|---|
| Python + numpy/pandas/matplotlib, sin librerías de ML | La consigna pide *implementar* Kohonen, Oja y Hopfield. PCA se hace calculando autovalores/autovectores de la matriz de correlaciones con `numpy.linalg.eigh` (procedimiento de la teórica 05, slide "Procedimiento"). |
| Estandarización z‑score antes de PCA, Kohonen y Oja | Teórica Kohonen 02.2 ("Importante: estandarizar todos los vectores"), teórica PCA 05.3 ("Covarianza vs Correlación": si las escalas difieren, la varianza la dominan las variables grandes → usar correlación = covarianza de variables estandarizadas), teórica Oja ("es mejor estandarizar todas las variables de entrada"). Acá `followers` va de 10⁴ a 10⁶ y los `_pct` de 0 a 100: sin estandarizar PCA sería "la componente followers". |
| Desvío con `ddof=1` (varianza muestral 1/(n‑1)) | Es la "varianza muestral" de la teórica PCA (slide "Medidas descriptivas"). Para 50 registros la diferencia con 1/n es irrelevante, pero se fija una convención. |
| Semillas fijas en todos los experimentos y promedios sobre varias semillas en los barridos | Kohonen, Oja y el ruido de Hopfield son estocásticos; sin semillas no se pueden reproducir ni comparar configuraciones. |
| Tests contra los ejemplos numéricos de las diapositivas (`tests/test_teorica.py`) | Verifica que `W` de Hopfield coincide con la slide 29, que ζ=(1,−1,−1,−1)→ξ¹ como en las slides 31‑32, que los autovalores del ejercicio de PCA dan 6/3/2, y la letra J del enunciado. |

---

## 1. Campañas de marketing

### 1.1 EDA y selección de variables (`ej1_eda.py` → `resultados/eda/`)

**Qué se encontró**

* 50 creadores × 31 variables numéricas (más `creator_id`, `name`).
* Las escalas son incomparables (`boxplot_crudo.png` vs `boxplot_estandarizado.png`), exactamente el
  caso de la slide "Ejemplo" de estandarización.
* **Cuatro columnas son combinaciones exactas de otras** (`derivadas.txt`):
  `price_ars = 1500·price_usd`, `organic_posts_pct = 100 − sponsored_posts_pct`,
  `total_engagement_per_1000 = likes + comments + saves + shares`,
  `estimated_argentina_reach = average_reach · argentina_audience_pct / 100`.
* Dos grupos composicionales: `topic_*` suman 100 y `content_*` suman 100.
* Bloques muy correlacionados (`pares_mas_correlacionados.csv`): *escala* (`followers`, `average_reach`,
  `average_impressions`, `average_video_views`, `price_usd`: r = 0.82–0.997) y *formato*
  (`video_posts_pct`, `short_form_posts_pct`, `trend_posts_pct`: r = 0.95–0.97, y con `likes_per_1000` r≈0.9).
  También `average_caption_words` vs alcance: r ≈ −0.86, y `content_tutorial` vs `content_entertainment`: r = −0.84.
* Cada creador tiene un tema claramente dominante (60–80 %): 15 fútbol, 13 perros, 11 maquillaje, 11 arte.
  Se guarda `tema_dominante.csv` **solo como etiqueta para colorear gráficos e interpretar**; nunca
  entra como entrada a los modelos (el aprendizaje es no supervisado).

**Decisión: se eliminan las 4 derivadas exactas y las 2 más redundantes del bloque de escala → 25 variables.**

* Las 4 derivadas exactas (`price_ars`, `organic_posts_pct`, `total_engagement_per_1000`,
  `estimated_argentina_reach`): la teórica de PCA dice que variables muy correlacionadas = *información
  redundante*; una copia exacta (r = 1) no agrega nada y duplica el peso de esa magnitud en toda
  distancia euclídea (Kohonen) y en la correlación (PCA/Oja).
* Del bloque de escala (`followers`, `average_reach`, `average_impressions`, `average_video_views`,
  `price_usd`; r entre 0.82 y 0.997) se sacan **`average_reach` y `followers`**, que son las dos más
  correlacionadas con el resto del bloque en `correlacion.png` (`average_reach`: 0.997 con video_views,
  0.987 con impressions, 0.918 con followers; `followers`: 0.91 con las tres). Quedan
  `average_impressions` y `average_video_views` como medida de "cuánta gente ve al creador" y
  `price_usd` como perfil comercial. Son 4 variables midiendo casi lo mismo: dejarlas todas hacía
  que la "escala" entrara 4 veces en la distancia euclídea de Kohonen y dominara PC1 todavía más.
* Por qué no podar más allá de ese bloque: la consigna pide agrupar "en términos de todas sus
  variables" y considerar "el perfil comercial, las características de su audiencia y el contenido";
  el problema de la agencia es justamente que elegía solo por visibilidad. La redundancia *no exacta*
  restante (p. ej. `video/short_form/trend_posts_pct`, r ≈ 0.95) es lo que PCA debe resumir.
* `followers` se mantiene **fuera del modelo pero como referencia**: es la "estrategia vieja" contra la
  que el punto 4 pide comparar (`rank_followers` en `indice_oja.csv`).
* Se verificó que con 27 variables (sin sacar `followers`/`average_reach`) los resultados son los
  mismos: PC1 pasa de 49.7 % a 47.6 % con las mismas cargas, el índice de Oja tiene Spearman 0.9975
  con el de 27 (mismo top‑10) y el mapa de Kohonen es el mismo agrupamiento con la grilla rotada
  (96 % de los pares de creadores conservan la relación juntos/separados).
* Por qué conservar las 4 `topic_*` y las 4 `content_*` aunque cada grupo sume 100: por
  interpretabilidad de los planos de Kohonen y del biplot (queremos ver "arte" y "perros" como
  direcciones propias). El costo es que la matriz de correlaciones tiene **2 autovalores nulos**
  (PC24 y PC25); como la varianza explicada se reparte sobre los 23 restantes esto no afecta nada.
  El script lo reporta explícitamente.

### 1.2 PCA (`ej1_pca.py` → `resultados/pca/`)

**Método**: X estandarizada → matriz de correlaciones `S = ZᵀZ/(n−1)` → `eigh` → ordenar autovalores
desc. → cargas = autovectores → `Y = Z·V` (slides "Buscamos AyA" y "Procedimiento").

**Resultados** (`autovalores.csv`, `scree.png`):

| | PC1 | PC2 | PC3 | PC4 | PC5 | PC6 |
|---|---|---|---|---|---|---|
| autovalor | 11.89 | 2.40 | 1.87 | 1.47 | 1.29 | 1.18 |
| % varianza | 47.6 | 9.6 | 7.5 | 5.9 | 5.2 | 4.7 |
| acumulada | 47.6 | 57.1 | 64.6 | 70.5 | 75.7 | 80.4 |

**¿Se reduce la dimensionalidad?** Sí para *visualizar e interpretar*, no para *entrenar Kohonen*:

* PC1 sola explica casi la mitad de la varianza; con 2 componentes el 57 %, con 6 el 80 %, con 10 el 90 %.
  Para el biplot (2D) se pierde el 43 % de la varianza, así que el biplot se usa como mapa cualitativo.
* Kohonen se entrena con las 25 variables estandarizadas (lo pide la consigna) y PCA se usa para
  *validar* lo que la red encontró (ver 1.3).
* El Índice del punto 3 es, por construcción, la reducción a 1 dimensión (PC1).

**Interpretación de las componentes** (`cargas.csv`, `cargas_pc1_pc2.png`, `biplot.png`):

* **PC1 (47.6 %) = "alcance masivo y formato viral" vs "profundidad/comunidad"**. Cargas positivas:
  `trend/video/short_form_posts_pct` (0.26), `average_impressions` (0.26), `average_video_views` (0.26),
  `price_usd` (0.25), `likes_per_1000`, `content_entertainment_pct`, `non_follower_reach_pct` (0.22), `shares`.
  Cargas negativas: `average_caption_words` (−0.27), `saves_per_1000` (−0.24), `profile_visits_per_1000`,
  `comments_per_1000`, `content_tutorial_pct`, `average_watch_time_seconds`, `content_review_pct` (≈ −0.2).
  Los `topic_*` casi no pesan en PC1 (|carga| ≤ 0.14): **PC1 no es "de qué habla" sino "cómo y a cuántos llega"**.
  Un creador con PC1 alto es grande, caro, hace shorts/tendencias/entretenimiento y le dan muchos
  likes y shares; uno con PC1 bajo es chico, escribe captions largos, hace tutoriales/reseñas y
  genera saves, comentarios, visitas al perfil y watch time (señales de audiencia involucrada).
* **PC2 (9.6 %) = "arte/showcase" vs "perros/Argentina/entretenimiento"**: `content_showcase_pct`,
  `topic_art_pct` (≈ 0.4), `likes`, `short_form`, `content_review`, `monthly_posts` (+0.2) contra
  `argentina_audience_pct` (−0.36), `topic_dogs_pct` (−0.28), `content_entertainment_pct` (−0.26).
* Convención de signo: un autovector y su opuesto son igualmente válidos (slide "Índice – autovector
  opuesto"). Se orienta PC1 para que `average_impressions` sea positiva ("más PC1 = más visibilidad") y PC2 para
  que `topic_art_pct` sea positiva. Es solo lectura, no cambia nada numérico.
* En el biplot los temas se separan: perros abajo‑izquierda, arte arriba, maquillaje y fútbol grandes
  a la derecha. Hay fútbol de dos tipos: masivo (derecha) y de nicho con comunidad (izquierda).

### 1.3 Kohonen (`ej1_kohonen.py`, `configs/kohonen.json` → `resultados/kohonen/`)

**Implementación (fiel a las slides "Algoritmo", "Regla de Kohonen", "Vecindario", "Inicialización")**

* Grilla rectangular k×k; cada neurona tiene W_j ∈ ℝ²⁵.
* Ganadora por distancia euclídea sobre datos estandarizados.
* Vecindario `N_k(i) = {n : ‖n − n_k‖ ≤ R(i)}` con distancia euclídea *en la grilla* (R = 1 → 4
  vecinos, R = √2 → 8 vecinos, slide "Vecindario"). Se usa `≤` y no `<` porque la slide dice que
  con R = 1 "solamente se actualizan las neuronas vecinas pegadas"; con `<` estricto R = 1 solo
  actualizaría a la ganadora.
* Actualización `W_j ← W_j + η(i)(Xᵖ − W_j)` para j en el vecindario.
* `R(i)` decrece linealmente desde R(0) hasta 1 (slide: "R(0) puede ser el tamaño total de la grilla
  y va decreciendo hasta R = 1"); también se probó constante.
* `η(i)` decreciente a 0. Se implementaron `η₀/(1+época)` (el "η(i)=1/i" de la slide, aplicado por
  época para que no muera en la primera decena de registros) y lineal hasta 0.01.
* Inicialización con muestras del conjunto (slide "Inicialización": evita unidades muertas) vs. uniforme.
* Iteraciones totales = épocas × 50 registros; 500 épocas = 25 000 iteraciones (orden de "500·N").

**Métricas para evaluar performance** (no hay etiquetas, así que se miden las dos cosas que la teórica
promete que una SOM debe lograr):

* **Error de cuantización (QE)**: distancia media de cada registro a su neurona ganadora. Mide que "los
  pesos se parecen a los datos de entrada" (slide "Convergencia").
* **Error topográfico (TE)**: fracción de registros cuya 1.ª y 2.ª neurona más cercanas no son vecinas
  pegadas. Mide que "neuronas vecinas contienen datos con algún grado de similitud" (slide
  "Vecindario"), es decir que el mapa preservó la topología.
* **Unidades muertas**: neuronas sin ningún registro (slide "Inicialización").

**Barrido** (`busqueda.csv`, `busqueda_marginal.png`, `busqueda_qe_te.png`): k ∈ {3,4,5,6}, R(0) ∈ {k, 2, 1},
η₀ ∈ {0.1, 0.5, 0.9}, esquema de η ∈ {inverso, lineal}, épocas ∈ {100, 500}, init ∈ {muestras, uniforme},
3 semillas cada uno → 864 entrenamientos. Conclusiones:

| Hiperparámetro | Efecto observado | Decisión |
|---|---|---|
| **k** | QE baja al crecer k (3: 3.45 → 6: 2.63) pero solo porque hay más prototipos: con 50 registros, k=5 deja ≈3.5 neuronas muertas y k=6 ≈10 de 36. k=3 (9 neuronas, ~5.5 creadores por neurona) mezcla temas. | **k = 4** (16 neuronas, ~3 creadores por neurona, 0–1 muertas). La teórica reconoce que "no hay un criterio demostrado para definir el tamaño de la grilla": se elige el mayor k que no genera huecos. |
| **R(0)** | R=1 constante (sin vecindario = competitivo puro) da el QE más bajo pero el TE más alto (más del doble que R(0)=k) y más unidades muertas: no auto‑organiza. R(0)=k→1 da TE≈0. | **R(0) = k = 4, decreciendo linealmente a 1**: es el setup de la teórica y el que produce un mapa donde la posición significa algo (necesario para la Matriz U). |
| **η₀ / esquema** | Diferencias de QE < 2 %. El esquema inverso `η₀/(1+época)` cae al piso en ~50 épocas y enmascara el efecto de las épocas; el lineal aprovecha todo el entrenamiento y da TE menor (0.006 vs 0.012) y menos unidades muertas. | **η: 0.5 → 0.01 lineal**. η₀ = 0.5 cumple "η(0) < 1" y con R grande al principio mueve todo el mapa; después afina. |
| **épocas** | 100 → 500 mejora QE ~1.5 % con esquema lineal. | **500 épocas** (25 000 iteraciones, cuesta < 1 s). |
| **init** | Casi igual en QE; muestras da menos muertas con k grande. | **muestras** (recomendación explícita de la teórica). |

La configuración final es además de las **más reproducibles**: desvío de QE entre semillas 0.006
contra hasta 0.05 en otras configuraciones con k=4 (y 0.12 con k=6) (`busqueda.csv`, columna `QE_std`).

**Modelo final**: QE = 3.15, TE = 0.00, 1 unidad muerta (de 16), U media = 2.51 (`entrenamiento.png`).

**Análisis de la grilla** (`hits.png`, `hits_por_tema.png`, `planos_*.png`, `resumen_neuronas.txt`, `perfiles_neuronas.csv`):

```
            col 0                   col 1                  col 2                    col 3
fila 0  profundidad máx. (2)    perros/tutorial (5)    perros frecuentes (2)    AR‑céntrico/mezcla (3)
fila 1  reseñas/captions (4)    fútbol reseña (1)      fútbol entretenim. (2)   maquillaje viral (3)
fila 2  showcase/patroc. (4)    fútbol AR (1)          fútbol nicho‑pago (4)    masivo patrocinado (2)
fila 3  arte showcase (3)       arte viral (7)         — (muerta)               MASIVOS premium (7)
```

* **Esquina inferior derecha** (3,3): `@mateconideas`, `@brillourbano`, `@maquillajereal`,
  `@golconmate`, `@futbolsinvueltas`, `@crearenreels`, `@galeriacotidiana`: impressions/views a +2 σ,
  price +1.7, entretenimiento +1.2; comentarios/captions/saves/watch_time a −1.2 σ. Son los
  "mega‑creadores" sin importar el tema (hay fútbol, maquillaje, perros y arte en la misma neurona).
* **Esquina superior izquierda** (0,0)+(0,1)+(1,0): `@ideasconhuella`, `@mundoanimal.ar` (profile_visits
  +2.0, captions +1.9, saves +1.7, comentarios +1.6), `@huellasyconsejos`, `@consejoscaninos`,
  `@guiadehuellas`, `@perrosfelices.ar`, `@clubdelasmascotas` (perros +1.8, saves +1.3, tutoriales +1.0),
  y en (1,0) `@pasionalbirroja`, `@futbolentrepases`, `@brochasycolor`, `@reelargentino` (reseñas +1.0,
  captions +1.2, patrocinio +1.0). Creadores chicos con audiencia que guarda, comenta y mira hasta el final.
* **Fila superior**: perros (temas +1.6…+1.8 en (0,1) y (0,2)).
* **Columna derecha**: maquillaje, de viral/entretenimiento (1,3) a masivo (3,3).
* **Fila inferior izquierda** (3,0)+(3,1): arte; (3,1) junta a los 7 creadores de arte con shorts/likes altos.
* **Centro** (1,2), (2,1), (2,2): fútbol; (2,2) es fútbol caro con alcance a no‑seguidores (`@vidaperruna`,
  `@tacticavisual`, `@lookdecancha`, `@makeupconsejos`).
* Los temas quedan en regiones contiguas aunque la red nunca vio la etiqueta (`hits_por_tema.png`):
  perros arriba, arte abajo‑izquierda, maquillaje a la derecha, fútbol en el centro‑izquierda.

**¿Coincide con PC1/PC2?** Sí (`som_vs_pca.png`): el PC1 promedio por neurona crece monótonamente en
diagonal desde (0,0) (−5.8) hasta (3,3) (+4.9), y el PC2 promedio es máximo abajo‑izquierda (arte, +2.2)
y mínimo arriba (perros/AR, −1.7). El mapa de 16 neuronas, entrenado con las 25 variables, reproduce
el plano PC1‑PC2 **y además** separa cosas que el biplot mezcla (p. ej. maquillaje viral (1,3) vs.
fútbol nicho‑pago (2,2), que en PC1‑PC2 se superponen).

**Matriz U** (`umatrix.png`; promedio de la distancia euclídea entre W_j y los W de sus 8 vecinas, slide
"Matriz U"):

* Valores bajos (1.7–2.3) en las esquinas (0,0), (3,0), (3,3) y en (2,3), (0,3): son *regiones
  homogéneas* (profundidad, arte, masivos). Sus neuronas vecinas tienen prototipos parecidos → un mismo
  cluster ocupa varias neuronas.
* Valores altos (2.9–3.1) en (0,2), (1,1), (1,2), (2,2): una **cresta vertical** en la columna 2 (filas
  0–2) que se extiende a (1,1). **Una distancia alta entre neuronas vecinas indica que el mapa tuvo que
  "estirarse" para cubrir un salto en el espacio de datos: es una frontera entre clusters, no una
  transición suave.** Acá separa el bloque "chico/profundo/perros/arte" (izquierda) del bloque
  "masivo/viral/maquillaje" (derecha), que es exactamente el eje PC1. Las neuronas sobre la cresta
  tienen pocos registros (1–2 creadores: `@pasiondepotrero`, `@futbolcononda`, `@patitasenred`,
  `@jugadayestilo`) o están muertas ((3,2)): son prototipos intermedios donde casi no hay datos reales.
* La Matriz U también explica por qué k=5 y k=6 generan unidades muertas: la red necesita neuronas
  para tender el puente sobre ese salto aunque no haya creadores ahí.

### 1.4 Índice con la regla de Oja (`ej1_oja.py`, `configs/oja.json` → `resultados/oja/`)

**Implementación** (slide "Algoritmo"): neurona lineal, `w` inicial uniforme en [0,1],
`y = x·w`, `w += η·y·(x − y·w)`, datos estandarizados (media 0), registros en orden aleatorio por época.

**Tasa de aprendizaje – por qué η = 10⁻³ constante** (`barrido_eta.csv`, `barrido_eta.png`):

* La teórica propone η₀ = 1/(1.2·λ₁) y, como λ₁ no se conoce, "estandarizar y empezar con η₀ ≤ 0.5"
  o bien "η = 10⁻³". Con nuestros datos λ₁ = 11.89 → η₀ teórico = 0.070.
* **Con 25 variables estandarizadas, η ≥ 0.05 diverge siempre** (overflow en la primera época), incluso
  0.070 y los esquemas 0.5→1/(1+época) y 0.1→1/(1+época). La razón: la cota 1/(1.2λ₁) es para el
  promedio, pero la regla se aplica registro a registro y para los creadores extremos `y² ≈ 40`
  (PC1 ≈ 6.2), con lo que `η·y²` supera 2 y `w` explota antes de que η tenga tiempo de bajar.
  La recomendación "η₀ ≤ 0.5" vale cuando hay pocas variables (λ₁ ≈ 1–2); con 25 no.
* η = 10⁻²: converge pero oscila (cos final 0.977, ‖w‖ = 1.033). η = 10⁻³: cos = 0.9999 en 6 épocas,
  ‖w‖ = 1.003. η = 10⁻⁴: cos = 1.0000 pero tarda 69 épocas. Se elige **10⁻³, 300 épocas** (la opción
  literal de la slide "Implementación").

**Resultado**: `cos(w, autovector PC1) = 0.99992`, `‖w‖ = 1.0031`, diferencia máxima por carga 0.0068
(`pesos_vs_pc1.png`); correlación entre el índice de Oja y la proyección PC1 de PCA: **r = 0.99999**
(`indice_oja_vs_pc1.png`). Es decir, la neurona de Oja encontró iterativamente el mismo autovector que
el cálculo exacto, como demuestra la teórica (slide "Convergencia").

**¿Qué representa el índice?** Es PC1: una suma ponderada de las 25 variables con los pesos de
`pesos_vs_pc1.csv`. Ordena a los creadores por **"escala de alcance masivo y formato viral"**: positivo
= muchas impresiones/views, caro, shorts, tendencias, entretenimiento, muchos likes y
shares; negativo = chico, captions largos, tutoriales/reseñas, saves, comentarios, visitas al perfil,
watch time alto. Top: `@futbolsinvueltas` (6.2), `@galeriacotidiana`, `@brillourbano`, `@crearenreels`,
`@golconmate`, `@makeupconsejos`. Fondo: `@mundoanimal.ar` (−6.8), `@consejoscaninos`, `@huellasyconsejos`
(`indice_oja.csv`).

**¿Sería correcto elegir automáticamente al creador con el valor más alto para cualquier campaña? No**, por tres razones que salen de los resultados:

1. **El índice casi no mira el tema** (|carga| ≤ 0.14 en `topic_*`). El n.º 1 global es de fútbol, el 2
   de arte, el 3 de maquillaje y el 4 (`@crearenreels`) de perros. Para un refugio el "mejor" sería un
   creador de fútbol.
2. **El índice premia precisamente lo que a la agencia le estaba fallando.** El enunciado dice que
   seleccionaban por seguidores y las campañas no convertían. Aunque `followers` **no** entra en el
   modelo, el índice tiene Spearman = 0.90 con el ranking por `followers` (`indice_oja.csv`), o sea que
   es una versión refinada de la misma
   estrategia: mide visibilidad, y además `price_usd` carga positivo (los de arriba son los caros).
3. **Las señales de conversión cargan negativo.** Saves, comentarios, visitas al perfil, watch time y
   tutoriales/reseñas son lo que haría que alguien consulte por una adopción o compre un producto, y
   están en el extremo *bajo* del índice. Para los refugios (objetivo: consultas y adopciones) el
   ranking adecuado es casi el inverso dentro del tema perros.

Dónde sí sirve: como **índice de escala** para campañas de *awareness*, para fijar un presupuesto (está
alineado con el precio) o para elegir dentro de un tema (`ranking_por_tema.csv`). Y sirve para
**detectar desacuerdos con la estrategia por seguidores**: `@makeupconsejos` (fútbol) es 20.º por
seguidores y 6.º por índice (formato viral, no‑seguidores 1.2 σ) o `@galeriacotidiana` 11.º → 2.º; al
revés, `@cosmeticaconsciente` 10.º por seguidores → 28.º por índice y `@brilloytendencia` 7.º → 18.º.

### 1.5 Decisiones para las campañas (síntesis de la evidencia)

**Campaña de fútbol: ¿alcance masivo, descubrimiento o profundidad educativa?**
Para "marcas de ropa de entrenamiento" el objetivo es vender. Kohonen muestra dos perfiles de fútbol
muy distintos: la esquina (3,3)/(2,3) (`@futbolsinvueltas`, `@golconmate`, `@tribunacreativa`: impresiones
+2 σ, `non_follower_reach_pct` alto = descubrimiento, pero watch_time −1.1, reseñas −1.0, captions
−1.2) y la columna izquierda (1,0)/(2,0) (`@pasionalbirroja`, `@futbolentrepases`, `@canchaaldia`: chicos,
pero con reseñas/showcase/saves/comentarios altos y patrocinio alto, es decir ya hacen contenido de
producto y su audiencia responde). Recomendación: **combinar**: un creador masivo de la esquina para
*descubrimiento* (`non_follower_reach_pct` alto significa que llega a gente que no lo sigue) y 2–3 de la
columna izquierda / neurona (2,2) (`@lookdecancha`, `@tacticavisual`: fútbol + precio alto + no‑seguidores) para
*profundidad/reseña* del producto. Priorizar solo alcance masivo repite el error original; priorizar
solo profundidad no genera descubrimiento.

**Refugios (adopciones)**: el bloque perros está en la fila superior, en el extremo *bajo* del índice:
(0,1) `@huellasyconsejos`, `@consejoscaninos`, `@guiadehuellas`, `@perrosfelices.ar`, `@clubdelasmascotas`
(perros +1.8, saves +1.3, captions +1.3, tutoriales +1.0) y (0,2) `@estiloydetalle`, `@adoptafelicidad`
(frecuencia +1.1, watch time +1.0, visitas al perfil +0.9).
Son baratos (75–530 USD) y su audiencia guarda y comenta → ideales para "impulsar consultas". El único
perro masivo (`@crearenreels`, 3 800 USD, PC1 = 4.9) sirve para visibilidad pero su audiencia no
interactúa en profundidad.

**Belleza**: maquillaje masivo (`@brillourbano`, `@mateconideas`, `@maquillajereal`, 3 000–6 300 USD) para
lanzamientos; `@bellezasimple`/`@tallerurbano`/`@brilloytendencia` (neurona (1,3): entretenimiento +1.3,
video +0.8) para contenido viral de precio medio, y `@cosmeticaconsciente` (0,3: shares +0.6, tutoriales
+0.5, audiencia argentina) para reseñas.

**Arte**: fila inferior izquierda. `@galeriacotidiana` es el único arte masivo (cae en (3,3)); el resto
son chicos: (3,1) los de shorts/likes (`@pincelcotidiano`, `@artealpaso`, `@entrelienzos`…) y (3,0)/(2,0)
los de showcase + reseñas + patrocinio alto (`@trazosyreels`, `@canvasycafe`, `@manosalaobra`,
`@arteentusmanos`) — perfil "muestro materiales", ideal para comercios de arte.

**Selección por `followers` vs. el sistema**: por seguidores se obtiene casi el mismo top (Spearman 0.90
con el índice, aun sin usar `followers` como variable) → es rápida, transparente y barata de mantener, pero ciega al tema, al precio por alcance y a
las señales de conversión; y confunde "muchos seguidores" con "llego a quien me interesa"
(`argentina_audience_pct` tiene carga ≈ 0 en PC1 y es la dimensión principal de PC2). El sistema
propuesto cuesta más explicar al cliente pero separa perfiles (Kohonen) y da un ranking continuo (Oja)
que se puede restringir por región del mapa.

**Mantenimiento del sistema**

* *Nueva campaña*: no hay que reentrenar nada. Se elige la región del mapa que corresponde al objetivo
  (descubrimiento → esquina inf. derecha; conversión/educación → esquina sup. izquierda; tema → bloque) y se
  ordena dentro de ella por el índice. Los `Standardizer` guardan media y desvío para proyectar.
* *Nuevos creadores*: se estandarizan con la media/desvío **del entrenamiento**, se busca su neurona
  ganadora y su índice `x·w`. No requiere reentrenar; si entran muchos (o muy distintos, p. ej. uno de
  10 M de seguidores) conviene reentrenar porque la grilla de 4×4 fue dimensionada para ~50 registros y
  la estandarización cambia.
* *Nuevas variables*: cambia la dimensión de W (Kohonen) y de w (Oja) y la matriz de correlaciones → hay
  que rehacer EDA (¿es derivada de otra?), PCA, barrido y reentrenar todo. Es la operación cara.
* *Actualización mensual*: las métricas son "medidas a fin de mes", así que media/desvío, PC1 y el mapa
  cambian mes a mes. Lo razonable es reentrenar mensualmente con semilla fija y **comparar** QE/TE,
  varianza de PC1 y cargas con el mes anterior; si las cargas de PC1 se mantienen (hoy: escala + formato)
  el índice sigue siendo comparable; si cambian, los rankings históricos no son comparables entre sí.
  Hay que recordar también que la orientación del autovector es arbitraria: se fija el signo con
  `average_impressions > 0` para que "más índice = más visibilidad" todos los meses.
* *Limitaciones*: 50 registros es poco para una SOM (de ahí k=4); el dataset sintético tiene nombres
  que no coinciden con el tema (`@vidaperruna` es fútbol), lo que recuerda que hay que mirar las
  variables y no el nombre; y todo el análisis supone que las métricas del mes son representativas.

---

## 2. Memoria asociativa (Hopfield)

### 2.0 Implementación (`src/hopfield.py`)

* `W = (1/N)·K·Kᵀ` con `K` = patrones en columna y diagonal en 0 (slides 21 y 30). Simétrica.
* `S(0) = ζ`; `S(t+1) = sign(W·S(t))`, y si `h_i = 0` la neurona conserva su estado (slide 15).
* Se itera hasta que `S` no cambia (estado estable) o se repite un estado anterior (ciclo).
* Energía `H = −½ Σ w_ij S_i S_j` calculada en cada paso.
* **Actualización sincrónica** (todas las neuronas a la vez) porque es la que usa el ejemplo de la
  teórica (`S(t+1) = sign(W S(t))`). Se implementó también la asincrónica (una neurona por vez, orden
  aleatorio), para la cual la demostración de que H nunca crece aplica exactamente; en los
  experimentos la energía nunca aumentó con la sincrónica, pero sí aparecieron ciclos de período 2
  (hasta 22 % de los casos con 50 % de ruido en letras), cosa que la teórica anticipa ("puede
  desembocar en ciclos").
* Ruido: cada componente se invierte con probabilidad p ("sal y pimienta"). Se eligió este modelo porque
  el estado es binario y la teórica define la consulta como un patrón "con perturbaciones".

### 2.1 Letras 5×5 (`ej2_letras.py` → `resultados/hopfield_letras/`)

**Qué letras almacenar – por análisis de ortogonalidad.** Se dibujaron las 26 letras en 5×5 (la J es
exactamente la del enunciado) y se calculó `ξ^μ·ξ^ν` para todos los pares
(`ortogonalidad_abecedario.png`) y para los 14 950 grupos de 4 (`mejores_grupos.csv`,
`peores_grupos.csv`). La teórica (slides "Estabilidad") muestra que el patrón ν es estable si el
*crosstalk* `(1/N) Σ_{μ≠ν} ξ_i^μ (ξ^μ·ξ^ν)` no da vuelta el signo de `ξ_i^ν`, lo que exige
patrones "más o menos ortogonales". **Se eligen G, R, T, V**: todos los productos internos valen ±1
(sobre N = 25), el mínimo posible con 25 bits (impar). El peor grupo, H M N W, tiene productos de
hasta 21/25.

**Capacidad.** `p ≤ 0.15·N = 3.75` → con 4 letras estamos en el límite teórico. Aun así las 4 son
puntos fijos porque son casi ortogonales (el 0.15 N es para patrones *aleatorios*). `capacidad.csv` /
`capacidad.png`: agregando letras en orden greedy de menor crosstalk, hasta p = 5 el 100 % de los patrones
es estable y la recuperación con 10 % de ruido ≥ 97 %; en p = 6 cae a 83 % / 69 %; en p = 8 a 38 % / 27 %;
en p ≥ 11 ningún patrón es estable. La transición ocurre justo donde `crosstalk_max` salta de 0.04 a
0.28–0.36, confirmando que lo que manda es la ortogonalidad, y que la cota 0.15 N es del orden correcto.

**Ortogonales vs. correlacionadas** (`ortogonal_vs_correlacionado.png`): con HMNW almacenadas la red
recupera el 25 % de las consultas **incluso sin ruido** (3 de las 4 letras ni siquiera son puntos
fijos); con GRTV el 100 %.

**Ruido** (`ruido.csv`, `ruido.png`, 200 repeticiones × 4 letras por nivel): recuperación perfecta hasta
8 % (2 bits), 99 % con 12 % (3 bits), 91 % con 20 % (5 bits), 71 % con 28 % (7 bits), 55 % con 32 %, 32 %
con 40 % (10 bits) y 10 % con 50 % (la consulta ya no tiene información: en ese caso la red cae al
patrón negado 7 %, a otro patrón 52 % o a un espurio 32 %). Las iteraciones crecen suavemente de 1 a
≈3.5: la red converge muy rápido porque N es chico.

**Energía** (`energia.png`, `ejemplo_evolucion.png`): los 4 patrones tienen H = −10.56 (el mismo valor
porque son casi ortogonales); una consulta con 12 % de ruido arranca en H ≈ −5.3 y baja en un paso al
mínimo; H nunca sube entre pasos. El estado espurio tiene H = −7.36: es un mínimo local **menos
profundo** que los patrones almacenados, exactamente lo que describe la slide "Estados espurios".

**Estado espurio** (`espurio.png`, `espurio.json`): se ingresó una letra con 40 % de bits invertidos y la
red convergió en 4 iteraciones a un patrón estable que no es ninguna de G/R/T/V ni sus negativos
(overlaps −0.68, 0.28, 0.36, −0.28): un punto fijo "mezcla", con parecido a −G y a T. El script
verifica además si el espurio es `sign(±ξ^a ± ξ^b ± ξ^c)` (mezcla de tres, el espurio clásico); en este
caso no lo es, es otro mínimo local de H.

### 2.2 QR 37×37 (`ej2_qr.py`, `configs/hopfield_qr.json` → `resultados/hopfield_qr/`)

**Representación.** Los PNG de `datos/hopfield` ya son 37×37 (QR versión 5, nivel L). Negro = +1,
blanco = −1 (`src/qr.py`; si la imagen fuera más grande se muestrea el centro de cada módulo). N = 1369
neuronas → W de 1369×1369. Para comprobar que "el lector de QR puede decodificarlos" se usa un lector
real (`cv2.QRCodeDetector`) sobre la salida de la red: los 5 originales decodifican a sus URLs.

**Resultado principal: con los 5 QR almacenados la red no funciona, y la teórica explica por qué.**

* Solapamiento `ξ^μ·ξ^ν/N` entre QR: 0.58–0.86 (`ortogonalidad.png`). El 69.8 % de los módulos (955 de
  1369) es idéntico en los 5: los patrones de posicionamiento, alineación, timing, el formato y, sobre
  todo, los ~70 caracteres iniciales de URL que comparten (`patrones.png`, último panel).
* Con p = 5, `0.15·N = 205` sobra… pero el otro requisito ("más o menos ortogonales") se viola
  groseramente. Aplicando la condición de estabilidad `sign(h_i^ν) = ξ_i^ν` a cada QR almacenado,
  tienen 71 a 225 bits inestables (`cinco_qr_estabilidad.csv`). Por eso **ningún QR es punto fijo**:
  presentado *sin ruido*, cada uno converge en 2 iteraciones al **mismo** estado espurio
  (`cinco_qr_falla.png`), una "media" de los cinco (overlaps 0.67–0.90 con cada uno) que el lector no
  decodifica. Su energía (−2173) es **más baja** que la de cualquier QR almacenado (−1790 … −2077):
  cuando los patrones están correlacionados el mínimo global de H es la mezcla, no los patrones.
* Subconjuntos (`subconjuntos.csv`): **los 10 pares funcionan (100 % estables, 100 % de recuperación con
  20 % de ruido) y ningún trío/cuarteto funciona (0 %)**. Tiene explicación exacta: con p = 2 el
  crosstalk sobre el patrón 1 es un único término `m₁₂·ξ_i²` con |m₁₂| < 1, que nunca puede superar al
  término propio (de módulo 1). Con p = 3 se suman dos términos de hasta 0.86 cada uno y sí lo superan.

**Caso de uso con un par** (`par = ["food", "wine"]`, elegido a propósito: es el par **más** correlacionado,
0.86; cualquier otro funciona igual o mejor):

* Ruido aleatorio (`ruido.csv`, `ruido.png`, 30 repeticiones × 2 QR por nivel): **100 % de recuperación
  exacta hasta 40 % de módulos invertidos** (≈ 550 bits), 85 % con 45 % y 23 % con 50 % (ahí la consulta
  es ruido puro y la red va al otro QR 37 %, al negado 30 %, o a un espurio 10 %). Mientras tanto, el
  lector de QR **no decodifica ninguna consulta con ≥ 5 % de ruido** y sí decodifica el 100 % de las
  salidas de Hopfield hasta 40 %. Es decir, la red aporta exactamente lo que el enunciado quería: la
  corrección de errores del QR (nivel L tolera ~7 %) se queda corta y la memoria asociativa la extiende
  hasta 40 %.
* Iteraciones: **2 siempre** (una para corregir, otra para verificar estabilidad) hasta 40 %; 2.1 en
  promedio con 50 %. Con N grande y p = 2, un paso sincrónico ya proyecta sobre el patrón más cercano.
* Energía (`energia.png`, `ejemplo_evolucion.png`): H(patrón) = −1189.5; con 25 % de ruido la consulta
  arranca en −277 y en un paso llega a −1189.5; con 50 % arranca en ≈ 0 (sin correlación con nada). H
  nunca creció en ningún paso.
* Deterioros "realistas" del enunciado (`deterioros.png`, `deterioros.csv`): esquina rota de 10×10 y de
  16×16, manchas negras de radio 6 y 10, rayones de 2 y 5 filas, un rasgado de 8 columnas con ruido y
  borrado del 30 % y 50 % de los módulos. **Los 18 casos se recuperan exactamente en 2 iteraciones y el
  lector decodifica la salida**, mientras que de las consultas solo decodifica la esquina 10×10 (53 bits,
  dentro de la tolerancia del QR).
* Capacidad (`capacidad.png`): con patrones **aleatorios** (≈ ortogonales) la cota 0.15 N ≈ 205 se
  observa: 100 % estables hasta p = 50, 80 % en p = 100, 35 % en p = 150, 0 % en p = 200 (el criterio
  "todos los bits exactos" es más estricto que el de la cota, por eso cae un poco antes). En cambio,
  al par de QR le bastan 25 patrones aleatorios adicionales para dejar de ser estable: los ~96 módulos
  en que food y wine difieren tienen un margen de solo 1 − 0.86 = 0.14, que cualquier crosstalk extra
  supera.

**Limitaciones del caso de uso**

1. **Los QR son patrones intrínsecamente correlacionados**: estructura fija + prefijo de URL común. La
   regla de Hebb no los puede almacenar de a más de dos. Para que la trattoria tenga sus 5 cartas haría
   falta cambiar el modelo (p. ej. una regla de aprendizaje que descorrelacione, o almacenar solo la zona
   de datos que difiere), lo cual excede lo dado en la teórica; con lo visto en clase la solución honesta
   es una red por par de menús o URLs diseñadas para ser distintas (dominio corto y path aleatorio).
2. **La red solo devuelve lo que ya conoce**: si la dueña cambia un menú, el QR nuevo no se recupera (y
   peor: un QR nuevo parecido es "corregido" hacia uno viejo). No reemplaza al lector, lo complementa.
3. **Escala**: W tiene N² = 1.87 M pesos para un QR de 37×37; un QR impreso real de versión 10 (57×57)
   ya necesita 10.5 M. Y la foto real debe ser previamente segmentada, enderezada y binarizada a la
   grilla de módulos, que es la parte difícil del problema y no la resuelve Hopfield.
4. **Ruido estructurado vs. aleatorio**: la red tolera 40 % de ruido uniforme pero una esquina rota se
   lleva un *finder pattern* entero; en nuestros casos lo recuperó, pero porque el par almacenado
   comparte esa esquina (el crosstalk juega a favor). Con más patrones o esquinas que difieran no está
   garantizado.
5. **Estados espurios y negados**: con mucho ruido la red converge con confianza a un QR equivocado, al
   negado (que el lector no lee) o a una mezcla. No hay forma interna de saber que se equivocó; por eso en
   la práctica hay que validar la salida con el lector/checksum del QR, como hace el script.
6. **Modo sincrónico puede ciclar** (ciclos de período 2 en 10 % de los casos con 50 % de ruido). El modo
   asincrónico (implementado, `update="async"`) evita ciclos a costa de más iteraciones.

---

## 3. Qué entregan los scripts

| Script | Genera |
|---|---|
| `ej1_eda.py` | `resultados/eda/` (describe, boxplots, correlaciones, derivadas, variables seleccionadas) |
| `ej1_pca.py` | `resultados/pca/` (autovalores, scree, cargas, biplot, índice PC1, proyección) |
| `ej1_kohonen.py` | `resultados/kohonen/` (barrido de 864 corridas, hits, Matriz U, planos, perfiles, SOM vs PCA) |
| `ej1_oja.py` | `resultados/oja/` (barrido de η, convergencia, pesos vs PC1, índice, ranking por tema) |
| `ej2_letras.py` | `resultados/hopfield_letras/` (ortogonalidad, capacidad, ruido, energía, espurio) |
| `ej2_qr.py` | `resultados/hopfield_qr/` (falla con 5, subconjuntos, ruido, deterioros, energía, capacidad, `salida.txt`) |
| `tests/test_teorica.py` | verificación contra los ejemplos de las diapositivas |
