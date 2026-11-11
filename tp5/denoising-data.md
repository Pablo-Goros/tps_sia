# Datos y reproducción del ejercicio 1.b

Este documento describe las interfaces y archivos para reutilizar la evidencia.
Por pedido del usuario, el flujo no genera gráficos ni informe. El fundamento
académico es el par `(entrada ruidosa, objetivo limpio)` de OFF-022 p. 22,
recogido en `wiki/topics/autoencoders.md` del repositorio del curso. Los niveles,
semillas, tamaños y presupuesto son decisiones del plan, no requisitos textuales
de PA-004 1.b. Se conservan los 32 patrones originales.

## Comandos

Desde `tp5/`, con Python 3.10+ y las dependencias de `requirements.txt`:

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.denoise prepare --config configs/denoising.json --output results/denoising-new
python -m src.denoise train --campaign results/denoising-new
python -m src.denoise evaluate --campaign results/denoising-new
```

`prepare` crea referencia y colecciones. Para una campaña creada con B1 anterior,
`python -m src.denoise collections --campaign results/denoising` añade las
colecciones; si ya existen, verifica sus hashes. `train` inicia una sola vez las
seis corridas desde cero: dos variantes × semillas 42/43/44, 3000 épocas cada una,
lote 32. Rechaza campañas iniciadas, sin reintentos ni reanudación automática.
Una interrupción deja estado y épocas en los manifiestos, sin selección final.
Repetir entrenamientos consume otro presupuesto; evaluar no entrena.

Para recalcular toda la evaluación desde los checkpoints guardados sin pisar nada:

```bash
python -m src.denoise evaluate --campaign results/denoising --output results/denoising/evaluation-reloaded
```

Cada salida de evaluación debe ser nueva. Para trasladar la campaña, copiar todo
su directorio: las rutas operativas son relativas a él. Las rutas absolutas en
manifiestos individuales conservan procedencia histórica. `results/` está ignorado
por Git; guardar una copia completa si se mueve el proyecto o se comparte evidencia.

## Arquitecturas y protocolo

`manifest.json` conserva la selección de 1.a, sus hashes, el protocolo y las dos
variantes resueltas. `winner` usa las capas de la selección y `latent-10` cambia
solamente la capa central a 10 unidades. Esta comparación examina si relajar la
compresión facilita recuperar perturbaciones. Es una hipótesis experimental.
Se heredan optimizador, tasa y pérdida; capas internas tanh y salida logística.
Los DAE usan pesos nuevos. El AE congelado de 1.a solo se evalúa.

El ruido invierte cada bit independientemente. En cada época se sortea un nivel
uniformemente entre `[0,.05,.10,.20]` por carácter. La secuencia de ruido depende
de `SeedSequence([semilla_corrida, época_absoluta, 1])`, con épocas desde 1,
y un `default_rng` independiente del MLP. La probabilidad no impone un número
exacto de cambios. No se deduplican coincidencias entre realizaciones.

Validación y evaluación usan semillas 20261110 y 20261111, con 100 realizaciones
por carácter y nivel positivo, y una sola en nivel cero. Son 12832 ejemplos por
colección. La selección promedia **por igual** los errores binarios medios de
`.05,.10,.20`; desempata por MSE medio y luego época más temprana. Cero es
diagnóstico. `.30` se excluye de la historia de validación y queda reservado
para evaluación final fuera del rango entrenado. No se usa la colección final
para seleccionar pesos, arquitectura o semilla.

Se evalúa al inicio, cada 100 épocas y al final; se cumplen las 3000 épocas sin
parar por tolerancia. Se compara cada variante por la media de los mejores
checkpoints de sus tres semillas; la corrida representativa usa los mismos
criterios y luego semilla menor. Un empate entre variantes conserva el orden
del manifiesto. La variación entre semillas se presenta por separado de la
variación entre ejemplos. No se interpreta como generalización a letras nuevas.

## Archivos y campos

- `reference/`: selección, modelo, configuración, manifiesto de origen y `font.h`
  congelados. `manifest.json` de campaña conserva sus SHA256.
- `collections/manifest.json`: semillas, orden, versión NumPy, hashes y conteos.
- `collections/{validation,final}.npz`: `inputs` y `targets` de forma `(12832,35)`,
  `levels`, `character_indices` y `realization_indices` de forma `(12832,)`.
  El orden es nivel, carácter, realización. `character_indices` va de 0 a 31;
  el código del carácter es índice + 96. Cada vector se dibuja como `(7,5)` en
  orden de filas, de arriba abajo. `targets` contiene el original limpio.
- `training.json`: estado global, versiones de ejecución, consumo de épocas,
  manifiestos de corridas y hashes del protocolo/validación. No hay selección
  definitiva si alguna corrida quedó incompleta.
- `runs/<variante>-seed-<semilla>/config.json`: arquitectura, optimizador, pérdida,
  semilla, presupuesto, niveles y hashes usados en esa corrida.
- `runs/.../last_model.npz`: estado al finalizar o interrumpirse, incluidos pesos,
  estado del optimizador, RNG y épocas acumuladas del MLP.
- `runs/.../best_model.npz`: checkpoint elegido por validación, posiblemente
  anterior a la última época. Su hash y época están en `manifest.json` de la corrida.
- `runs/.../history.json`: `training` conserva la historia completa del MLP;
  `validation` contiene `epoch`, `score` y métricas `levels` en cada evaluación.
  `training.costo` es la pérdida optimizada: BCE suma sobre píxeles y promedia
  muestras, o `0.5*mean(sum(error²,axis=1))` para el error cuadrático.
  `training.mse` es MSE por píxel. Las entradas ruidosas cambian cada época;
  la pérdida de entrenamiento no reemplaza la validación fija.
- `selection.json`: ganadora, representante, medias por variante, todas las
  corridas y rutas/hashes relativos de los mejores checkpoints.
- `validation-summary.csv`: una fila por corrida con época elegida, error medio
  y MSE promediados entre los tres niveles de selección.

## Evaluación y preparación de gráficos posteriores

`evaluation/metrics.json` registra la selección, hashes de modelos y colección,
umbral, niveles, política de ejemplos y agregados. Incluye el AE original,
los seis DAE y la entrada ruidosa sin procesar. Los archivos siguientes se
alinean fila a fila con `collections/final.npz`:

- `<modelo>.npz`: `predictions` continuas `(N,35)`, `binary` binaria `(N,35)`,
  `pixel_errors`, `input_pixel_errors`, `error_reduction` y `mse_per_example`
  `(N,)`; para modelos entrenados, también `codes` `(N,ancho_latente)`.
- `by-level.csv`: una fila por modelo y nivel; `character_index` vacío significa
  todos los caracteres. `count` permite distinguir las 32 muestras limpias de
  las 3200 por nivel positivo.
- `by-character.csv`: una fila por modelo, nivel y carácter.
- `seed-aggregates.csv`: media, desvío poblacional, mínimo y máximo entre las
  tres semillas por variante y nivel. Cada semilla pesa lo mismo.

`mean_pixel_errors` cuenta diferencias binarias respecto del limpio por imagen,
con umbral `>=0.5`. `input_mean_pixel_errors` cuenta el ruido original;
`mean_error_reduction` es entrada menos salida (positivo significa mejora).
`exact_fraction`, `within_tolerance_fraction` y `max_pixel_errors` describen
recuperación exacta, con ≤1 error y peor caso. `mse_per_pixel` compara la salida
continua con el limpio. `training_loss` en las tablas conserva por compatibilidad
medio error cuadrático (17.5 × MSE), incluso si el modelo optimizó BCE.
`improved_fraction`, `worsened_fraction` y `unchanged_fraction` comparan conteos
por ejemplo; no comparan MSE. No se debe promediar todos los niveles mezclando
sus diferentes tamaños: seleccionar explícitamente niveles y promediar sus medias.

Para curvas de entrenamiento, usar `history.training.epochs` frente a `costo`
o `mse`; para validación, `history.validation[*].epoch` frente a `score` o las
métricas de cada nivel. Para recuperación por ruido, usar `by-level.csv` y
`seed-aggregates.csv`. Para láminas, combinar `final.targets`, `final.inputs`,
`ae-reference.npz` y el NPZ de la corrida identificada por `selected_run`.

`metrics.json.example_indices` congela los índices de la primera realización
de los caracteres `[0,1,8,16,24,31]` en todos los niveles, incluidos cero y .30;
la política es previa a inspeccionar resultados. Conservar esos índices para
mostrar original / ruido / AE / DAE sin escoger solo casos favorables.
No hace falta reentrenar para producir ninguna de estas figuras.

## Verificaciones guardadas

La campaña ejecutada conserva `verification.json` con el presupuesto consumido,
checkpoints y hashes comprobados, recálculo de validación, comparación de las dos
evaluaciones desde persistencia, cantidad de arrays idénticos y conservación de
archivos originales. `evaluation-reloaded/` permite cotejar la repetición numérica
sin volver a entrenar. Las pruebas del código se ejecutan desde `tp5/`:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
