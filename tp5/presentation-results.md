# TP5 — Resultados de la campaña basada en la presentación

Fecha: 2026-11-10. Campaña completa. Mejora confirmada con semillas nuevas; selección final actualizada.

## Condiciones y referencia

Se ejecutaron **43 corridas nuevas y 54300 épocas**, frente al máximo autorizado de 43 y 321000. Tiempo de campaña: 10.02 minutos, incluyendo generación de artefactos y verificaciones; no es una comparación de velocidad entre optimizadores.

Entorno: Python 3.14.4, NumPy 2.4.4, Matplotlib 3.11.1. BLAS/OMP sin fijar. Las tres referencias fueron reconstruidas con autorización explícita: los modelos broad/final históricos y Python 3.13 no estaban disponibles. Los resultados nuevos se comparan entre sí en este entorno y no se hacen pasar por las corridas históricas.

El hash del archivo font.h difiere por CRLF frente a LF; al normalizar finales de línea coincide con el contenido conocido. No se modificó el dataset. El piloto anterior reproduce exactamente sus métricas. Se usan 32 caracteres, lote 32, evaluación cada 100 épocas, umbral 0.5 y parada cuando cada carácter tiene como máximo un píxel incorrecto. No se continúa hasta reconstrucción exacta después de alcanzar tolerancia.

La salida logística local es sigmoid(2·beta·h), beta=1; BCE incluye ese factor en el gradiente. Se conserva la reducción suma sobre píxeles/media sobre muestras. La pérdida optimizada está en history.costo; metrics.training_loss conserva el medio error cuadrático por compatibilidad. No se comparan valores de objetivos distintos para seleccionar.

La agenda cosenoidal usa lr(t)=lr0·(1+cos(pi·t/T))/2, t=0,…,T−1. Cero es el límite en T; no se agrega una actualización con tasa cero. Se verifica al recargar cada historia de tasas.

## Criterio y decisiones

Se prioriza cantidad de semillas que cumplen, luego media del máximo error, error binario medio y MSE por píxel. Empates completos conservan el candidato anterior. Las semillas 42–44 se usan para explorar; el finalista se congela antes de evaluar 45–49. Se informa también el criterio histórico, sin cambiarlo silenciosamente.

### Pérdida × activación latente

| Configuración | Éxitos | Máximo error medio | Error medio | MSE/píxel | Épocas medias |
| --- | ---: | ---: | ---: | ---: | ---: |
| half_squared_error / tanh / rmsprop 0.002 / constante / actual / 3000 épocas | 1/3 | 1.6667 | 0.45833 | 0.01315696 | 2733.3 |
| **bce / tanh / rmsprop 0.002 / constante / actual / 3000 épocas** | 3/3 | 1.0000 | 0.05208 | 0.00373861 | 1033.3 |
| half_squared_error / linear / rmsprop 0.002 / constante / actual / 3000 épocas | 0/3 | 2.0000 | 0.32292 | 0.00923678 | 3000.0 |
| bce / linear / rmsprop 0.002 / constante / actual / 3000 épocas | 3/3 | 1.0000 | 0.07292 | 0.00521257 | 833.3 |

Decisión: `bce-tanh`. Ganador según ranking histórico: `bce-tanh`.
[Protocolo y evidencia por corrida](results/presentation-campaign-2026-11-10/01-loss-latent/summary.json).

### Optimizador

| Configuración | Éxitos | Máximo error medio | Error medio | MSE/píxel | Épocas medias |
| --- | ---: | ---: | ---: | ---: | ---: |
| bce / tanh / rmsprop 0.002 / constante / actual / 3000 épocas | 3/3 | 1.0000 | 0.05208 | 0.00373861 | 1033.3 |
| **bce / tanh / adam 0.002 / constante / actual / 3000 épocas** | 3/3 | 0.3333 | 0.01042 | 0.00306504 | 666.7 |

Decisión: `bce-tanh-adam`. Ganador según ranking histórico: `bce-tanh-adam`.
[Protocolo y evidencia por corrida](results/presentation-campaign-2026-11-10/02-optimizer/summary.json).

### Tasa × scheduler

| Configuración | Éxitos | Máximo error medio | Error medio | MSE/píxel | Épocas medias |
| --- | ---: | ---: | ---: | ---: | ---: |
| **bce / tanh / adam 0.002 / constante / actual / 3000 épocas** | 3/3 | 0.3333 | 0.01042 | 0.00306504 | 666.7 |
| bce / tanh / adam 0.002 / coseno / actual / 3000 épocas | 3/3 | 1.0000 | 0.03125 | 0.00375300 | 666.7 |
| bce / tanh / adam 0.003 / constante / actual / 3000 épocas | 3/3 | 0.3333 | 0.01042 | 0.00326095 | 500.0 |
| bce / tanh / adam 0.003 / coseno / actual / 3000 épocas | 3/3 | 0.6667 | 0.02083 | 0.00360360 | 500.0 |

Decisión: `bce-tanh-adam`. Ganador según ranking histórico: `bce-tanh-adam`.
[Protocolo y evidencia por corrida](results/presentation-campaign-2026-11-10/03-rate-schedule/summary.json).

### Arquitectura

| Configuración | Éxitos | Máximo error medio | Error medio | MSE/píxel | Épocas medias |
| --- | ---: | ---: | ---: | ---: | ---: |
| **bce / tanh / adam 0.002 / constante / actual / 3000 épocas** | 3/3 | 0.3333 | 0.01042 | 0.00306504 | 666.7 |
| bce / tanh / adam 0.002 / constante / angosta / 3000 épocas | 3/3 | 1.0000 | 0.10417 | 0.00725567 | 800.0 |

Decisión: `bce-tanh-adam`. Ganador según ranking histórico: `bce-tanh-adam`.
[Protocolo y evidencia por corrida](results/presentation-campaign-2026-11-10/04-architecture/summary.json).

### Presupuesto de entrenamiento

| Configuración | Éxitos | Máximo error medio | Error medio | MSE/píxel | Épocas medias |
| --- | ---: | ---: | ---: | ---: | ---: |
| **bce / tanh / adam 0.002 / constante / actual / 3000 épocas** | 3/3 | 0.3333 | 0.01042 | 0.00306504 | 666.7 |
| bce / tanh / adam 0.002 / constante / actual / 15000 épocas | 3/3 | 0.3333 | 0.01042 | 0.00306504 | 666.7 |
| bce / tanh / adam 0.002 / coseno / actual / 15000 épocas | 3/3 | 0.6667 | 0.03125 | 0.00430477 | 633.3 |

Decisión: `bce-tanh-adam`. Ganador según ranking histórico: `bce-tanh-adam`.
[Protocolo y evidencia por corrida](results/presentation-campaign-2026-11-10/05-training-budget/summary.json).

### Confirmación con semillas nuevas

| Configuración | Éxitos | Máximo error medio | Error medio | MSE/píxel | Épocas medias |
| --- | ---: | ---: | ---: | ---: | ---: |
| half_squared_error / tanh / rmsprop 0.002 / constante / actual / 3000 épocas | 0/5 | 3.2000 | 0.61250 | 0.01641758 | 3000.0 |
| **bce / tanh / adam 0.002 / constante / actual / 3000 épocas** | 5/5 | 0.8000 | 0.03125 | 0.00463866 | 640.0 |

Decisión: `confirmation-finalist`. Ganador según ranking histórico: `confirmation-finalist`.
[Protocolo y evidencia por corrida](results/presentation-campaign-2026-11-10/06-confirmation/summary.json).

## Confirmación por semilla

| Semilla | Referencia: máximo error | Finalista: máximo error | Épocas referencia | Épocas finalista |
| --- | ---: | ---: | ---: | ---: |
| 45 | 2 | 1 | 3000 | 600 |
| 46 | 2 | 1 | 3000 | 600 |
| 47 | 2 | 1 | 3000 | 600 |
| 48 | 3 | 1 | 3000 | 800 |
| 49 | 7 | 0 | 3000 | 600 |

## Artefactos y límites

[Tabla de las 43 corridas](results/presentation-campaign-2026-11-10/all-new-runs.csv), [gráfico](results/presentation-campaign-2026-11-10/overview.png), [decisiones](results/presentation-campaign-2026-11-10/decisions.json), [presupuesto](results/presentation-campaign-2026-11-10/budget.json) y [finalista congelado](results/presentation-campaign-2026-11-10/frozen-finalist.json).

Se verificaron sin cambios 61 archivos anteriores por SHA256. Cada etapa recarga todos sus modelos y reproduce métricas, predicciones, códigos y tasas. Pruebas: 18 de TP5, 49 compartidas de TP3 y 11 de scheduler/regularización/BatchNorm. Las comprobaciones de gradientes incluyen ambas pérdidas y activaciones, beta distinto de uno y salidas saturadas. La continuación cosenoidal coincide exactamente tras guardar y recargar.

La búsqueda posterior a pérdida/latente es secuencial: no explora todas las interacciones entre arquitectura, optimizador y tasas. Cinco semillas nuevas aportan confirmación limitada sobre inicialización en estos mismos 32 patrones, no generalización a datos nuevos. No se replicó el protocolo 30/30 del otro grupo ni se adoptó su exigencia de cero errores.

Modelo publicado: `confirmation-finalist-seed-49`. Máximo error 0, error medio 0.0, 32/32 patrones exactos y 32/32 dentro de tolerancia. SHA256: `e39ab072ed146e11a337d82e5f625a775d8cc73628d90355746a8ece653fa85e`.

[Selección y procedencia](results/final/selection.json), [reconstrucciones](results/final/evaluation/reconstructions.png), [latente](results/final/latent/latent.png) y [candidatos](results/final/exploration/candidates.json). La revisión visual de una nueva letra queda pendiente.
