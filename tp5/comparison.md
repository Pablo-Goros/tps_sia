# Comparación de arquitecturas y optimizadores

Estado actualizado al 2026-11-10: campaña factorial completa, 180/180 corridas
terminadas. Modelo seleccionado: `deeper-rmsprop-0p002-seed-44`.
La reconstrucción de los 32 patrones cumple la tolerancia; la revisión visual
de una nueva letra sigue pendiente.

## Protocolo de la campaña factorial

El [protocolo ejecutado](results/comparison-broad/protocol.json), idéntico a
[la configuración](configs/comparison-broad.json), registra la autorización
del 2026-11-08. Se evaluaron 60 variantes con semillas compartidas 42, 43 y 44:
5 arquitecturas × 4 optimizadores × 3 tasas × 3 semillas = **180 corridas**.

| Nombre | Arquitectura |
| --- | --- |
| base | 35→20→2→20→35 |
| wide | 35→40→2→40→35 |
| wider | 35→60→2→60→35 |
| deep | 35→30→15→2→15→30→35 |
| deeper | 35→40→20→10→2→10→20→40→35 |

| Optimizador | Tasas de aprendizaje ensayadas | Otros parámetros efectivos |
| --- | --- | --- |
| Adam | 0.0005, 0.001, 0.002 | beta1=0.9, beta2=0.999, epsilon=1e-8 |
| RMSProp | 0.0005, 0.001, 0.002 | rho=0.9, epsilon=1e-8 |
| Momentum | 0.02, 0.05, 0.1 | momentum=0.9 |
| SGD | 0.01, 0.03, 0.05 | Sin momentum |

Todas las corridas usan los 32 patrones originales, lote 32, máximo 3000 épocas,
evaluación cada 100 y parada cuando cada patrón tiene como máximo un píxel
incorrecto. Activaciones internas tanh, código de dos dimensiones y salida
logística; umbral binario 0.5. No hay regularización, augmentation ni batch
normalization. Los parámetros efectivos constan en cada `manifest.json`.

Las 180 corridas terminaron: 178 agotaron 3000 épocas y dos alcanzaron tolerancia
antes. Se consumieron **538300 épocas**, frente al máximo de 540000 de esta
campaña. El piloto y las etapas anteriores no integran estas 180 corridas.

## Criterio y resultados

Se ordenan las variantes por la media entre semillas del máximo error de píxeles,
luego por error medio y finalmente MSE por píxel; un empate se resuelve por nombre.
Después se elige una corrida de la variante ganadora con el mismo orden individual
y desempate por identificador. No se elige primero la mejor semilla de toda la
búsqueda. Todas las variantes deben estar completas para efectuar la selección.

| Arquitectura | Mejor variante | Media del máximo error | Media del error por patrón | Éxitos de esa variante |
| --- | --- | ---: | ---: | ---: |
| base | Adam, 0.002 | 4.0000 | 1.18750 | 0/3 |
| wide | Adam, 0.002 | 4.3333 | 1.29167 | 0/3 |
| wider | Adam, 0.002 | 4.6667 | 1.40625 | 0/3 |
| deep | RMSProp, 0.002 | 2.0000 | 0.54167 | 0/3 |
| **deeper** | **RMSProp, 0.002** | **1.6667** | **0.45833** | **1/3** |

La tabla muestra la mejor combinación de optimizador y tasa por arquitectura,
no la media de sus 36 corridas. Las dos medias se calculan sobre tres semillas.

| Semilla de deeper / RMSProp 0.002 | Épocas | Máximo error | Error medio por patrón | Patrones dentro de tolerancia | MSE por píxel |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 3000 | 2 | 0.46875 | 30/32 | 0.01344831 |
| 43 | 3000 | 2 | 0.46875 | 31/32 | 0.01340004 |
| **44** | **2000** | **1** | **0.43750** | **32/32** | **0.01302040** |

Solo dos de las 180 corridas cumplieron la tolerancia:
`deeper-rmsprop-0p002-seed-44` y `deeper-rmsprop-0p0005-seed-43`.
La segunda llegó a 2300 épocas, con error medio 0.375 y MSE 0.01513840.
Aunque tiene menor error binario medio individual, su variante pierde por la media
del máximo error entre semillas (2.0 frente a 1.6667). La selección respeta así
el criterio registrado, sin ocultar esa alternativa.

La evidencia favorece la arquitectura deeper dentro de este presupuesto y estas
configuraciones. La variante ganadora tiene éxito en 1/3 semillas: no demuestra
convergencia consistente, superioridad universal ni generalización a otros datos.
No fue necesario adoptar un subconjunto del dataset.

## Modelo y evidencia final

El [resumen de campaña](results/comparison-broad/summary.json) identifica la variante
y corrida ganadoras. [selection.json](results/final/selection.json) registra su
procedencia y hash; [model.npz](results/final/model.npz) es una copia exacta del
modelo de esa corrida, junto con su [configuración](results/final/config.json).
Las rutas de los metadatos finales son relativas a `tp5/` para facilitar compartirlos.

SHA256 del modelo: `ad4e31ecaed339aa9a35404769284f55eff8a9fbff6ddad72c55aa1e9de73fca`.
La [evaluación](results/final/evaluation/metrics.json), el
[espacio latente](results/final/latent/latent.png) y los
[candidatos](results/final/exploration/candidates.json) se regeneraron recargando
ese mismo modelo, sin entrenamiento nuevo. La revisión de candidatos permanece
`pending`: novedad binaria no equivale a una letra validada por el estudiante.

## Etapas anteriores e interrupción conservada

La primera campaña comparó base, wide y deep con Adam 0.001 y semillas 42/43.
La [etapa de arquitecturas](results/comparison-architectures/summary.json) terminó
sus seis corridas y eligió deep. La
[etapa de optimizadores](results/comparison-optimizers/summary.json) comparó Adam
con Momentum 0.05 sobre deep, reutilizando las dos corridas Adam. Eligió
`momentum-seed-43`, con máximo error 3; no cumplió la tolerancia.

Hubo ocho corridas únicas completas en esas etapas. El intento interrumpido de
deep/42 permanece en `results/comparison-architectures/interrupted/`; el reintento
completo se realizó y no constituye una reanudación desde checkpoint. Su cargo
conservador de 3000 épocas eleva el máximo acumulado de esa campaña de 24000 a
27000 épocas. El piloto de 10000 épocas fue independiente.

`results/overnight-status.json` describe esa campaña previa y conserva su selección
histórica. La selección vigente para la entrega es `results/final/selection.json`,
basada en `results/comparison-broad/summary.json`.

## Datos disponibles para futuros gráficos

No se necesitan nuevos entrenamientos para graficar las corridas existentes:

- [runs.csv](results/comparison-broad/runs.csv): las 180 corridas, métricas finales,
  semilla, variante, épocas, duración y motivo de parada.
- [aggregates.csv](results/comparison-broad/aggregates.csv): medias, desviaciones
  poblacionales y tasa de éxito de las 60 variantes, con tres semillas cada una.
- `summary.json`: datos individuales y agregados, errores por patrón y selección.
- Cada directorio de corrida conserva `config.json`, `manifest.json`,
  `history.json`, `predictions.npz` y `model.npz`. Las evaluaciones de reconstrucción
  se guardan cada 100 épocas; la historia también contiene costos de entrenamiento.

Esto permite mapas de calor por arquitectura/optimizador/tasa, distribuciones entre
semillas, proporciones de éxito, curvas de aprendizaje y errores por carácter.
Ya existe [comparison.png](results/comparison-broad/comparison.png), con las 180
corridas; nuevas visualizaciones más legibles quedan para un trabajo posterior.
Las duraciones dependen del equipo y de interrupciones o pausas: no deben usarse
solas para afirmar que un optimizador es más rápido.

## Reproducción

Desde `tp5/`, instalar `requirements.txt`. Los comandos de evaluación y exploración
están en [results/report.md](results/report.md); no entrenan. Para reproducir toda
la campaña, usar un directorio nuevo y asumir el costo de hasta 540000 épocas:

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
python -m src.compare --protocol configs/comparison-broad.json --output results/comparison-broad-reproduction
```

La salida debe estar vacía. No sobrescribir la campaña archivada ni ejecutar
`src.overnight` para seleccionar esta campaña factorial: ese comando corresponde
a las dos etapas anteriores.
