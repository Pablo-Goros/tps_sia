# Comparación de arquitecturas y optimizadores

## Selección vigente — campaña de la presentación (2026-11-10)

La campaña posterior descrita en [presentation-results.md](presentation-results.md)
completó 43 corridas y 54300 épocas. BCE y Adam 0.002 constante, con arquitectura
deeper y latente tanh, cumplieron en 5/5 semillas nuevas frente a 0/5 de la
referencia reconstruida. Se seleccionó `confirmation-finalist-seed-49`: los
32 caracteres se reconstruyen sin errores a época 600. La procedencia está en
`results/final/selection.json`; la revisión visual de letras nuevas sigue pendiente.

Los modelos históricos broad/final no estaban disponibles en este checkout;
se reconstruyeron los controles con autorización del usuario en el entorno
actual. Las secciones siguientes conservan el registro de las campañas previas
y sus selecciones históricas; no describen la selección vigente.

Estado histórico al 2026-11-10: campaña factorial completa, 180/180 corridas
terminadas. Modelo seleccionado: `deeper-rmsprop-0p002-seed-44`.
La reconstrucción de los 32 patrones cumple la tolerancia; la revisión visual
de una nueva letra sigue pendiente.

Actualizaciones posteriores del mismo día: se ejecutaron la continuación
controlada, una comparación local de tasas y una comparación de rho de RMSProp,
descritas al final de este documento. Ninguna mejoró la proporción de éxitos
sobre la referencia y la selección vigente permanece intacta.

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

## Continuación controlada RMSProp — 2026-11-10

El usuario autorizó únicamente el [protocolo de continuación](configs/continuation-rmsprop.json):
partir por separado de los modelos de las semillas 42 y 43 a época 3000 y
comparar continuar con tasa 0.002 contra reducirla a 0.001. Se conservaron pesos,
los 16 tensores de momentos RMSProp, contador de actualizaciones, RNG e historia.
No cambió la arquitectura, el lote 32, rho=0.9, epsilon=1e-8, activaciones,
objetivo cuadrático ni umbral 0.5. No se usó un scheduler automático.

Presupuesto: cuatro continuaciones de hasta 1000 épocas adicionales, máximo
4000 épocas adicionales y 600 segundos compartidos, sin reintentos automáticos.
Evaluación cada 100 épocas y parada al cumplir la tolerancia. La hipótesis era
que reducir la tasa al final ayudaría a corregir los píxeles pendientes; el
control con tasa constante separa ese efecto del entrenamiento adicional.

El [preflight](results/continuation-rmsprop-2026-11-10/preflight.json) verificó los
hashes de modelos y dataset, métricas al recargar, momentos y contadores antes
de entrenar. Se usaron Python 3.13.7, NumPy 2.5.1 y Matplotlib 3.11.2, las mismas
versiones registradas en la campaña factorial. La ejecución consumió **4000
épocas adicionales en 88.34 segundos**, completando las cuatro ramas hasta
época acumulada 4000. Ninguna alcanzó tolerancia.

| Semilla | Tasa en continuación | Máximo error | Error medio por patrón | Patrones dentro de tolerancia | MSE por píxel |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.002 (control) | 2 | 0.40625 | 30/32 | 0.0116161963 |
| 42 | 0.001 | 2 | 0.40625 | 30/32 | 0.0116116074 |
| 43 | 0.002 (control) | 2 | 0.43750 | 31/32 | 0.0125045037 |
| 43 | 0.001 | 2 | 0.40625 | 31/32 | 0.0116111695 |

En ambas tasas, los patrones fuera de tolerancia siguen siendo `0x71` y `0x77`
para la semilla 42, y `0x7d` para la 43; cada uno tiene dos píxeles incorrectos.
La tasa reducida mejoró principalmente el error medio y MSE de la semilla 43,
pero no resolvió su patrón pendiente. La semilla 42 mejoró respecto de época
3000 también con tasa constante. Menor MSE no implicó cumplir el error máximo.

Sobre las dos semillas emparejadas, ambas alternativas tuvieron **0/2 éxitos** y
media del máximo error 2.0. Con el orden agregado histórico, 0.001 queda por
delante en criterios secundarios: error medio 0.40625 frente a 0.421875, y MSE
medio 0.0116113885 frente a 0.0120603500. Esto no demuestra mayor consistencia.
La semilla 44 exitosa no fue entrenada: sigue existiendo el modelo que cumple,
y la cobertura de las tres semillas originales continúa siendo 1/3. No son
cuatro inicializaciones independientes ni una validación con semillas nuevas.

Limitación del entorno: el prefijo `OPENBLAS_NUM_THREADS=1` usado desde WSL no
fue transferido al proceso Python de Windows. Un sondeo posterior con la misma
invocación devolvió la variable ausente; no quedó controlado el número de hilos
BLAS. Las cuatro ramas usaron la misma invocación. No se repitió entrenamiento
para corregir este detalle ni se comparan duraciones con la campaña original.

La [verificación de recarga](results/continuation-rmsprop-2026-11-10/reload-verification.json)
reprodujo exactamente las métricas guardadas de las cuatro ramas, verificando
4000 épocas/actualizaciones y sus historias de tasas. Las 14 pruebas de TP5
pasaron, incluida equivalencia exacta de continuación en memoria y tras recarga
con ambas tasas, conservación del origen, rechazo de sobrescritura y límite
de tiempo vencido sin actualizaciones.

Se comprobaron por SHA256 **1999 archivos originales sin cambios** en
`comparison-broad/` y `final/`; la evidencia está en
[preservation.json](results/continuation-rmsprop-2026-11-10/preservation.json).
El [resumen](results/continuation-rmsprop-2026-11-10/summary.json) y las cuatro
ramas conservan métricas, modelos, historias y procedencia. Los resultados
están ignorados por Git y deben transferirse por separado si se necesitan en
otro equipo. `final/selection.json` sigue señalando el modelo original de
semilla 44; no se ejecutó ninguna campaña posterior ni revisión de candidatos.

## Tasas fijas cercanas de RMSProp — 2026-11-10

Después de revisar la continuación, el usuario autorizó la siguiente propuesta:
probar tasas fijas **0.0015 y 0.0025 desde cero**, semillas 42/43/44, hasta 3000
épocas por corrida. El [protocolo](configs/comparison-rmsprop-local.json) conserva
arquitectura deeper, activaciones tanh/logística, código de dos dimensiones,
lote 32, rho=0.9, epsilon=1e-8, umbral 0.5 y ausencia de regularización,
augmentation y batch normalization. Se evaluó cada 100 épocas y se detuvo al
cumplir la tolerancia. La hipótesis es que una tasa cercana aplicada durante
toda la trayectoria podría mejorar la consistencia; no se modificó rho ni se
usó scheduler.

Presupuesto: **seis corridas nuevas y hasta 18000 épocas**. Las tres corridas
originales de 0.002 se reutilizaron a su presupuesto original, sin repetirlas
ni extenderlas. Por eso el resumen contiene nueve filas y solo seis nuevos
entrenamientos. No se usan como referencia las continuaciones hasta época 4000.

El [preflight](results/comparison-rmsprop-local-2026-11-10/preflight.json) comprobó
el protocolo y las tres referencias por hash, dataset y métricas al recargar.
Se usó el comparador existente sin modificar código de entrenamiento, con
Python 3.13.7, NumPy 2.5.1 y Matplotlib 3.11.2. Las variables de hilos BLAS/OMP
estuvieron sin fijar, igual que en los manifiestos originales efectivos;
el número de hilos no está controlado y no se comparan velocidades.

Se consumieron **17500 épocas nuevas**: cinco corridas agotaron las 3000 épocas
y una cumplió a las 2500. La suma de duraciones de las seis corridas nuevas fue
**411.64 segundos** (6 min 52 s), incluyendo sus artefactos. Esa suma no incluye
el tiempo histórico de las corridas reutilizadas ni toda la preparación y
verificación de la campaña.

| Tasa | Semilla | Origen | Épocas | Máximo error | Error medio por patrón | Patrones dentro de tolerancia | MSE por píxel |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.0015 | 42 | Nueva | 3000 | 2 | 0.43750 | 29/32 | 0.0125921779 |
| 0.0015 | 43 | Nueva | 3000 | 2 | 0.21875 | 31/32 | 0.0062633260 |
| 0.0015 | 44 | Nueva | 3000 | 2 | 0.40625 | 30/32 | 0.0117016161 |
| 0.002 | 42 | Reutilizada | 3000 | 2 | 0.46875 | 30/32 | 0.0134483064 |
| 0.002 | 43 | Reutilizada | 3000 | 2 | 0.46875 | 31/32 | 0.0134000393 |
| 0.002 | 44 | Reutilizada | 2000 | 1 | 0.43750 | 32/32 | 0.0130203957 |
| 0.0025 | 42 | Nueva | 3000 | 2 | 0.34375 | 31/32 | 0.0098478562 |
| 0.0025 | 43 | Nueva | 3000 | 2 | 0.62500 | 25/32 | 0.0178609037 |
| 0.0025 | 44 | Nueva | 2500 | 1 | 0.46875 | 32/32 | 0.0134879845 |

| Tasa | Éxitos | Media del máximo error | Media del error por patrón | Media del MSE por píxel |
| --- | ---: | ---: | ---: | ---: |
| 0.0015 | 0/3 | 2.0000 | 0.35416667 | 0.0101857067 |
| **0.002** | **1/3** | **1.6667** | **0.45833333** | **0.0132895805** |
| 0.0025 | 1/3 | 1.6667 | 0.47916667 | 0.0137322481 |

**No se observó mejora de consistencia.** La tasa 0.0025 produjo otro modelo
que cumple, pero en la misma semilla 44; no resolvió las semillas pendientes.
La tasa 0.0015 redujo el error medio y MSE agregados, pero dejó todos los modelos
con máximo error 2. En particular, su semilla 43 tiene 26 patrones exactos y
solo un patrón fuera de tolerancia; un MSE menor no reemplaza el requisito de
error máximo por patrón.

El criterio histórico conserva como ganadora **0.002**: empata con 0.0025 en
media del máximo error y gana en error medio; 0.0015 pierde en el primer
criterio. El [resumen](results/comparison-rmsprop-local-2026-11-10/summary.json)
selecciona la misma corrida original `deeper-rmsprop-0p002-seed-44`. No se
reemplazó `results/final/selection.json` ni se regeneraron candidatos.

La [verificación](results/comparison-rmsprop-local-2026-11-10/verification.json)
recargó los nueve modelos y reprodujo exactamente sus métricas, predicciones y
códigos persistidos. Comprobó que solo varía la tasa en la configuración efectiva,
las historias de tasas y actualizaciones, los motivos de parada y el presupuesto
nuevo. No se agregaron pruebas que duplicaran la configuración ni se repitieron
entrenamientos para verificarla. Se comprobaron por SHA256 **2041 archivos
anteriores intactos** en `comparison-broad/`, `final/` y la continuación previa;
los hashes están en [preservation.json](results/comparison-rmsprop-local-2026-11-10/preservation.json).

Las tres semillas ya participaron en la búsqueda original: estos resultados
exploran tasas sobre esas inicializaciones y no son validación independiente
con semillas nuevas. No demuestran superioridad universal ni imposibilidad de
mejorar. Los artefactos de esta campaña también están ignorados por Git y deben
compartirse por separado. No se ejecutaron otros experimentos.

## Memoria de RMSProp: rho — 2026-11-10

El usuario autorizó comparar **rho=0.85 y 0.95**, con tasa fija **0.002**,
semillas 42/43/44 desde cero y máximo 3000 épocas por corrida. El
[protocolo](configs/comparison-rmsprop-rho.json) mantuvo arquitectura deeper,
activaciones tanh/logística, código bidimensional, lote 32, epsilon=1e-8,
umbral 0.5 y ausencia de L2, augmentation y batch normalization. Evaluación
cada 100 épocas, parada al cumplir todos los patrones y sin scheduler.

La hipótesis es que cambiar la memoria del promedio móvil de gradientes
cuadráticos ayude a resolver los patrones pendientes. La implementación aplica
`s_t = rho*s_(t-1) + (1-rho)*g_t**2`, sin corrección de sesgo, y actualiza con
`learning_rate*g_t/(sqrt(s_t)+epsilon)`. Un rho menor da más peso al gradiente
reciente y uno mayor conserva más memoria. Esto describe el mecanismo; no
garantiza mejorar la reconstrucción. Referencias: `tp3/shared/optimizers.py`,
RMSProp (desde `tps_sia/`); OFF-013 pp. 8–10; PA-004 p. 1, apartado 1.a.2.

Presupuesto: **seis corridas nuevas y hasta 18000 épocas**. Se reutilizaron las
tres referencias originales de rho=0.9, sin repetirlas ni continuarlas. Las
nueve filas del resumen corresponden a seis entrenamientos nuevos y tres
referencias a su presupuesto original; no se mezclan con la continuación hasta
época 4000 ni con las tasas vecinas del experimento anterior.

El [preflight](results/comparison-rmsprop-rho-2026-11-10/preflight.json) validó el
protocolo, los valores efectivos de rho y tasa, el presupuesto y las referencias
por hash, dataset y métricas al recargar. Se usó el comparador existente sin
modificar código de entrenamiento: Python 3.13.7, NumPy 2.5.1 y Matplotlib
3.11.2. Las variables BLAS/OMP siguieron sin fijar, como constan efectivamente
en los manifiestos originales; no se comparan velocidades entre campañas.

Todas las corridas nuevas agotaron 3000 épocas: se consumieron **18000 épocas**.
La suma de sus duraciones fue **412.22 segundos** (6 min 52 s), incluyendo los
artefactos de cada corrida, sin contar preparación, verificación ni tiempo
histórico de referencias reutilizadas.

| Rho | Semilla | Origen | Épocas | Máximo error | Error medio por patrón | Patrones dentro de tolerancia | MSE por píxel |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.85 | 42 | Nueva | 3000 | 2 | 0.31250 | 31/32 | 0.0090170509 |
| 0.85 | 43 | Nueva | 3000 | 2 | 0.31250 | 31/32 | 0.0089552985 |
| 0.85 | 44 | Nueva | 3000 | 2 | 0.34375 | 30/32 | 0.0098459798 |
| 0.9 | 42 | Reutilizada | 3000 | 2 | 0.46875 | 30/32 | 0.0134483064 |
| 0.9 | 43 | Reutilizada | 3000 | 2 | 0.46875 | 31/32 | 0.0134000393 |
| 0.9 | 44 | Reutilizada | 2000 | 1 | 0.43750 | 32/32 | 0.0130203957 |
| 0.95 | 42 | Nueva | 3000 | 2 | 0.40625 | 30/32 | 0.0116427681 |
| 0.95 | 43 | Nueva | 3000 | 2 | 0.50000 | 27/32 | 0.0145434995 |
| 0.95 | 44 | Nueva | 3000 | 2 | 0.56250 | 30/32 | 0.0161165735 |

| Rho | Éxitos | Media del máximo error | Media del error por patrón | Media del MSE por píxel |
| --- | ---: | ---: | ---: | ---: |
| 0.85 | 0/3 | 2.0000 | 0.32291667 | 0.0092727764 |
| **0.9** | **1/3** | **1.6667** | **0.45833333** | **0.0132895805** |
| 0.95 | 0/3 | 2.0000 | 0.48958333 | 0.0141009470 |

**No mejoró la consistencia.** Rho=0.85 redujo el error medio y MSE agregados,
y dejó 23 patrones exactos en cada semilla, pero mantuvo uno, uno y dos
patrones fuera de tolerancia respectivamente. Rho=0.95 tampoco produjo modelos
que cumplieran. No se alcanzó el objetivo exploratorio de superar 1/3 éxitos.

El criterio histórico conserva como ganadora **rho=0.9**, por menor media del
máximo error; las mejoras secundarias de rho=0.85 no cambian esa selección.
El [resumen](results/comparison-rmsprop-rho-2026-11-10/summary.json) selecciona
la corrida original `deeper-rmsprop-0p002-seed-44`. `final/selection.json`, el
modelo final y la revisión pendiente de candidatos permanecen intactos.

La [verificación](results/comparison-rmsprop-rho-2026-11-10/verification.json)
recargó los nueve modelos y reprodujo exactamente métricas, predicciones y
códigos. Comprobó historias, actualizaciones, épocas, motivos de parada y
presupuesto; solo varía rho en la configuración efectiva. Los
[hashes de preservación](results/comparison-rmsprop-rho-2026-11-10/preservation.json)
verificaron **2123 archivos anteriores intactos** en `comparison-broad/`,
`final/`, la continuación y la comparación local de tasas.

Estas tres semillas ya se usaron para explorar hiperparámetros: el resultado
no estima robustez con inicializaciones nuevas ni demuestra que otros valores
de rho no puedan mejorar. Los artefactos nuevos están ignorados por Git y deben
transferirse por separado. No se ejecutaron experimentos posteriores.
