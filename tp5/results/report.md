# TP5, apartado 1.a — resultados y reproducción

Actualizado el 2026-11-10. La campaña factorial terminó sus **180 corridas** y
el modelo seleccionado reconstruye los **32 patrones con como máximo un píxel
incorrecto por patrón**. Se generaron el gráfico latente y la exploración del
mismo modelo. Sigue pendiente que el estudiante identifique y valide visualmente
una nueva letra: el apartado 1.a todavía no se presenta como completado.

## Consigna y fundamento

La [consigna PA-004, p. 1, apartado 1.a](../../../sources/extracted/PA-004.md)
pide comprimir los 32 patrones de 5×7 en dos dimensiones, estudiar arquitecturas
y optimización, reconstruir con un máximo de un píxel incorrecto por patrón,
graficar los códigos y generar una letra nueva. El
[original](../../../sources/practical-assignment/SIATP5.pdf) permanece intacto.
Este informe cubre el autoencoder básico; denoising y VAE quedan fuera de su alcance.

El encoder obtiene un código y el decoder reconstruye la entrada; cada patrón
se utiliza como su propio objetivo de entrenamiento. Este fundamento procede de
[OFF-022, pp. 3–6](../../../sources/extracted/OFF-022.md) y de la
[wiki de autoencoders](../../../wiki/topics/autoencoders.md#funcionamiento).
Explorar coordenadas puede generar imágenes, pero un autoencoder básico no
garantiza muestras interpretables en cualquier región del espacio latente
[OFF-022, pp. 28–38](../../../sources/extracted/OFF-022.md).

## Evidencia del apartado 1.a

| Punto | Evidencia y estado |
| --- | --- |
| 1. Encoder y decoder con código de dos dimensiones | Modelo final `35→40→20→10→2→10→20→40→35`, [configuración](final/config.json) y [modelo](final/model.npz) |
| 2. Estudio comparativo y tolerancia individual | 180/180 corridas terminadas; [protocolo y comparación](../comparison.md), [resumen](comparison-broad/summary.json) y [métricas finales](final/evaluation/metrics.json); cumplen los 32 patrones |
| 3. Gráfico de los códigos en dos dimensiones | [Figura](final/latent/latent.png) y [32 coordenadas etiquetadas](final/latent/latent.json) del modelo final |
| 4. Generación de una nueva letra | [Grilla](final/exploration/grid.png), [interpolaciones](final/exploration/interpolations.png) y [candidatos](final/exploration/candidates.json) regenerados; revisión visual `pending` |

Un candidato distinto de todos los originales tiene novedad binaria. Solo una
revisión visual explícita del estudiante permite presentarlo como una nueva
letra, conforme al [glosario](../../GLOSSARY.md).

## Dataset y medición

Se utiliza `Font3[32][7]` del [font.h original](../data/font.h), códigos
`0x60`–`0x7f`, incluido DEL, sin reducir el conjunto. Los cinco bits de cada fila
se leen del más significativo al menos significativo y se conservan las siete
filas: imágenes `(32,7,5)` y vectores `(32,35)`.

SHA256 del dataset: `095b431bdfa2d228eeaa337fb45cd963088cc13380c189e01436e446767502f8`.

Se entrenan los 32 patrones como entradas y objetivos. La reconstrucción se
binariza con `prediction >= 0.5`. Se cuentan errores por patrón: cumplir significa
que **cada uno de los 32 patrones** tiene cero o un error; un promedio bajo no
sustituye esta condición. Es una evaluación de reconstrucción del entrenamiento,
no una estimación de generalización a letras desconocidas.

El costo es `0.5 * mean(sum((prediction-target)^2, axis=1))`, equivalente a
`17.5 * MSE_por_pixel`. La función continua guía el entrenamiento; la condición
de aceptación se comprueba sobre las reconstrucciones binarias.

## Campañas realizadas y selección

El piloto base, con Adam 0.001 y semilla 42, completó 10000 épocas: máximo error 4,
error medio 0.96875, 14/32 patrones exactos y 21/32 dentro de tolerancia.
Sus [métricas](pilot/metrics.json) y [manifiesto](pilot/manifest.json) se conservan.

La campaña inicial por etapas completó seis corridas de arquitectura y cuatro
entradas de comparación de optimizadores, dos de ellas reutilizadas: ocho corridas
únicas completas, además de un intento interrumpido archivado. Eligió deep con
Momentum y semilla 43, cuyo máximo error fue 3. Esa selección no cumplió la
tolerancia y fue superada por la campaña factorial. El detalle histórico y el
presupuesto de reintento están en [comparison.md](../comparison.md).

La campaña factorial autorizada el 2026-11-08 comparó cinco arquitecturas, cuatro
optimizadores, tres tasas por optimizador y tres semillas (42, 43, 44). Todas las
corridas usaron lote 32, máximo 3000 épocas, evaluación cada 100 y parada al
cumplir tolerancia. Se completaron 180/180, con 538300 épocas consumidas sobre
un máximo de 540000. Dos corridas alcanzaron tolerancia; las otras 178 agotaron
el presupuesto. No se mezclan el piloto ni las etapas previas en estos resultados.

Se seleccionó por las medias entre semillas, en orden: máximo error, error medio
y MSE. Luego se eligió la mejor corrida de la variante ganadora con el mismo
orden individual. La ganadora es **deeper / RMSProp 0.002**; sus máximos de error
fueron 2, 2 y 1, respectivamente. Su tasa de éxito es **1/3 semillas**, por lo que
no se afirma convergencia consistente. Otra corrida, deeper / RMSProp 0.0005 /
semilla 43, también cumplió; su variante perdió por el criterio agregado.
Los resultados completos y las limitaciones se detallan en [comparison.md](../comparison.md).

## Modelo final y reconstrucción

Corrida: **`deeper-rmsprop-0p002-seed-44`**.

- Arquitectura: `35→40→20→10→2→10→20→40→35`.
- Activación interna y código: tanh; salida logística.
- RMSProp: tasa 0.002, rho 0.9, epsilon 1e-8; lote 32; semilla 44.
- Parada por tolerancia a las 2000 épocas, dentro del máximo de 3000.
- Sin L2, augmentation ni batch normalization; implementación sobre `tp3.shared`.

| Métrica recalculada desde el modelo guardado | Resultado |
| --- | ---: |
| Máximo de errores por patrón | **1** |
| Error medio de píxeles por patrón | 0.4375 |
| Patrones exactos | 18/32 |
| Patrones con exactamente un error | 14/32 |
| Patrones dentro de tolerancia | **32/32** |
| MSE por píxel | 0.013020395748171211 |
| Costo de reconstrucción | 0.2278569255929962 |

La [lámina de reconstrucciones](final/evaluation/reconstructions.png) muestra
los originales y sus reconstrucciones con los errores individuales.
El [modelo final](final/model.npz) es una copia exacta del
[modelo de la corrida ganadora](comparison-broad/deeper-rmsprop-0p002-seed-44/model.npz).
SHA256: `ad4e31ecaed339aa9a35404769284f55eff8a9fbff6ddad72c55aa1e9de73fca`.

[selection.json](final/selection.json) vincula campaña, variante, semilla,
configuración, dataset y hash. Los metadatos de `final/` usan rutas relativas a
`tp5/`; los manifiestos históricos conservan las rutas del entorno original.
`overnight-status.json` corresponde a la campaña anterior, no a la selección vigente.

## Recarga y reproducción de las figuras

Desde `tp5/`, instalar las dependencias de `requirements.txt`. Estos comandos
recargan el modelo final sin entrenar y usan destinos nuevos para conservar los
artefactos publicados:

```powershell
python -m src.train evaluate --model results/final/model.npz --output results/final-evaluation-reproduction
python -m src.explore --model results/final/model.npz --output results/final-latent-reproduction
python -m src.explore --model results/final/model.npz --config configs/exploration.json --output results/final-candidates-reproduction
```

El gráfico latente contiene 32 puntos etiquetados. La exploración utiliza dos
pares de patrones, nueve pasos por interpolación y una grilla de 9×9 sobre el
rango ocupado por los códigos: 99 candidatos en total. Cada candidato conserva
coordenadas, salida continua, imagen binaria y distancia mínima de Hamming a los
32 originales (`0` significa copia; mayor que `0`, candidato distinto).

La revisión sigue pendiente. Para registrar después una elección explícita,
usar `src.explore --review` según el [README](../README.md#interpolación-grilla-y-revisión-visual).
Si no se identifica una letra adecuada, registrar `no_suitable_letter` y mantener
pendiente esa demostración. No se trasladó una revisión del modelo anterior.

## Datos para futuros gráficos comparativos

Se conservan [runs.csv](comparison-broad/runs.csv) con las 180 corridas,
[aggregates.csv](comparison-broad/aggregates.csv) con las 60 variantes y
[summary.json](comparison-broad/summary.json) con métricas y errores por patrón.
Cada corrida guarda historia, configuración, predicciones y modelo.
Esto permite generar posteriormente gráficos por arquitectura, optimizador,
tasa y semilla, tasas de éxito, errores por carácter y curvas de aprendizaje
sin nuevos entrenamientos. Ya existe [comparison.png](comparison-broad/comparison.png);
no se generaron nuevos gráficos globales en esta actualización.

## Verificación y pendientes

Revisión del 2026-11-10: se recargaron los 180 modelos y se recalcularon las
predicciones y métricas; se verificaron configuraciones, hashes de modelos y
dataset, las 60 agregaciones y la selección. No se detectaron inconsistencias.
La evidencia de `final/` se regeneró sin entrenamiento, con el mismo hash del
ganador y revisión visual pendiente. Los archivos finales anteriores se
conservaron localmente en `tmp/tp5-final-update-2026-11-10/previous-final/`, fuera
de la entrega. Las fuentes, la wiki y el código de entrenamiento no se modificaron.

Para comprobar el código con pruebas cortas desde `tp5/`:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Las pruebas no sustituyen la evidencia experimental ni exigen convergencia.
El pendiente académico del apartado 1.a es la validación visual de una nueva
letra del modelo final. La reconstrucción dentro de tolerancia, el estudio
comparativo y el gráfico latente ya tienen evidencia guardada.
