# TP5, apartado 1.a — resultados y reproducción

Estado: entrega de código verificable, con evidencia experimental parcial.
La selección final entre arquitecturas y optimizadores y la demostración de
una nueva letra siguen pendientes. Las campañas costosas restantes se ejecutarán
por separado, según la decisión del estudiante; no se presentan como realizadas.

## Requisitos y fundamento

La [consigna PA-004, p. 1, apartado 1.a](../../../sources/extracted/PA-004.md)
pide una representación de dos dimensiones de los 32 patrones de 5×7, estudiar
arquitecturas y optimización con un máximo de un píxel incorrecto por patrón,
graficar sus códigos y generar una letra nueva. El
[original de la consigna](../../../sources/practical-assignment/SIATP5.pdf)
permanece intacto. El alcance excluye denoising y VAE.

Un encoder transforma la entrada en un código; un decoder intenta recuperar el
patrón original. Se entrena cada entrada usando ese mismo patrón como objetivo
y se ajustan los parámetros mediante aprendizaje de MLP. Este es el fundamento
docente de la implementación, no un resultado experimental: [OFF-022, pp. 3–6](../../../sources/extracted/OFF-022.md)
y [wiki, Funcionamiento](../../../wiki/topics/autoencoders.md#funcionamiento).
La exploración de coordenadas latentes permite producir imágenes, pero un
autoencoder básico no garantiza estructura estadística que vuelva interpretables
todas las regiones: [OFF-022, pp. 28–38](../../../sources/extracted/OFF-022.md).

## Evidencia para los cuatro puntos del apartado 1.a

| Punto | Implementación y evidencia disponible | Resultado pendiente |
| --- | --- | --- |
| 1. Encoder y decoder con código de dos dimensiones | [Interfaz del modelo](../src/autoencoder.py); base `35→20→2→20→35`; persistencia y equivalencia codificar→decodificar probadas | Identificar el modelo elegido por la comparación completa |
| 2. Estudio de arquitecturas y técnicas de optimización; tolerancia individual | [Piloto](pilot/README.md), [métricas de los 32 patrones](pilot/metrics.json), [reconstrucciones](pilot/reconstructions.png), [curvas](pilot/curves.png); [protocolo comparativo](../comparison.md) | Completar o revisar las etapas acordadas y seleccionar un modelo sin ocultar corridas interrumpidas |
| 3. Gráfico en dos dimensiones | [Comando de visualización](../README.md#visualización-del-espacio-latente), pruebas de 32 etiquetas y coordenadas recargables | Regenerar la figura con el modelo finalmente seleccionado |
| 4. Nueva letra fuera del entrenamiento | [Interpolaciones, grilla y revisión](../README.md#interpolación-grilla-y-revisión-visual); novedad binaria y regeneración verificadas por pruebas | Ejecutar exploración del modelo final y obtener revisión visual explícita del estudiante |

Los puntos 3 y 4 tienen código y pruebas; esto no equivale a haber producido y
validado la evidencia final del ejercicio. Un candidato nuevo se diferencia de
todos los originales; solo después de una revisión visual puede llamarse nueva
letra, conforme al [glosario](../../GLOSSARY.md).

## Datos, arquitectura y medición

Se conserva sin transcribir `Font3[32][7]` de [font.h](../data/font.h), códigos
`0x60`–`0x7f` en orden, incluido DEL. Cada entero se interpreta como cinco bits
de izquierda a derecha, del más significativo al menos significativo; las siete
filas mantienen su orientación. Las imágenes tienen forma `(32,7,5)` y los
vectores `(32,35)`. SHA256 del original:
`095b431bdfa2d228eeaa337fb45cd963088cc13380c189e01436e446767502f8`.

La arquitectura base tiene encoder `35→20→2` y decoder `2→20→35`, tanh en las
capas internas, incluido el código, y salida logística. El wrapper reutiliza el
entrenamiento, retropropagación y optimizadores de `tp3.shared`; no incorpora
regularización, augmentation ni batch normalization. Las alternativas propuestas
y autorizadas se describen en [comparison.md](../comparison.md).

Se usan los 32 patrones tanto como entradas como objetivos. La salida binaria
es `prediction >= 0.5`. Se informa el número de píxeles incorrectos por patrón,
el máximo, el promedio y las fracciones exacta y dentro de tolerancia. El conjunto
solo está aprendido dentro de tolerancia cuando los **32 patrones**, cada uno,
tienen como máximo un error. La pérdida del MLP es
`0.5 * mean(sum((prediction-target)^2, axis=1))`; el MSE por píxel promedia los
1120 valores, de modo que la pérdida equivale a `17.5 * MSE`. Ninguna de estas
magnitudes continuas sustituye el criterio binario individual.

## Piloto efectivamente ejecutado

Modelo: [pilot/model.npz](pilot/model.npz), corrida `pilot`, semilla 42,
[configuración efectiva](pilot/config.json), [manifiesto](pilot/manifest.json).
Adam, tasa 0.001, lote 32, máximo 10.000 épocas, evaluación cada 100; se acordó
parar si todos los patrones cumplían la tolerancia. Se completaron 10.000 épocas,
con motivo `max_epochs`, duración registrada 1770.12 segundos.

| Métrica final | Resultado |
| --- | --- |
| Máximo de errores por patrón | 4 |
| Error medio de píxeles por patrón | 0.96875 |
| Patrones exactos | 14/32 |
| Patrones dentro de tolerancia | 21/32 |
| MSE por píxel | 0.027751202772635465 |
| Pérdida de entrenamiento | 0.4856460485211206 |

El conjunto completo **no fue aprendido dentro de tolerancia** por este piloto.
El peor patrón es `0x73 s`, con cuatro errores. Los errores individuales están
identificados en [metrics.json](pilot/metrics.json) y visibles junto a cada
reconstrucción en [reconstructions.png](pilot/reconstructions.png). Guardar y
recargar reprodujo sus códigos y métricas sin repetir entrenamiento. Este
modelo es evidencia del piloto; todavía no es una selección final comparativa.

El piloto no demuestra imposibilidad de aprender el conjunto ni autoriza adoptar
un subconjunto. No se redujo el dataset. La reconstrucción del entrenamiento
no estima generalización a letras desconocidas.

## Comparación y continuación

El [protocolo autorizado](../comparison.md) compara primero base, ancha y profunda
con Adam y semillas 42 y 43; luego compara Adam con Momentum sobre la arquitectura
seleccionada, reutilizando sus corridas Adam. Cada corrida tiene como máximo
3000 épocas, lote 32, evaluación cada 100 y parada al alcanzar tolerancia.
El máximo autorizado es ocho corridas únicas y 24.000 épocas nuevas, además del
piloto previo. La campaña detenida y cualquier ejecución posterior deben conservar
sus configuraciones, manifiestos, `progress.json`, tablas, motivos de parada y
resúmenes; una interrupción es un resultado registrado, no una convergencia.

Se elige variante por medias entre semillas, ordenando máximo error, error medio
y MSE; después se elige una corrida concreta de esa variante con el mismo orden.
Una variante incompleta no puede ganar. La elección final debe conservar corrida,
semilla, configuración y SHA256 de modelo y dataset. Las instrucciones para
completar las etapas sin sobrescribir evidencia están en [comparison.md](../comparison.md).
Si las corridas acordadas no alcanzan tolerancia, se revisa la evidencia con el
estudiante antes de ampliar presupuesto, cambiar el objetivo o justificar un
subconjunto. Una búsqueda limitada no es prueba matemática de imposibilidad.

## Recarga, evaluación y exploración final

Desde `tp5/`, instalar dependencias siguiendo el [README](../README.md).
La evaluación ya disponible del piloto no requiere entrenamiento:

```powershell
python -m src.train evaluate --model results/pilot/model.npz --output results/pilot-evaluation-new
```

Una vez seleccionada una corrida, sustituir `RUTA_MODELO_FINAL` por su `model.npz`.
Los destinos siguientes deben ser nuevos para conservar evidencia anterior:

```powershell
python -m src.train evaluate --model RUTA_MODELO_FINAL --output results/final-evaluation
python -m src.explore --model RUTA_MODELO_FINAL --output results/final-latent
python -m src.explore --model RUTA_MODELO_FINAL --config configs/exploration.json --output results/final-candidates
```

Estos tres comandos recargan el mismo modelo sin entrenarlo. Contrastar el SHA256
de los metadatos y las 32 métricas con la corrida seleccionada antes de presentar
las figuras. Las coordenadas de `latent.json` deben coincidir con `encode(X)`;
las de `candidates.json`, pasadas a `decode`, regeneran las imágenes. La grilla
recorre el rango ocupado por los códigos, dentro de `[-1,1]`. Cada candidato
registra su distancia mínima de Hamming respecto de los 32 originales: `d=0`
es una copia y `d>0` es un candidato nuevo.

El estudiante debe revisar las láminas del modelo final antes de registrar una
nueva letra. El [README](../README.md#interpolación-grilla-y-revisión-visual)
documenta cómo registrar la decisión y las coordenadas. Si no identifica una
letra adecuada, se conserva `no_suitable_letter` y la demostración permanece
pendiente. Si cambia el modelo, la revisión anterior pierde vigencia para la
entrega final y se repite sobre sus nuevas imágenes.

## Verificaciones y límites de la entrega

Desde `tp5/`:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Verificación del 2026-11-08: el ejecutor descubrió y ejecutó **12 pruebas**, todas aprobadas. El ejecutor debe informar un número positivo de pruebas. La cobertura pública
incluye carga y orientación, rechazo de entradas inválidas, umbral 0.5 y errores
conocidos, entrenamiento corto, continuidad y recarga, equivalencia de códigos y
reconstrucciones, artefactos, comparación resumida y regeneración de candidatos.
Las pruebas cortas no exigen convergencia ni sustituyen las corridas académicas.
No se reescribió ni modificó el TP3. El archivo original conserva su hash; las
fuentes académicas y la wiki permanecen intactas.

Pendientes de la entrega académica: completar/revisar la campaña, elegir un modelo
trazable, regenerar toda la evidencia desde ese modelo y revisar visualmente una
nueva letra. El código disponible permite realizar esas operaciones; este informe
no las acredita antes de que existan sus artefactos y la decisión del estudiante.
