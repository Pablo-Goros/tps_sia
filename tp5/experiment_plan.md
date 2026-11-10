# TP5 — Plan de experimentos a partir de la presentación

Fecha: 2026-11-10. Estado: completado. Se ejecutaron 43 corridas nuevas y
54300 épocas. La confirmación obtuvo 5/5 éxitos frente a 0/5 de la referencia.
Resultados y decisiones: [presentation-results.md](presentation-results.md).
El usuario autoriza probar los cambios y pide definir antes las decisiones.
Alcance: AE básico, ejercicio 1.a. Denoising y VAE quedan fuera de esta campaña.

Enmienda de ejecución autorizada el mismo día: los artefactos broad/final no
estaban en este checkout. El usuario autorizó reconstruir las tres referencias.
El máximo pasa a **43 corridas nuevas y 321000 épocas**. Se usa el entorno
disponible (Python 3.14.4, NumPy 2.4.4, Matplotlib 3.11.1) para todas las
comparaciones nuevas; no se identifican estas referencias con las históricas.
El piloto antiguo reproduce sus métricas. La diferencia de hash del archivo
font.h corresponde únicamente a CRLF frente a LF; su contenido se conserva.
La campaña se ejecuta con `src.presentation_campaign`; decisiones y evidencia
se guardan en `results/presentation-campaign-2026-11-10/`.

## Referencias y objetivo

Presentación del Grupo 02, diapositivas 7–10, incluidas sus configuraciones
como imágenes. Propone BCE, latente lineal, Adam, tasa 0.003, descenso cosenoidal
hasta cero, arquitectura 35-25-15-8-2-8-15-25-35 y hasta 15000 épocas.
Son resultados de otro grupo, no evidencia reproducida por nosotros.
Base local: comparison.md y results/final/selection.json; arquitectura deeper,
RMSProp 0.002, rho 0.9, tanh también en el código, salida logística y error
cuadrático. Objetivo: mejorar la consistencia de reconstrucción de los 32
patrones, manteniendo dos dimensiones latentes. No medir generalización a
caracteres no vistos a partir de estas reconstrucciones.

## Reglas comunes y decisiones

Semillas exploratorias emparejadas: 42, 43, 44. Corridas desde cero, lote 32,
evaluación cada 100 épocas, umbral 0.5, parada si todos los caracteres cumplen
como máximo un píxel incorrecto. Mantener datos, inicialización y demás opciones
efectivas de la referencia; verificar igualdad de pesos iniciales para cambios
que no alteran dimensiones. No añadir regularización ni ruido.

Orden de selección nuevo, declarado antes de ejecutar: mayor cantidad de semillas
que cumplen; menor media del máximo error; menor error binario medio; menor MSE
por píxel. Nunca comparar valores de BCE con valores de pérdida cuadrática para
seleccionar. Empates exactos conservan la configuración anterior. El criterio
histórico max_mean_mse se reporta por separado; no se cambia silenciosamente.
La media de épocas hasta cumplimiento se informa, pero no decide por sí sola.
Un avance solo en criterios secundarios se etiqueta como tal, no como mayor
consistencia. Con tres semillas las decisiones son exploratorias.

Cada etapa elige una configuración provisional que será la base de la siguiente.
No se sustituyen los artefactos finales durante la búsqueda. Si una combinación
posterior rescata un componente antes desfavorable, se adopta la combinación
según el mismo criterio. No se infiere que sus componentes funcionen aislados.

## Etapas

1. **Preparación, sin campaña.** Añadir opciones explícitas para pérdida y
   activación latente conservando los valores predeterminados. Adaptar forward,
   backward, encode/decode, configuración, historial, comparador y persistencia.
   BCE estable desde logits; conservar reducción suma sobre píxeles y media
   sobre muestras para evitar introducir un factor 35 accidental. Verificar
   gradientes por diferencias finitas, saturación, equivalencia encode/decode,
   recarga y compatibilidad de modelos anteriores. Como la MLP se comparte con
   TP3, ejecutar sus pruebas relevantes además de las de TP5. Verificar la
   semántica y persistencia del scheduler cosenoidal antes de usarlo.

2. **Pérdida × activación latente.** Comparar cuadrático/tanh, BCE/tanh,
   cuadrático/lineal y BCE/lineal, todos con RMSProp 0.002 y arquitectura actual.
   Máximo 3000 épocas. Reutilizar las tres referencias originales solo tras
   reproducir métricas y verificar configuración, dataset e inicialización.
   Nueve corridas nuevas, máximo 27000 épocas. Si la reproducción falla,
   detener la campaña y diagnosticar; no sumar corridas automáticamente.
   Siempre probar las cuatro celdas: aunque BCE aislado empeore, BCE/lineal
   podría mejorar. Seleccionar la combinación ganadora; si ninguna supera la
   base, conservar cuadrático/tanh.

3. **Optimizador.** Sobre la combinación ganadora, comparar RMSProp con Adam,
   ambos con tasa 0.002 constante y 3000 épocas. RMSProp se reutiliza de la
   etapa anterior; Adam usa beta1=0.9, beta2=0.999, epsilon=1e-8. Tres corridas
   nuevas, máximo 9000 épocas. Si Adam no mejora, conservar RMSProp. Esto
   compara ambos optimizadores en esa tasa, no sus mejores ajustes posibles.

4. **Tasa × scheduler.** Con el optimizador y combinación elegidos, comparar
   tasas iniciales 0.002/0.003 × constante/cosenoidal hasta cero en 3000 épocas.
   Reutilizar 0.002 constante. Nueve corridas nuevas, máximo 27000 épocas.
   Probar 0.003 con y sin scheduler aunque una sola de esas opciones empeore;
   permite detectar interacción. Seleccionar según la regla común.

5. **Arquitectura.** Comparar deeper actual con la de la presentación:
   35-25-15-8-2-8-15-25-35. Mantener la configuración ganadora de etapa 4.
   Tres corridas nuevas, máximo 9000 épocas; reutilizar deeper. Si la nueva
   arquitectura pierde o empata exactamente, conservar la actual. La elección
   es condicional a los hiperparámetros anteriores, no una búsqueda exhaustiva.

6. **Presupuesto de 15000 épocas.** Sobre la configuración provisional de etapa
   5, comparar la política de tasas seleccionada con su alternativa (constante
   o cosenoidal), misma tasa inicial, hasta 15000 épocas y parada por tolerancia.
   Seis corridas nuevas desde cero, máximo 90000 épocas. El coseno se redefine
   sobre 15000 épocas: no se continúa una agenda que ya llegó a cero. Si la
   tasa finalista es 0.002, esta etapa no constituye una réplica exacta de la
   receta Adam/0.003 de la presentación. Comparar también con el candidato de
   3000 épocas; adoptar el presupuesto largo solo si mejora según el criterio.
   Si empatan, conservar el presupuesto corto. Mantener resultados intermedios
   para explicar el efecto de más entrenamiento.

7. **Confirmación con semillas nuevas.** Congelar un único finalista antes de
   revelar 45–49. Compararlo con la referencia histórica en esas cinco semillas,
   con el mismo máximo de épocas del finalista y la misma regla de parada.
   Diez corridas nuevas, hasta 150000 épocas si se eligieron 15000; hasta 30000
   si se conservaron 3000. Confirmar solo si el finalista supera la referencia
   según el criterio común; si pierde o empata, conservar el modelo final actual
   y reportar que la mejora exploratoria no se confirmó. No ajustar de nuevo
   sobre estas semillas ni abrir barridos automáticos.

## Presupuesto, publicación y límites

Máximo: 40 corridas nuevas y 312000 épocas (con parada anticipada normalmente
serán menos). Sin reintentos, semillas adicionales ni barridos no enumerados.
El límite es por épocas, no una promesa de duración. Registrar tiempo por etapa;
si aparecen fallos numéricos o incompatibilidades, detener la etapa, registrar
la causa y corregir antes de reconsiderar la ejecución.

Cada corrida conserva configuración efectiva, semilla, pesos, estado del
optimizador, tasas, historia, métricas por carácter y hashes. Usar directorios
nuevos de resultados y comprobar hashes de evidencia anterior. Mantener el
entorno efectivo de las referencias y documentarlo; no comparar velocidades
sin control equivalente de hilos y equipo.

Solo tras confirmar una configuración superior, seleccionar dentro de ella una
corrida que cumpla todos los patrones, por máximo error, error medio, MSE y
semilla como desempate. Guardarla primero como candidata separada y verificar
recarga. Regenerar reconstrucciones, mapa latente y candidatos de letras;
guardar la procedencia de la selección anterior antes de actualizar la selección
vigente. El latente lineal exige ajustar el rango de exploración a los códigos
observados. La validez visual de una letra nueva sigue requiriendo revisión del
usuario: reconstrucción buena no la garantiza.

Si ninguna configuración mejora o ninguna corrida final cumple, conservar el
modelo existente. Documentar también resultados negativos, discrepancias con
el ranking histórico y límites de la búsqueda secuencial. No modificar la wiki.
