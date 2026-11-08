# 05: Explorar y validar candidatos nuevos

**Spec:** [TP5 — Autoencoder básico para caracteres](../spec.md)
**Status:** in-progress

## What to build

Extender el flujo público de exploración para producir interpolaciones entre códigos de caracteres y una grilla de imágenes decodificadas. El estudiante podrá inspeccionar candidatos nuevos, conocer su diferencia respecto del entrenamiento y conservar las coordenadas necesarias para regenerarlos.

Configurar los pares de caracteres y la resolución de exploración. Usar el modelo persistido y sus códigos, manteniendo las coordenadas dentro de `[-1,1]` para la base con tanh; explorar la grilla en el rango ocupado por los códigos. Registrar coordenadas, identificación del modelo y parámetros de exploración junto con las imágenes.

Binarizar con el mismo umbral de reconstrucción y comparar cada candidato presentado con los 32 patrones. Una distancia de Hamming mínima igual a cero indica una copia, no un candidato nuevo. Presentar los candidatos nuevos al usuario y esperar su revisión visual antes de identificar una nueva letra. Si no encuentra ninguno adecuado, registrar la limitación y acordar otra exploración; no dar la demostración por cumplida.

Puede trabajarse con el modelo base mientras se completa la comparación. La elección visual es válida para el modelo que generó las imágenes; la integración final debe revisar su vigencia si cambia ese modelo.

Contexto: [plan confirmado, generación del paso 6](../../../../tp5/assignment_plan.md), [glosario](../../../../GLOSSARY.md); criterios de aceptación 15–17 de la especificación.

## Acceptance criteria

- [ ] Los comandos y configuraciones documentados generan interpolaciones y una grilla desde un modelo guardado, sin repetir el entrenamiento.
- [ ] Las imágenes guardadas pueden regenerarse mediante sus coordenadas y el modelo registrado; la configuración conserva pares, resolución y rango utilizados.
- [ ] Se compara cada candidato presentado con los 32 patrones binarios usando el umbral `>= 0.5` y se registra su distancia mínima.
- [ ] Las copias exactas se identifican como tales y solo las imágenes distintas de todos los patrones se denominan candidatos nuevos.
- [ ] El usuario puede inspeccionar láminas y revisar visualmente los candidatos; la elección y sus coordenadas quedan registradas antes de declarar una nueva letra.
- [ ] Si no hay una letra adecuada, la demostración queda pendiente y el resultado observado se conserva sin sustituir la revisión visual por una métrica automática.
- [ ] Las pruebas del flujo público verifican regeneración y distancias conocidas, incluidas una copia exacta y una imagen distinta; no intentan determinar automáticamente si una imagen es una letra.
- [ ] El README explica cómo reproducir y revisar los candidatos y los límites de explorar un espacio latente sin estructura generativa garantizada.

## Blocked by

- [04: Visualizar los 32 caracteres en el espacio latente](04-visualize-character-latent-space.md)
