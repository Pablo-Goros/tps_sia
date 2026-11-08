# 04: Visualizar los 32 caracteres en el espacio latente

**Spec:** [TP5 — Autoencoder básico para caracteres](../spec.md)
**Status:** done

## What to build

Un comando que cargue un modelo persistido y produzca el gráfico de dos dimensiones con los códigos de los 32 patrones, junto con coordenadas y etiquetas guardadas. El estudiante puede relacionar cada carácter original con su representación sin volver a entrenar.

Utilizar la operación pública de codificación del modelo y mantener la asociación con el orden del dataset, incluidos símbolos y DEL. Guardar la identificación del modelo y del dataset usados para que la figura y las coordenadas sean reproducibles.

Esta entrega puede usar el modelo base del ticket 02; no requiere esperar la comparación experimental. La integración final regenerará la evidencia con el modelo seleccionado. Incluir las instrucciones necesarias en el README y verificar el comportamiento mediante el flujo público de carga y visualización.

Contexto: [plan confirmado, visualización del paso 6](../../../../tp5/assignment_plan.md); criterios de aceptación 7–8 y 14 de la especificación.

## Acceptance criteria

- [x] El comando documentado consume un modelo guardado y el dataset original y genera el gráfico sin ejecutar entrenamiento.
- [x] Aparecen los 32 patrones en dos dimensiones con etiquetas que permiten identificarlos; no se pierden patrones por coincidencia de coordenadas ni por etiquetas de símbolos vacías.
- [x] Se guardan exactamente 32 pares de coordenadas y su asociación inequívoca con los patrones.
- [x] Las coordenadas guardadas coinciden con la codificación pública del mismo modelo y se corresponden con los puntos representados.
- [x] Figura y metadatos identifican el modelo y dataset utilizados y pueden regenerarse desde los artefactos persistidos.
- [x] Una prueba del flujo público comprueba las coordenadas, etiquetas y archivos producidos sin inspeccionar detalles internos del MLP ni exigir una distribución particular de los códigos.
- [x] El README explica cómo visualizar otro modelo guardado, incluido el que resulte de la comparación.

## Blocked by

- [02: Entrenar, guardar y evaluar un autoencoder básico](02-train-and-evaluate-basic-autoencoder.md)
