# 02: Entrenar, guardar y evaluar un autoencoder básico

**Spec:** [TP5 — Autoencoder básico para caracteres](../spec.md)
**Status:** ready-for-agent

## What to build

El primer flujo completo de aprendizaje: cargar los 32 patrones, configurar y entrenar el autoencoder base, guardar el modelo y volver a cargarlo para reconstruir y evaluar los caracteres. El estudiante obtiene curvas, una lámina de originales y reconstrucciones y métricas por patrón mediante comandos y configuración JSON.

Encapsular el MLP y los optimizadores existentes del TP3 desde el TP5. La base acordada es `35→20→2→20→35`, con tanh interna, incluida la capa latente, salida logística, error cuadrático y Adam. Exponer codificación, decodificación, reconstrucción y persistencia; mantener el entrenamiento y la retropropagación en el código reutilizado. Evitar refactorizaciones amplias del TP3.

Usar los 32 patrones como entradas y objetivos. Binarizar con `valor >= 0.5`. Informar errores por patrón, máximo de errores, proporción de patrones exactos y dentro de tolerancia, MSE por píxel y pérdida de entrenamiento con su normalización documentada. No usar métricas de clasificación para evaluar las imágenes ni presentar los mismos datos como una validación independiente.

La entrega incluye acordar con el usuario el presupuesto inicial antes de entrenar experimentalmente: semilla, tasa, tamaño de lote, límite de épocas, frecuencia de evaluación y regla de parada. Presentar alternativas concretas y esperar su respuesta, conforme a la puerta de decisión del plan. Registrar lo acordado sin decidir silenciosamente los valores pendientes. Las pruebas automáticas cortas de consistencia no sustituyen ese experimento ni exigen convergencia.

Conservar configuración efectiva, identificación del dataset, versiones, duración, motivo de parada, historia, modelo y métricas. Las evaluaciones intermedias deben mantener pesos y estado del optimizador. Documentar cómo entrenar y cómo evaluar un modelo persistido sin repetir el entrenamiento.

Contexto: [plan confirmado, pasos 2–4](../../../../tp5/assignment_plan.md). Este ticket reúne esos comportamientos en una entrega demostrable; el presupuesto de comparación que todavía no se haya acordado se resuelve en el ticket 03. Criterios de aceptación 4–11 y 18 de la especificación, en lo relativo a la base.

## Acceptance criteria

- [ ] El presupuesto inicial se presenta al usuario y su elección queda registrada antes de ejecutar entrenamiento experimental; una decisión diferida mantiene esa ejecución pendiente.
- [ ] Un comando documentado consume JSON y genera modelo persistido, manifiesto, historia, métricas y gráficos usando el dataset original completo.
- [ ] El modelo base utiliza la arquitectura, activaciones, pérdida y optimizador acordados y reutiliza el entrenamiento del TP3.
- [ ] Codificar los 32 patrones produce `(32,2)` y reconstruir produce `(32,35)` con valores finitos; decodificar los códigos equivale numéricamente a reconstruir.
- [ ] Guardar y recargar permite reproducir los códigos y las reconstrucciones dentro de tolerancia numérica, sin entrenar nuevamente.
- [ ] Las métricas verifican el límite de un píxel individualmente para los 32 patrones; solo se declara conjunto aprendido dentro de tolerancia cuando todos cumplen.
- [ ] Se prueban errores conocidos, incluido el umbral exactamente igual a 0.5, y se diferencia el MSE por píxel de la pérdida usada al entrenar.
- [ ] Las curvas y la lámina de reconstrucciones coinciden con las métricas y se registra el motivo de parada, incluso si no se alcanza el objetivo académico.
- [ ] Una prueba corta del flujo público carga→entrenamiento→guardado→recarga→evaluación pasa sin exigir aprender los 32 patrones ni repetir pruebas internas de gradientes y optimizadores.
- [ ] README y registro experimental permiten reproducir la corrida; un fallo de convergencia se informa y no se oculta como éxito.

## Blocked by

- [01: Inspeccionar el dataset original](01-inspect-original-dataset.md)
