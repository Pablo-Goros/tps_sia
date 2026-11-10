# 03: Comparar arquitecturas y optimizadores por etapas

**Spec:** [TP5 — Autoencoder básico para caracteres](../spec.md)
**Status:** in-progress

## What to build

Un flujo de comparación que use la ejecución reproducible ya disponible para estudiar variantes del autoencoder con varias semillas y seleccionar un modelo cuya corrida se pueda rastrear. El estudiante podrá consultar tablas y gráficos de resultados sin perder las corridas que fallaron o no alcanzaron la tolerancia.

Antes de ejecutar la campaña, completar con el usuario el protocolo y presupuesto pendientes: arquitecturas exactas, optimizadores, semillas compartidas, tasas, lotes, límites, frecuencia de evaluación, reglas de parada y selección. Mostrar alternativas con costo total explícito y esperar su elección. La propuesta del plan compara primero la base con una red más ancha y otra más profunda, conservando el código de dos unidades, y después Adam con Momentum sobre la arquitectura seleccionada; no convertir esa propuesta en una decisión ya aprobada.

Ejecutar solo las etapas acordadas y conservar las configuraciones efectivas y los artefactos por corrida. Resumir máximo y promedio de errores de píxeles, MSE, fracciones de éxito, épocas y tiempos; distinguir los resultados por semilla de los resúmenes agregados. Identificar inequívocamente el modelo seleccionado y su criterio de selección.

Si las corridas acordadas no alcanzan el objetivo, revisar la evidencia con el usuario antes de ampliar el presupuesto, modificar el planteo o recurrir a un subconjunto. La ausencia de convergencia en esta búsqueda no constituye una demostración matemática de imposibilidad.

Contexto: [plan confirmado, puerta de decisión del paso 3 y paso 5](../../../../tp5/assignment_plan.md); criterios de aceptación 5, 10–13 y 18 de la especificación.

## Acceptance criteria

- [ ] El protocolo concreto y su presupuesto total quedan acordados y registrados antes de ejecutar la comparación; no se ejecutan combinaciones o ampliaciones no acordadas.
- [ ] La comparación se inicia mediante configuración y comandos documentados y reutiliza el flujo de entrenamiento y evaluación del ticket 02.
- [ ] Todas las variantes conservan 35 entradas, 35 salidas y un espacio latente de dos dimensiones, y se evalúan sobre los 32 patrones con el mismo criterio binario.
- [ ] Se ejecutan las arquitecturas, técnicas de optimización y semillas acordadas o se registran explícitamente los motivos de interrupción.
- [ ] Tablas y gráficos permiten contrastar error por píxeles, MSE, éxito y costo de entrenamiento, con trazabilidad hacia cada configuración y semilla; se incluyen corridas fallidas.
- [ ] El modelo seleccionado referencia una corrida y un artefacto concretos y su elección sigue el criterio acordado.
- [ ] El objetivo de aprender los 32 patrones dentro de tolerancia se informa explícitamente; si no se alcanza, el usuario revisa la evidencia y decide la continuación sin que se adopte silenciosamente un subconjunto.
- [ ] Una prueba pequeña del flujo de comparación comprueba la correspondencia entre resultados individuales y resumen sin lanzar una búsqueda larga ni imponer convergencia de una semilla arbitraria.
- [ ] El README y el registro de resultados explican cómo reproducir las etapas realizadas y sus límites.

## Blocked by

- [02: Entrenar, guardar y evaluar un autoencoder básico](02-train-and-evaluate-basic-autoencoder.md)
