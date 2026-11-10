# 06: Integrar la evidencia del ejercicio 1.a

**Spec:** [TP5 — Autoencoder básico para caracteres](../spec.md)
**Status:** in-progress

## What to build

Una entrega reproducible del apartado 1.a que integre la comparación experimental y la exploración visual sobre el modelo finalmente seleccionado. Completar README e informe con comandos efectivos, fundamentos, configuraciones, resultados y limitaciones, enlazando cada obligación de la consigna con evidencia verificable.

Recargar el modelo elegido por la comparación y recalcular sus métricas, gráfico latente, interpolaciones y grilla sin volver a entrenarlo. Si el modelo difiere del utilizado para elegir la letra en el ticket 05, regenerar los candidatos y repetir la revisión visual con el usuario. No atribuir imágenes de un modelo a otro.

Cubrir arquitectura de encoder y decoder, estudio de arquitecturas y optimización, máximo error por patrón sobre el conjunto completo, representación latente y generación. Distinguir el requisito académico de los resultados efectivamente alcanzados. Si se utilizó un subconjunto mediante decisión explícita, mostrar la evidencia y la justificación exigidas por la consigna. Citar la consigna para las tareas y las fuentes docentes para las explicaciones.

Ejecutar las verificaciones relevantes del flujo completo y documentar los comandos efectivos de pruebas. Reutilizar la evidencia de entrenamientos ya validados en lugar de repetir búsquedas largas. Si hubo modificaciones compartidas en el TP3, ejecutar además sus comprobaciones correspondientes. Conservar intactas las fuentes originales y no modificar la wiki.

Contexto: [plan confirmado, integración de los pasos 5–7](../../../../tp5/assignment_plan.md); todos los criterios de aceptación de la especificación, especialmente 10, 12–18.

## Acceptance criteria

- [ ] El informe identifica inequívocamente el modelo seleccionado, su configuración, semilla, corrida y dataset; métricas y figuras finales provienen de ese mismo modelo.
- [ ] Recargar el modelo permite recalcular métricas y regenerar la exploración siguiendo los comandos documentados, sin repetir el entrenamiento.
- [ ] Las reconstrucciones de los 32 patrones tienen errores individuales visibles y el informe declara explícitamente si el conjunto está aprendido dentro de tolerancia.
- [ ] La comparación incluye las técnicas y arquitecturas estudiadas y sus límites; cualquier subconjunto está respaldado por una decisión explícita y una justificación basada en evidencia.
- [ ] El gráfico latente muestra los 32 patrones identificados y las imágenes de generación conservan sus coordenadas y comprobaciones de novedad.
- [ ] La nueva letra fue revisada visualmente con el usuario para el modelo final; si cambia el modelo se vuelve a revisar, y una demostración todavía pendiente no se presenta como completada.
- [ ] El informe relaciona los cuatro puntos del apartado 1.a con resultados concretos y separa reconstrucción del entrenamiento de cualquier afirmación de generalización.
- [ ] El README contiene instalación, comandos efectivos, configuración, entrenamiento, comparación, recarga, evaluación, exploración y pruebas desde el TP5.
- [ ] Las verificaciones del flujo público y las pruebas puntuales acordadas pasan; se comprueba que el ejecutor realmente ejecute las pruebas declaradas. Si se modificó TP3, pasan las comprobaciones compartidas pertinentes.
- [ ] Se conserva el original del dataset y las fuentes académicas; las conclusiones se apoyan en evidencia producida y las explicaciones teóricas tienen referencias docentes.

## Blocked by

- [03: Comparar arquitecturas y optimizadores por etapas](03-compare-architectures-and-optimizers.md)
- [05: Explorar y validar candidatos nuevos](05-explore-and-validate-new-candidates.md)
