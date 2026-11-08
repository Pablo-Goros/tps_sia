# Piloto base — 2026-11-08

Presupuesto autorizado por el usuario y registrado en `../../assignment_plan.md`:
una corrida Adam, semilla 42, tasa 0.001, lote 32, arquitectura
`35→20→2→20→35`, máximo 10.000 épocas, evaluación cada 100 y parada si los
32 patrones alcanzan individualmente como máximo un error.

Se completaron las 10.000 épocas. Motivo de parada: `max_epochs`.
Duración registrada: 1770.12 segundos (incluye evaluación y generación de
artefactos). El MLP compartido conserva historia para rollback por época;
el costo de ejecución no debe extrapolarse linealmente a otros presupuestos.

El conjunto **no fue aprendido dentro de tolerancia**:

| Métrica | Resultado |
| --- | --- |
| Máximo de píxeles incorrectos por patrón | 4 |
| Promedio de píxeles incorrectos | 0.96875 |
| Patrones exactos | 14/32 (43.75 %) |
| Patrones con como máximo un error | 21/32 (65.625 %) |
| MSE por píxel | 0.027751202772635465 |
| Pérdida de entrenamiento | 0.4856460485211206 |

La pérdida es la mitad del error cuadrático sumado por patrón y promediado
entre patrones; el MSE promedia cada píxel. Se usa umbral `>= 0.5`.
Los errores individuales y las etiquetas están en [metrics.json](metrics.json),
la comparación visual en [reconstructions.png](reconstructions.png) y la
evolución en [curves.png](curves.png). El bajo promedio no reemplaza el límite
individual. El peor patrón es `0x73 s`, con cuatro errores.

[manifest.json](manifest.json) registra configuración, versiones, hash del
dataset, modelo, duración y parada. Se agregó `effective_model` recargando
el modelo guardado después de la corrida para explicitar también los valores
predeterminados del MLP y Adam; no se repitió entrenamiento. Las rutas absolutas
identifican el worktree donde se ejecutó; los archivos se trasladan junto con
el modelo. [model.npz](model.npz) conserva parámetros y estado de Adam.
[history.json](history.json) conserva cada época y las 101 evaluaciones.

Comandos desde `tp5/`:

```powershell
python -m src.train train --config configs/pilot.json --output results/pilot-repeat
python -m src.train evaluate --model results/pilot/model.npz --output results/pilot-evaluation
```

La recarga y evaluación independiente reprodujeron exactamente las métricas
guardadas. Las siete pruebas automáticas del TP5 pasan y no exigen convergencia.
El dataset original conserva SHA256
`095b431bdfa2d228eeaa337fb45cd963088cc13380c189e01436e446767502f8`.

Todos los 32 patrones se usaron como entrada y objetivo. Esta es evidencia de
reconstrucción del entrenamiento, no de generalización. No se adoptó ningún
subconjunto ni se amplió el presupuesto. La siguiente comparación requiere
revisar esta evidencia y acordar protocolo y presupuesto con el usuario.
El piloto por sí solo no demuestra imposibilidad de aprender el conjunto.
