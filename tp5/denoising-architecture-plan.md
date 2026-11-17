# Comparación de arquitecturas de denoising — 2026-11-17

Autorizada por el usuario en la conversación: las tres arquitecturas siguientes,
tres semillas y las cuatro dimensiones. Se permite reutilizar evidencia compatible.
Este protocolo queda fijado antes de iniciar los entrenamientos.

Ejecución completada: **36/36 combinaciones**, 33 corridas nuevas y tres
reutilizadas; 99.000 épocas nuevas. Fase de entrenamiento: 40,51 minutos.
La selección por validación es `simple-d20`, representante semilla 43.
El [informe](results/denoising-architectures-2026-11-17/report.md) contiene las
tablas, evaluación con ruido nuevo y gráficos. Se conserva la campaña anterior.

| Familia | Arquitectura, con `d` central |
|---|---|
| simple | 35→40→20→d→20→40→35 |
| current | 35→40→20→10→d→10→20→40→35 |
| deeper | 35→40→30→20→10→d→10→20→30→40→35 |

- `d`: 5, 10, 15 y 20; semillas: 42, 43 y 44. Total: 12 variantes y 36 corridas.
- Se reutilizan exclusivamente las tres corridas `latent-10` de la campaña
  anterior, con Python 3.14.4 y NumPy 2.4.4. Antes de copiarlas se comparan
  configuración completa, dataset, validación, hashes, historia, épocas y puntaje
  recalculado del mejor checkpoint. Se conserva procedencia por archivo.
- Presupuesto nuevo: 33 corridas × 3000 épocas = 99.000 épocas. No hay reintentos,
  continuación ni barridos adicionales. Las 36 corridas representan 108.000 épocas.
- Se mantienen BCE, Adam 0,002, tanh interna, salida logística, lote 32,
  validación cada 100 épocas y selección de checkpoint sin parada anticipada.
- Ruido de entrenamiento: bit flips independientes, nivel por carácter y época
  uniforme entre 0, 0,05, 0,10 y 0,20. RNG independiente del modelo, indexado por
  semilla y época absoluta, como la campaña anterior.
- Se reutiliza la colección de validación anterior (semilla 20261110). El puntaje
  promedia por igual el error binario de 0,05/0,10/0,20; desempata MSE, luego época
  más temprana. Variantes: media de sus tres mejores checkpoints, desempate por
  MSE y luego orden del protocolo (simple/current/deeper, dimensiones ascendentes).
  Representante: mejor puntaje individual, desempate por semilla menor. No se
  introduce un margen de empate arbitrario después de ver los resultados.
- La selección se congela antes de generar la evaluación nueva, semilla 20261117,
  con niveles 0/0,05/0,10/0,20/0,30 y 100 realizaciones por carácter/nivel positivo.
  Cero tiene una sola realización. El nivel 0,30 queda fuera del rango entrenado.
  La evaluación no se usa para elegir arquitectura, dimensión, semilla o época.
- Se preservan todos los archivos de `results/denoising/` por SHA256. La campaña
  nueva se guarda en `results/denoising-architectures-2026-11-17/` sin cambiar la
  selección anterior ni la de 1.a.

## Interpretación prevista

La comparación estudia estructuras completas: variar profundidad cambia también
los parámetros. `current` y `deeper` mantienen capas de 10 alrededor del código;
con `d=15/20` se amplía la capa central pero se conserva esa compresión previa.
No se interpreta que todas las familias tengan el mismo cuello de botella efectivo.
Tres semillas muestran variación de inicialización y entrenamiento, sin asegurar
superioridad universal. La evaluación usa ruido nuevo sobre los mismos 32
caracteres; no estima recuperación de caracteres desconocidos.

El fundamento académico del DAE es el par entrada contaminada/objetivo limpio
(OFF-022, p. 22; `wiki/topics/autoencoders.md` del curso). Las arquitecturas,
dimensiones y presupuesto son decisiones experimentales del grupo.

## Reproducción

Desde `tp5/`, con los artefactos originales disponibles y el entorno indicado:

```powershell
python -m src.denoise_architecture_campaign run --config configs/denoising-architectures.json --output results/denoising-architectures-reproduction
```

La salida debe ser nueva. El comando realiza preflight, copia de referencia,
entrenamiento, selección, tablas de validación y evaluación independiente.
Consume un presupuesto nuevo de 99.000 épocas; no se ejecuta al evaluar:

```powershell
python -m src.denoise_architecture_campaign evaluate --campaign results/denoising-architectures-2026-11-17 --output results/denoising-architectures-evaluation-reloaded
```

Cada evaluación requiere una salida nueva. Se guardan modelos de todas las
corridas y tablas finales de todas las variantes; los arrays de predicción se
conservan para la ganadora, la referencia `current-d10` y el AE congelado de 1.a.

Para verificar los checkpoints y la preservación de la campaña anterior, y
generar el informe y los gráficos desde la evidencia ya guardada:

```powershell
python -m src.denoise_architecture_report --campaign results/denoising-architectures-2026-11-17
```

Este comando no entrena. Las figuras de validación muestran medias y rango
entre semillas; los ejemplos de recuperación siguen una política de caracteres
y realizaciones fijada antes de inspeccionar la evaluación nueva.
