# Continuación de la ganadora de denoising — 2026-11-17

El usuario autorizó entrenar la mejor arquitectura con más épocas, proponiendo
6000 o más. Se fija antes de ejecutar el primer aumento a **6000 épocas totales**,
en las mismas tres semillas 42, 43 y 44. Arquitectura seleccionada por la campaña
anterior: `35→40→20→20→20→40→35` (`simple-d20`).

- Continuar desde `last_model.npz` de época 3000, no desde los mejores checkpoints
  que pueden ser anteriores. Conservar pesos, momentos y contador de Adam, RNG e
  historia. El ruido se indexa por época absoluta 3001–6000.
- Presupuesto: **3 continuaciones × 3000 épocas adicionales = 9000 épocas nuevas**.
  No hay reintentos automáticos ni ampliación posterior del presupuesto.
- Mantener BCE, Adam 0,002 constante, lote 32, tanh interna y salida logística.
  Mismos niveles de ruido, colección de validación y evaluaciones cada 100 épocas.
- Mantener el mejor checkpoint previo como candidato y elegir el mejor entre las
  épocas 0–6000. Puntaje: error medio por igual de 0,05/0,10/0,20; desempates por
  MSE y luego época más temprana. Representante: mejor puntaje, luego semilla menor.
- Congelar los checkpoints elegidos antes de generar una evaluación con semilla
  nueva **20261118**, niveles 0/0,05/0,10/0,20/0,30 y 100 realizaciones por carácter
  y nivel positivo. Comparar ambos presupuestos sobre esas mismas entradas.
- Guardar la campaña en `results/denoising-6000-2026-11-17/`, sin modificar los
  resultados anteriores. Verificar hashes de la campaña anterior y que la
  continuación corta coincida exactamente con entrenamiento ininterrumpido.

El fundamento del entrenamiento entrada contaminada/objetivo limpio es OFF-022,
p. 22. El aumento de épocas es una decisión experimental, no una exigencia de
PA-004. Se estudian perturbaciones nuevas de caracteres conocidos, no letras nuevas.

Desde `tp5/`:

```powershell
python -m src.denoise_continuation run --source results/denoising-architectures-2026-11-17 --output results/denoising-6000-2026-11-17
```

La salida debe ser nueva. Repetir el comando en otra salida consume otras 9000
épocas; el informe incluye un comando de evaluación que no entrena.

## Resultado

Continuación completada: 9000 épocas nuevas en 10,50 minutos. Las tres semillas
mejoraron su mejor puntaje; el promedio de validación bajó de 1,01354 a 0,87313
(13,85 %). La representante es la semilla 44, checkpoint de época 5100.
Los checkpoints y la evaluación independiente están en el
[informe](results/denoising-6000-2026-11-17/report.md).
