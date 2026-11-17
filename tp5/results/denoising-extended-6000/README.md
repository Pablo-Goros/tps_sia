# Extensión exploratoria de 1.b: de 3000 a 6000 épocas

Análisis adicional, **fuera del protocolo aprobado**. Los resultados principales siguen
siendo los de `../denoising/` (3000 épocas). Se discute en la sección 6 de
[../denoising/report.md](../denoising/report.md).

## Qué contiene

| Archivo | Contenido |
|---|---|
| `PROTOCOL.md` | Regla fijada **antes** de ejecutar por primera vez |
| `runs/<corrida>/` | `best_model.npz` (mejor checkpoint en 0–6000), `last_model.npz` (época 6000) y `history.json` (validación cada 100 épocas, incluye las 3000 originales) |
| `selection-6000.json` | Mejor checkpoint por corrida y selección (misma regla que el protocolo original) |
| `by-level-6000.csv` | Evaluación sobre la colección final, una sola vez, después de fijar la selección |
| `comparison-3000-vs-6000.csv` | Media, mínimo y máximo entre semillas por variante, nivel y métrica, a 3000 y 6000 épocas |
| `figures/` | Dos gráficos representativos (PNG y SVG) |
| `campaign-hashes-before.json`, `campaign-hashes-after.json` | SHA256 de los 80 archivos de `../denoising/` antes y después de la corrida: idénticos |
| `extension.json`, `run.log` | Metadatos y registro de la corrida |

## Cómo se generó y verificó

```bash
python -m src.denoise_extension run --campaign results/denoising --output results/denoising-extended-nuevo
python -m src.denoise_extension figures --extension results/denoising-extended-6000
```

- Los resultados de esta carpeta salieron de una primera ejecución con un script local
  equivalente. Después se reprodujo la corrida completa con `src/denoise_extension.py`
  en un directorio temporal: pesos, estados del optimizador, historias de validación,
  selección, evaluación y comparación coinciden exactamente. Lo único que difiere es el
  tiempo de reloj que el MLP guarda en su historia (`tiempo_segundos`).
- Desde esa primera ejecución, el único archivo de `../denoising/` que cambió fue
  `report.md` (documentación).
- Entorno: Python 3.11.9 y NumPy 2.3.5. La campaña original usó Python 3.14.4 y
  NumPy 2.4.4: el punto de partida es el mismo (los estados publicados), pero la
  numérica de las épocas nuevas podría diferir levemente en otro entorno.
