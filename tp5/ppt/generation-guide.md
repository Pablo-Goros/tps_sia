# Generación: objetivo y resultado

Cuatro comparaciones independientes, con el mismo azul de la reconstrucción del
dataset. Cada gráfico muestra el objetivo a la izquierda y la salida del decoder
a la derecha. Disponibles en PNG de 2400 × 1440 y SVG.

| Objetivo | PNG para la PPT | SVG | Píxeles distintos del objetivo | Distancia mínima a los 32 originales |
| --- | --- | --- | ---: | ---: |
| ñ | [Gráfico](assets/09-generation-enie.png) | [Vectorial](assets/09-generation-enie.svg) | 4/35 | 1 |
| 3 | [Gráfico](assets/10-generation-three.png) | [Vectorial](assets/10-generation-three.svg) | 5/35 | 1 |
| D | [Gráfico](assets/11-generation-d.png) | [Vectorial](assets/11-generation-d.svg) | 5/35 | 5 |
| ! | [Gráfico](assets/12-generation-exclamation.png) | [Vectorial](assets/12-generation-exclamation.svg) | 1/35 | 1 |

Los cuatro objetivos son dibujos binarios nuevos de 7 × 5, definidos en
`generate_targets.py`. Ninguno pertenece a Font3. Las cuatro salidas generadas
también son distintas de todos sus patrones. La distancia al objetivo mide
fidelidad; la distancia mínima al dataset comprueba novedad. Son medidas
diferentes: una salida nueva puede parecerse poco a lo buscado.

## Cómo se obtuvieron

Se cargó el mismo modelo final de la reconstrucción, semilla 49, sin entrenar ni
modificar sus pesos. Para cada objetivo se buscaron coordenadas latentes cuyo
decoder produjera una aproximación: grilla de 301 × 301 dentro del rectángulo
ocupado por los códigos originales, seguida de tres refinamientos locales de
41 × 41. Se binarizó con `>= 0.5` y se descartaron las copias exactas de Font3.
Se priorizó menor cantidad de píxeles distintos del objetivo; los empates se
resolvieron por menor error cuadrático sobre la salida continua.

El objetivo guía la búsqueda de coordenadas; no se pasa por el encoder ni se
incorpora al entrenamiento. Las figuras presentan las mejores salidas nuevas
halladas con esta búsqueda finita, no un óptimo global. No se consiguió ninguno
de los cuatro dibujos exactamente. Tampoco se asignó una interpretación de letra
o emoji a las salidas deformadas.

Para decir en la presentación:

> Buscamos cuatro formas que no estaban en el dataset: ñ, 3, D y !.
> Mantuvimos fijo el decoder y recorrimos el espacio latente para aproximarlas.
> Obtuvimos patrones nuevos, pero no reprodujimos exactamente los objetivos.
> Reconstruir los caracteres aprendidos no garantiza generar cualquier símbolo.

El fundamento de explorar el latente mediante el decoder está en OFF-022,
pp. 28–38, y en la sección «Variantes y usos» de la wiki de autoencoders. Las
cifras de esta tabla proceden de este experimento, no de las fuentes docentes.

## Reproducción y evidencia

Desde `tp5/`, con las dependencias de TP5 y el Python del entorno virtual:

```bash
python ppt/generate_targets.py
```

Se requieren `results/final/model.npz` y `data/font.h`. El script verifica que
modelo y dataset conserven sus SHA256, regenera cada salida a partir de las
coordenadas seleccionadas y comprueba las dos distancias. La evidencia está en
[generation-provenance.json](assets/generation-provenance.json): dibujos objetivo,
coordenadas completas, salidas continuas y binarias, métricas, protocolo y hashes.
