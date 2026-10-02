## Diagnóstico general

El proyecto tiene un motor de Algoritmos Genéticos sustancialmente implementado y cubre casi todos los requisitos algorítmicos obligatorios del Ejercicio 2. Sin embargo, hoy lo clasificaría como un **prototipo funcional avanzado, pero todavía no listo para entregar**.

El código puede ejecutar el ciclo completo si se le proporciona una imagen válida, pero la configuración incluida no funciona porque falta `ejemplos/japon.png`. La suite arroja **26 pruebas aprobadas y 3 fallidas**.

No modifiqué archivos del proyecto; el árbol Git quedó limpio en la rama `TP2`.

## Cómo funciona el algoritmo

### 1. Representación

Cada individuo contiene un cromosoma ordenado de `T` triángulos:

- Seis coordenadas normalizadas para los tres vértices.
- Color RGB.
- Transparencia alfa.
- El orden del cromosoma determina el orden de superposición.

Los valores iniciales se generan aleatoriamente. Las coordenadas están en `[0,1]`, RGB en `[0,255]` y alfa inicialmente en `[30,220]`.

Implementación: [triangle.py](C:/Users/Mateo/Documents/Projects/tps_sia/tp2/src/models/triangle.py:8), [chromosome.py](C:/Users/Mateo/Documents/Projects/tps_sia/tp2/src/models/chromosome.py:8), [individual.py](C:/Users/Mateo/Documents/Projects/tps_sia/tp2/src/models/individual.py:9).

### 2. Renderizado

Para evaluar un individuo:

1. Se crea un canvas RGBA con el color configurado.
2. Cada triángulo se dibuja en una capa transparente independiente.
3. Las capas se combinan mediante composición alfa.
4. El resultado se convierte a RGB para compararlo con el objetivo.

Se evalúa normalmente a `64×64`, aunque la imagen final se renderiza en la resolución original.

Implementación: [renderer.py](C:/Users/Mateo/Documents/Projects/tps_sia/tp2/src/engine/renderer.py:8).

### 3. Aptitud

La imagen candidata se compara con el objetivo mediante MSE sobre los tres canales RGB:

```text
fitness = 1 / (1 + MSE / 255²)
```

Se maximiza el fitness y se minimiza el MSE. La relación es monótona y consistente.

Hay una consideración importante: como el MSE RGB posible está entre `0` y `65025`, el fitness real queda entre aproximadamente `0.5` y `1`, no entre `0` y `1`. Esto comprime bastante las diferencias entre individuos y puede generar **muy poca presión selectiva en ruleta y universal**.

Además, el MSE reportado corresponde a la resolución de evaluación, no necesariamente al error de la imagen final a resolución completa.

Implementación: [fitness.py](C:/Users/Mateo/Documents/Projects/tps_sia/tp2/src/engine/fitness.py:9).

### 4. Ciclo evolutivo

El flujo de cada corrida es:

```text
Población aleatoria N
        ↓
Evaluación inicial
        ↓
Selección de K padres
        ↓
Cruza por parejas
        ↓
Mutación de los K hijos
        ↓
Evaluación de hijos
        ↓
Supervivencia → nueva población N
        ↓
Métricas, snapshot y condiciones de parada
```

Se conserva externamente una copia del mejor individuo histórico, incluso si una estrategia de supervivencia lo elimina de la población.

Implementación: [genetic_algorithm.py](C:/Users/Mateo/Documents/Projects/tps_sia/tp2/src/engine/genetic_algorithm.py:17).

## Cobertura de requisitos

| Requisito | Estado | Observación |
|---|---:|---|
| Individuo como imagen completa | Implementado | Cromosoma ordenado de triángulos |
| Geometría, RGB y alfa | Implementado | Coordenadas normalizadas |
| Orden de capas | Implementado | El orden del cromosoma es significativo |
| Elite | Implementado | Selección determinista por fitness |
| Ruleta | Implementado | Muestreo con reemplazo |
| Universal/SUS | Implementado | Punteros equidistantes |
| Ranking | Implementado | Pseudo-fitness lineal por posición |
| Boltzmann | Implementado | Temperatura exponencial decreciente |
| Torneo determinístico | Implementado | Mejor entre `M` candidatos |
| Torneo probabilístico | Implementado | Mejor con probabilidad configurable |
| Supervivencia aditiva | Implementado | Selección sobre padres + hijos |
| Supervivencia exclusiva | Implementado | Hijos y, si faltan, padres |
| Cruce de dos puntos | Implementado | Intercambio de segmento |
| Cruce uniforme | Implementado | Decisión independiente por triángulo |
| Mutación de gen | Implementado | Un triángulo |
| Multigen limitada | Implementado | Entre 1 y `M` triángulos |
| Multigen uniforme | Implementado | Probabilidad independiente por triángulo |
| No uniforme | Implementado | Magnitud decreciente |
| Máximo de generaciones | Implementado | Tiene un posible desfase de conteo |
| Timeout | Implementado | Se controla entre generaciones |
| Umbral de fitness/MSE | Implementado | Configurable por JSON |
| Estancamiento | Implementado | Ventana sobre mejor fitness |
| Estructura/diversidad | Parcial | Usa varianza de fitness |
| Imagen final | Implementado | `mejor_imagen.png` |
| Exportación de triángulos | Implementado parcialmente | JSON no autosuficiente |
| Métricas CSV | Implementado | Fitness, MSE, diversidad y tiempo |
| Reproducibilidad por semilla | No implementado | No se expone ni registra semilla |
| Presentación académica | No encontrada | Sólo está localmente el enunciado |
| Ejemplo ejecutable incluido | No disponible | Falta `ejemplos/japon.png` |

## Operadores implementados

Los siete métodos de selección requeridos están disponibles mediante una factory en [selection.py](C:/Users/Mateo/Documents/Projects/tps_sia/tp2/src/operators/selection.py:295).

Las cruzas son:

- Dos puntos: intercambia triángulos completos entre dos cortes.
- Uniforme: decide por locus de qué padre proviene cada triángulo.

Ambas mantienen el número de triángulos y producen copias profundas. En dos puntos se permiten cortes iguales o en los extremos, por lo que algunas operaciones seleccionadas como “cruza” pueden no intercambiar nada.

Implementación: [crossover.py](C:/Users/Mateo/Documents/Projects/tps_sia/tp2/src/operators/crossover.py:23).

Las cuatro mutaciones operan a nivel de triángulo. Cuando se elige un triángulo:

- 80% de las veces se perturban sus diez atributos con ruido gaussiano.
- 20% se regenera completamente.
- La mutación no uniforme disminuye progresivamente la magnitud y la probabilidad de regeneración.

Los límites se preservan por clipping. Podrían aparecer triángulos degenerados por coordenadas coincidentes después del clipping, pero siguen siendo dibujables.

Implementación: [mutation.py](C:/Users/Mateo/Documents/Projects/tps_sia/tp2/src/operators/mutation.py:22).

Las supervivencias aditiva, exclusiva y una extensión de brecha generacional están en [survival.py](C:/Users/Mateo/Documents/Projects/tps_sia/tp2/src/operators/survival.py:25).

## Estado ejecutable y pruebas

Resultado actual de `pytest -q`:

```text
26 passed
3 failed
```

Las fallas son:

1. `test_fitness_evaluator`: falta `tp2/ejemplos/japon.png`.
2. `test_full_pipeline_run`: la misma imagen ausente.
3. `test_plot_metrics`: `matplotlib` no está instalado.

El tercer caso revela una inconsistencia de dependencias: `matplotlib` está en el extra `analysis`, pero las pruebas lo necesitan incondicionalmente. Instalar solamente las dependencias principales y el extra `dev` no alcanza para ejecutar toda la suite.

Aun así, ejecuté una corrida pequeña con una imagen temporal y el motor produjo correctamente:

- Imagen final.
- JSON de triángulos.
- CSV de métricas.
- Cantidad correcta de triángulos.

Por lo tanto, el pipeline interno funciona; lo que falla es el escenario reproducible incluido en el repositorio.

## Problemas concretos encontrados

### Prioridad alta

- **No hay semilla reproducible.** Se usa el módulo global `random`, pero no existe `seed` en configuración, CLI, resultados ni métricas.
- **Falta la imagen de ejemplo.** La configuración predeterminada y dos pruebas dependen de ella.
- **El empaquetado es inconsistente.** [pyproject.toml](C:/Users/Mateo/Documents/Projects/tps_sia/tp2/pyproject.toml:27) busca paquetes `tp2*` dentro de `tp2`, mientras [setup.py](C:/Users/Mateo/Documents/Projects/tps_sia/tp2/setup.py:6) trata al directorio padre como raíz. La metadata generada quedó sin paquete de nivel superior, por lo que el comando instalado `image-ga` puede quedar apuntando a un módulo no empaquetado.
- **La exportación no permite reconstrucción totalmente autónoma.** El JSON conserva triángulos y orden, pero no registra dimensiones, color de fondo, configuración, semilla ni resolución de evaluación.
- **No existe una presentación de entrega.** El PDF encontrado es el enunciado y además `docs/` está ignorado por Git.

### Prioridad media

- El fitness comprimido en `[0.5,1]` hace débil a la selección proporcional.
- La “diversidad genética” es solamente varianza de fitness. Dos poblaciones genéticamente muy diferentes pueden tener diversidad cero si todos obtienen el mismo fitness.
- Hay un desfase semántico en generaciones: con máximo `3`, se registran generaciones `0,1,2` y se reporta `generations_completed=3`. Nunca se evoluciona ni registra la generación numerada `3`.
- El timeout no interrumpe una evaluación en curso; puede excederse por lo que dure una generación completa.
- No se validan sistemáticamente `population_size`, `num_offspring`, `pc`, `pm`, resoluciones, canales del fondo ni ventanas de parada.
- El torneo probabilístico falla con una población de un solo individuo por intentar desempaquetar dos candidatos desde una muestra de tamaño uno.
- `max_genes_to_mutate > cantidad de triángulos` introduce una distribución sesgada en la mutación multigen limitada.
- Las métricas CSV registran el mejor de la población actual, mientras que consola y salida final usan el mejor histórico. La curva denominada “mejor fitness” podría descender con supervivencias no elitistas.
- El tope de 200 triángulos se describe como configurable en el README, pero es una constante del código.

## Calidad de las pruebas

Las pruebas comprueban presencia y funcionamiento básico de todos los operadores, pero son superficiales en aspectos importantes:

- Selecciones: verifican principalmente cantidad de resultados, no distribuciones.
- Cruzas: verifican longitud, no procedencia correcta de los genes.
- Mutaciones: no comprueban todos los rangos ni la reducción no uniforme.
- No hay pruebas de transparencia u orden de capas.
- No hay pruebas de carga/guardado completo de configuración.
- No se prueba reconstrucción desde el JSON.
- No se prueban varios criterios de parada.
- No se prueban entradas inválidas.
- Las pruebas estocásticas tampoco fijan semilla.

## Documentación

El [README.md](C:/Users/Mateo/Documents/Projects/tps_sia/tp2/README.md:1) explica bien:

- Representación.
- MSE y fitness.
- Lista de operadores.
- Configuración básica.
- Archivos de salida.
- Ejecución desde CLI.

Le falta desarrollar mejor:

- Cuándo conviene cada tipo de cruza.
- Cuándo conviene cada mutación.
- Consecuencias de la presión selectiva.
- Complejidad y resultados experimentales.
- Configuración mínima verificada.
- Semilla y reproducibilidad.
- Limitaciones de optimizar a baja resolución.
- Un ejemplo que realmente exista.

## Rendimiento esperado

La evaluación domina el costo. Aproximadamente:

```text
Inicialización: O(N × T × Wₑ × Hₑ)
Cada generación: O(K × T × Wₑ × Hₑ)
Snapshot: O(T × Wₒ × Hₒ)
```

Donde:

- `N`: población.
- `K`: descendencia.
- `T`: triángulos.
- `Wₑ × Hₑ`: resolución de evaluación.
- `Wₒ × Hₒ`: resolución original.

El renderer crea una capa completa por triángulo, por lo que aumentar triángulos o resolución tiene impacto prácticamente lineal y considerable.

## Veredicto

La base conceptual y el motor están bien encaminados: **todos los operadores obligatorios están presentes, son implementación propia y el pipeline central funciona**. La brecha principal está en reproducibilidad, empaquetado, assets, validación, calidad de métricas y preparación de la entrega.

Antes de experimentar seriamente o preparar la defensa, priorizaría:

1. Arreglar imagen de ejemplo, empaquetado y dependencias.
2. Incorporar y registrar una semilla.
3. Corregir conteo de generaciones y casos límite.
4. Hacer autosuficiente la exportación.
5. Mejorar la métrica de diversidad y estudiar el escalado del fitness.
6. Ampliar pruebas.
7. Agregar experimentos reproducibles y la presentación.