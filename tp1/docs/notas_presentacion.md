# Notas para la presentación — TP1 Sokoban

Estas notas organizan una exposición de aproximadamente 12 a 15 minutos. No
son una presentación terminada: el archivo PowerPoint debe armarse fuera del
repositorio usando las cuatro figuras reproducibles como material visual.

## 1. Problema, objetivo y alcance

- Presentar Sokoban como un problema de planificación en un espacio de estados.
- Objetivo del trabajo: comparar BFS, DFS, Greedy y A* cuando cada movimiento
  del jugador tiene costo 1.
- Aclarar que mover sin empujar también cuenta; minimizar costo equivale a
  minimizar la longitud completa del camino.
- Alcance deliberado: interfaz de consola y tableros ASCII, sin GUI, GIF ni
  video. Los niveles grandes quedan como extras; el experimento usa cuatro
  niveles entendidos y representativos.

Mensaje oral sugerido: “No buscamos sólo empujes mínimos; buscamos movimientos
mínimos del jugador bajo las reglas completas de Sokoban”.

## 2. Representación y reglas del dominio

- Separación estática/dinámica:
  - `Mapa`: pisos explícitos, paredes y objetivos.
  - `Estado`: posición del jugador y `frozenset` de cajas.
- La inmutabilidad permite comparar estados por valor y guardarlos en conjuntos
  de visitados.
- Coordenadas cartesianas `(x, y)` y orden determinista `U,D,L,R`.
- Cada arista representa un movimiento legal de costo 1.
- El parser conserva filas desiguales: una celda ausente no se transforma en
  piso por completar un rectángulo.
- La solución se reconstruye mediante enlaces al nodo padre y después se puede
  reproducir usando las mismas reglas que generaron los sucesores.

Visual sugerido: un nivel pequeño, señalando mapa estático, jugador, cajas y
objetivos, junto con un ejemplo de caminata y otro de empuje.

## 3. Algoritmos de búsqueda y métricas

| Algoritmo | Frontera/prioridad | Tratamiento de repetidos | Garantía |
|---|---|---|---|
| BFS | FIFO | Visitado al encolar. | Óptimo. |
| DFS | LIFO | Visitado al apilar; sucesores invertidos para respetar `U,D,L,R`. | No óptimo. |
| Greedy | Menor `h` y desempate estable. | Visitado al insertar. | No óptimo. |
| A* | Menor `g+h` y desempate estable. | Mejor `g`, reapertura y descarte de entradas obsoletas. | Óptimo con `h` admisible. |

Definiciones que conviene enfatizar:

- El nodo objetivo se reconoce antes de expandirlo.
- El límite se controla antes de generar sucesores; nunca se supera.
- `nodos_expandidos` aumenta sólo cuando se generan sucesores.
- `nodos_frontera` cuenta entradas restantes al terminar.
- `max_nodos_frontera` es el máximo histórico desde la frontera inicial.
- `exito`, `fracaso` y `corte` son estados distintos.

## 4. Heurísticas y poda de deadlocks

- Ambas heurísticas resuelven asignación uno a uno caja–objetivo mediante
  programación dinámica con bitmask.
- `manhattan` suma el matching de menor costo con distancia Manhattan.
- `empujes_inversos` usa distancias calculadas desde cada objetivo mediante
  empujes inversos; respeta paredes y devuelve infinito si el matching completo
  es imposible.
- Son admisibles porque ignoran desplazamientos adicionales del jugador y otras
  restricciones dinámicas: nunca sobreestiman una solución real.
- Cada corrida crea una caché heurística privada.

Poda segura:

- Las tablas de empujes inversos identifican celdas desde las cuales una caja no
  puede llegar a ningún objetivo.
- También se detectan bloques congelados 2×2 de paredes y cajas.
- Sólo se podan empujes demostrablemente irresolubles; caminar nunca se poda por
  estas reglas.

Visual sugerido: contrastar Manhattan con un rodeo impuesto por una pared.

## 5. Metodología experimental y reproducibilidad

- Fuente única: `configuracion.json`.
- Semilla `20260820`, cuatro niveles, cuatro algoritmos, dos heurísticas para
  métodos informados y cinco repeticiones por combinación.
- Total: 120 ejecuciones y 24 grupos de resumen.
- Todas las rutas son relativas al JSON.
- Los niveles y sus tablas estáticas se preparan antes de medir.
- Una instancia local de `random.Random` baraja únicamente el orden de corridas.
- Cada repetición reinicia frontera, visitados, mejores costos y caché
  heurística.
- El CSV crudo contiene una fila por intento; el resumen aplica las mismas
  reglas estadísticas a todos los grupos.

Semántica del resumen:

- Tasa de éxito: todos los intentos.
- Costo promedio: sólo éxitos; queda vacío si no hay éxitos.
- Promedios de nodos, frontera y tiempo: todos los intentos.
- Desvío temporal: las cinco repeticiones.
- Fracasos y cortes: columnas separadas.

## 6. Resultados, figuras y estados no exitosos

Resultados estables principales:

| Nivel | BFS | DFS | Greedy | A* |
|---|---:|---:|---:|---:|
| Trivial | costo 6 / 14 exp. | 6 / 9 | 6 / 14 | 6 / 14 |
| Fácil | 13 / 927 | 447 / 1147 | 13 / 35 | 13 / 292 |
| Medio | 18 / 4703 | 228 / 1274 | 18 / 56 | 18 / 1552 |

Lectura:

- BFS y A* coinciden en el óptimo.
- Greedy encuentra el mismo costo en estos niveles con muchas menos
  expansiones, pero no ofrece garantía general.
- DFS devuelve caminos muy largos en Fácil y Medio.
- Las dos heurísticas coinciden en costo y expansiones en las instancias
  resolubles; un nivel con rodeos más decisivos permitiría separarlas mejor.

Tabla obligatoria de estados no exitosos:

| Nivel | Configuración | Éxitos | Fracasos | Cortes |
|---|---|---:|---:|---:|
| Sin solución | BFS | 0 | 5 | 0 |
| Sin solución | DFS | 0 | 5 | 0 |
| Sin solución | Greedy + Manhattan | 0 | 5 | 0 |
| Sin solución | Greedy + Empujes inversos | 0 | 5 | 0 |
| Sin solución | A* + Manhattan | 0 | 5 | 0 |
| Sin solución | A* + Empujes inversos | 0 | 5 | 0 |

Orden sugerido de figuras: `costo_solucion.png`, `nodos_expandidos.png`,
`max_nodos_frontera.png` y `tiempo_promedio.png`. En las dos figuras de nodos,
explicar que la escala pasa a logarítmica con una razón máximo/mínimo de al
menos 100. En tiempo, señalar las barras de desvío estándar.

## 7. Conclusiones, limitaciones y cierre

- La estructura de frontera cambia el comportamiento aun con reglas y estados
  compartidos.
- A* ofrece el mejor compromiso teórico: conserva optimalidad y reduce el
  trabajo frente a BFS, aunque en estos niveles Greedy expande todavía menos.
- La poda segura reduce estados sin cambiar soluciones óptimas.
- El nivel sin solución comprueba que agotar el espacio es distinto de cortar
  por un límite.
- Los tiempos dependen del hardware, sistema operativo, versión de Python,
  carga y cachés. Comparar costos y nodos es más estable; para tiempo se usan
  media y desvío de cinco repeticiones.
- Limitaciones: búsqueda por movimientos, deadlocks conservadores y conjunto de
  niveles pequeño. Posibles extensiones: búsqueda por empujes, macroacciones y
  deadlocks más sofisticados.

Cierre/demostración sugerida:

1. Ejecutar `python -m sokoban resolver` sobre el nivel trivial con
   `--mostrar-estados`.
2. Mostrar `configuracion.json` y los dos CSV.
3. Ejecutar `python -m sokoban graficar`.
4. Cerrar con la diferencia entre “rápido en estas instancias” y “óptimo con
   garantía”.

El PowerPoint final, sus transiciones y cualquier material manual se mantienen
fuera del repositorio. Aquí quedan sólo código, datos, figuras y notas que se
pueden regenerar.
