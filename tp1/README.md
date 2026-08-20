# TP1 - Métodos de Búsqueda: Sokoban

Motor de búsqueda de soluciones para **Sokoban** (Ejercicio 2 del TP1 de SIA).
Implementa BFS, DFS, IDDFS, Greedy y A\*, con tres heurísticas admisibles, e
informa al terminar: resultado, costo, nodos expandidos, nodos frontera,
solución y tiempo de procesamiento.

Cada movimiento del jugador cuesta 1 (empuje una caja o no), así que **optimizar
el costo es optimizar la cantidad de movimientos**, que es lo que pide el
enunciado.

---

## Requisitos

- **Python 3.9 o superior**. El motor usa solo la biblioteca estándar.
- Dependencias **opcionales**, únicamente para las salidas visuales
  (más `ffmpeg` en el PATH si se quieren videos MP4):

```bash
pip install -r requirements.txt   # Pillow (GIF), matplotlib (gráficos), pygame (ventana)
```

Sin ellas todo funciona igual: la visualización por consola no requiere nada.

---

## Cómo ejecutar

```bash
cd tp1
python3 main.py --nivel niveles/nivel_03_medio.txt                  # A* por defecto
python3 main.py --nivel niveles/nivel_03_medio.txt --algoritmo bfs
python3 main.py --nivel niveles/nivel_04_dificil.txt --algoritmo todos
python3 main.py --nivel niveles/nivel_02_facil.txt --algoritmo astar --heuristica manhattan
python3 main.py --listar                                            # algoritmos y heurísticas
```

### Opciones

| Opción | Descripción |
|---|---|
| `--nivel RUTA` | archivo del nivel en formato XSB (ver abajo) |
| `--algoritmo {bfs,dfs,iddfs,greedy,astar,todos}` | método de búsqueda (default `astar`) |
| `--heuristica NOMBRE` | heurística para Greedy y A\* (default `matching`) |
| `--sin-poda` | desactiva la detección de deadlocks (útil para medir su impacto) |
| `--max-nodos N` / `--timeout S` | cortes de seguridad para corridas que explotan |
| `--pasos` / `--pausa S` | imprime el tablero paso a paso por consola |
| `--gif RUTA` / `--ms-por-paso MS` | genera un GIF animado de la solución (Pillow) |
| `--animar` | reproduce la solución en una ventana (pygame) |
| `--json RUTA` / `--csv RUTA` | guarda las métricas |
| `--listar` | lista algoritmos y heurísticas disponibles |

### Salida

```
$ python3 main.py --nivel niveles/nivel_04_dificil.txt --algoritmo astar

Nivel: niveles/nivel_04_dificil.txt
##########
#        #
#  $  $  #
#  .##.  #
#  @     #
#  .##.  #
#  $  $  #
#        #
##########

Cajas: 4 | Objetivos: 4 | Poda de deadlocks: si

Algoritmo:              A*  (heuristica: matching)
Resultado:              EXITO
Costo de la solucion:   24 movimientos
Nodos expandidos:       9354
Nodos frontera:         4843
Nodos generados:        17395
Tiempo de procesamiento: 0.0927 s
Solucion (U/D/L/R):     ULUURDURRRDRDDDDDLUDLLLU
Solucion:               arriba, izquierda, arriba, arriba, derecha, abajo, ...
```

### Visualizador (GIF paso a paso)

`visualizar.py` resuelve el nivel y escribe un GIF (o un video MP4) con **un
frame por movimiento**: barra de progreso, número de paso, acción aplicada,
contador de cajas en objetivo y la caja recién empujada resaltada en amarillo.

```bash
python3 visualizar.py --nivel niveles/nivel_03_medio.txt                 # A*, GIF
python3 visualizar.py --nivel niveles/nivel_03_medio.txt --formato mp4   # video
python3 visualizar.py --nivel niveles/nivel_04_dificil.txt --algoritmo todos
python3 visualizar.py --nivel niveles/nivel_02_facil.txt --ms 200 --escala 56
python3 visualizar.py --nivel niveles/nivel_01_trivial.txt --frames salidas/frames
```

Sin `--salida` el GIF va a `salidas/<nivel>_<algoritmo>.gif`; con
`--algoritmo todos` genera uno por método (y avisa cuáles no llegaron a
resolver el nivel), que es la forma más directa de **ver** la diferencia entre
la solución óptima de A\* y el paseo de 56 338 movimientos de DFS.

| Opción | Descripción |
|---|---|
| `--algoritmo` / `--heuristica` | qué resolver (igual que en `main.py`); `todos` = una salida por método |
| `--formato {gif,mp4}` | formato de salida (default `gif`); `mp4` requiere ffmpeg |
| `--salida RUTA` | archivo de salida —la extensión define el formato— o carpeta destino con `--algoritmo todos` |
| `--frames CARPETA` | además, un PNG por paso (para poner pasos sueltos en la presentación) |
| `--ms N` | milisegundos por frame del GIF (default 300) |
| `--fps N` | cuadros por segundo del video (default 12, o 60 si el camino supera 600 pasos) |
| `--escala N` | lado de cada celda en píxeles (default 44) |
| `--submuestreo N` | dibuja 1 de cada N pasos |
| `--max-frames N` | tope de frames: default 400 en GIF, **sin tope en video** (`0` = sin tope) |
| `--sin-encabezado` | solo el tablero, sin barra ni contadores |
| `--timeout S` / `--max-nodos N` | cortes de la búsqueda |

**GIF vs video.** DFS puede devolver caminos de decenas de miles de movimientos;
un GIF con un frame por cada uno sería inmanejable, así que por defecto se topea
en 400 frames y se dibuja 1 de cada N pasos (lo avisa por pantalla). El video no
tiene ese tope: los frames se generan de a uno y se escriben directo al stdin de
ffmpeg, con memoria constante, así se puede ver el camino completo **sin
saltos**:

```bash
# los 56 338 movimientos de DFS, uno por uno, en un video de ~15 min
python3 visualizar.py --nivel niveles/nivel_04_dificil.txt --algoritmo dfs --formato mp4

# lo mismo pero al doble de velocidad
python3 visualizar.py --nivel niveles/nivel_04_dificil.txt --algoritmo dfs \
    --formato mp4 --fps 120
```

`main.py --gif` sigue funcionando para el caso simple (usa el mismo renderer y,
con `--algoritmo todos`, dibuja la mejor solución encontrada).

### Comparativa de todos los métodos

```bash
python3 benchmark.py                       # todos los niveles x todos los métodos
python3 benchmark.py --graficos            # + PNGs comparativos (matplotlib)
python3 benchmark.py --niveles 'niveles/nivel_0[1-4]*.txt' --repeticiones 5
python3 benchmark.py --algoritmos astar --heuristicas manhattan matching
```

Genera `salidas/benchmark.csv`, `salidas/benchmark.md` y, con `--graficos`,
`salidas/graficos/*.png`.

---

## Formato de los niveles

Formato XSB estándar; las líneas que empiezan con `;` son comentarios.

| Símbolo | Significado |
|---|---|
| `#` | pared |
| ` ` | piso |
| `.` | objetivo |
| `$` | caja |
| `*` | caja sobre objetivo |
| `@` | jugador |
| `+` | jugador sobre objetivo |

Al parsear se valida que haya exactamente un jugador, que la cantidad de cajas
coincida con la de objetivos y que el tablero esté **cerrado** (si el jugador
pudiera escaparse, el espacio de estados sería infinito).

Niveles incluidos:

| Nivel | Cajas | Para qué sirve |
|---|---|---|
| `nivel_00_sin_solucion.txt` | 2 | los objetivos son inalcanzables: la búsqueda debe informar fracaso |
| `nivel_01_trivial.txt` | 1 | verificación de punta a punta |
| `nivel_02_facil.txt` | 2 | todos los métodos terminan en milisegundos |
| `nivel_03_medio.txt` | 3 | ya se nota la diferencia entre informados y desinformados |
| `nivel_04_dificil.txt` | 4 | paredes internas; BFS empieza a sufrir |
| `nivel_05_seis_cajas.txt` | 6 | explosión combinatoria: solo A\* y Greedy terminan |
| `nivel_06_original.txt` | 6 | nivel 1 del Sokoban original; solo Greedy y DFS terminan |

---

## Diseño

Capas independientes: cada una se comunica con la de al lado por una interfaz
mínima y no sabe cómo está implementada. La búsqueda entera ocurre "en
abstracto": nunca dibuja nada.

```
estado.py         Mapa + Estado + reglas del juego (aplicar_accion)
distancias.py     tablas precalculadas del mapa (empujes, celdas muertas)
sucesores.py      generación de sucesores + poda de deadlocks
nodo.py           Nodo (padre, acción, g, h) y reconstrucción del camino
frontera.py       FIFO / LIFO / prioridad: lo único que cambia por algoritmo
heuristicas.py    heurísticas admisibles
busqueda.py       bucle genérico, IDDFS y wrapper de métricas
visualizacion.py  consola, GIF y ventana animada
```

En la raíz: `main.py` (motor + métricas), `visualizar.py` (GIFs paso a paso) y
`benchmark.py` (corridas comparativas).

### Estructura de estado

Lo **estático** se separa de lo **dinámico**: las paredes y los objetivos no
cambian nunca durante la búsqueda, así que no tiene sentido copiarlos una vez por
nodo.

```python
@dataclass(frozen=True)
class Mapa:            # se crea una sola vez, se pasa por referencia
    paredes: frozenset[tuple[int, int]]
    objetivos: frozenset[tuple[int, int]]
    ancho: int
    alto: int

@dataclass(frozen=True)
class Estado:          # esto es lo que se genera por cada nodo
    jugador: tuple[int, int]
    cajas: frozenset[tuple[int, int]]
```

- `frozen=True` + `frozenset` ⇒ `Estado` es **inmutable y hasheable**, que es lo
  que permite el set de visitados con detección de repetidos en O(1) y la
  comparación por valor (dos tableros iguales *son* el mismo estado).
- Coordenadas en vez de una grilla de caracteres: generar sucesores mira 4 celdas
  alrededor del jugador en lugar de recorrer todo el tablero, y hashear un
  `frozenset` de 3-6 posiciones es mucho más barato que hashear una matriz N×M.
- Test de objetivo: `estado.cajas == mapa.objetivos`.

`Nodo` (estado, padre, acción, `g`, `h`) va aparte de `Estado` a propósito: si
`g` o `padre` vivieran adentro del estado, dos nodos con el mismo tablero
alcanzado por caminos distintos dejarían de ser iguales y se rompería la
detección de repetidos.

### Espacio de acciones

Cuatro acciones (`arriba`, `abajo`, `izquierda`, `derecha`), costo 1 cada una.
Un movimiento es válido si el destino no es pared; si en el destino hay una caja,
hay que mirar **dos celdas hacia adelante**: la caja se empuja solo si la
siguiente no es pared ni tiene otra caja. Todo el filtrado ocurre antes de crear
el estado nuevo, así que cualquier sucesor que llega a la frontera ya es jugable.

### Bucle de búsqueda

BFS, DFS, Greedy y A\* son **el mismo bucle**; lo único que cambia es la
estructura de la frontera y si se le pasa o no una heurística:

```python
while not frontera.esta_vacia():
    nodo = frontera.sacar()
    if nodo.estado in visitados:      # pudo entrar por varios caminos
        continue
    visitados.add(nodo.estado)
    if nodo.estado.es_objetivo(mapa):
        return nodo
    for hijo in expandir(nodo, mapa, heuristica):
        if hijo.estado not in visitados:
            frontera.agregar(hijo)
```

| Método | Frontera | ¿Óptimo? |
|---|---|---|
| BFS | FIFO | sí (todos los movimientos cuestan lo mismo) |
| DFS | LIFO | no |
| IDDFS | pila con límite de profundidad creciente | sí |
| Greedy | heap ordenado por `h` | no |
| A\* | heap ordenado por `f = g + h` | sí, con heurística admisible |

DFS **necesita** el set de visitados: Sokoban tiene ciclos (se puede empujar una
caja y devolverla a su lugar), sin eso no termina nunca. IDDFS usa un bucle
propio, con memoria lineal en la profundidad, y por eso es el único que mantiene
la frontera chiquísima a costa de reexpandir los niveles superiores en cada
iteración.

La solución se reconstruye subiendo por los punteros al padre desde el nodo
objetivo hasta la raíz.

### Heurísticas

Todas ignoran (o acotan por debajo) el costo de mover al jugador, que es
justamente lo que las mantiene admisibles.

| Nombre | Admisible | Idea |
|---|---|---|
| `manhattan` | sí | suma de la distancia Manhattan de cada caja a su objetivo más cercano |
| `manhattan_jugador` | sí | `manhattan` + los pasos que el jugador necesita para llegar a una caja |
| `matching` | sí | asignación óptima caja↔objetivo (método húngaro) sobre distancias Manhattan (default) |

**Por qué son admisibles:**

- `manhattan`: cada empuje mueve una sola caja una sola celda, con lo cual reduce
  la suma en a lo sumo 1. Como en el estado final la suma vale 0, hacen falta al
  menos `h` empujes, y cada empuje es un movimiento.
- `manhattan_jugador`: para empujar *cualquier* caja el jugador primero tiene que
  quedar pegado a ella, lo que exige al menos `d − 1` movimientos que **no** son
  empujes, con `d` la distancia a la caja más cercana. Esos movimientos son
  disjuntos de los empujes que cuenta `manhattan`, así que sumar ambos límites
  inferiores sigue dando un límite inferior.
- `matching`: `manhattan` asigna varias cajas al mismo objetivo, que es
  físicamente imposible. Toda solución induce alguna asignación uno a uno, y el
  método húngaro devuelve la más barata de todas: sigue siendo un límite inferior
  y **domina** a `manhattan`.
Vale la cadena de dominancia `manhattan ≤ matching`, y a mayor `h` sin perder
admisibilidad, menos nodos expande A\*. La referencia es BFS, que es A\* con
`h = 0`:

| A\* sobre `nivel_04_dificil` | Costo | Nodos expandidos |
|---|---|---|
| BFS (equivale a `h = 0`) | 24 | 39 939 |
| `manhattan` | 24 | 12 618 |
| `manhattan_jugador` | 24 | 9 237 |
| `matching` | 24 | 9 356 |

El salto grande es el primero. `matching` domina a `manhattan` en teoría, pero en
estos tableros casi no hay paredes entre las cajas y los objetivos, así que la
ventaja práctica es chica; `manhattan_jugador` llega a un número parecido siendo
más barata de calcular. Para que la diferencia se note haría falta un nivel con
pasillos y rodeos.

### Detección de deadlocks

Sokoban tiene estados desde los cuales ya es imposible ganar aunque falten
movimientos por hacer. Se detectan dos familias, y los empujes que llevan a ellos
se podan como si fueran movimientos inválidos:

1. **Deadlock estático (celdas muertas):** celdas desde las cuales ninguna caja
   puede llegar ya a ningún objetivo (esquinas y tramos de pared sin objetivos).
   Salen gratis de la tabla de empujes: si una celda no aparece en ninguna, no
   hay forma de sacar la caja de ahí. Se calcula una sola vez por mapa.
2. **Deadlock por congelamiento:** la caja recién empujada quedó dentro de un
   bloque de 2×2 formado solo por paredes y cajas, con al menos una caja fuera de
   objetivo. Ese bloque ya no se puede mover nunca más.

La poda es **conservadora**: descarta únicamente estados que con certeza no
llevan a solución, así que no se pierde la optimalidad de BFS ni de A\*. Se puede
desactivar con `--sin-poda` para medir su impacto (y comprobar que el costo
óptimo no cambia).

### Visualización

Es una capa posterior e independiente: recibe `Estado` y `Mapa` (nunca `Nodo`,
frontera, ni nada de la búsqueda) y se limita a reproducir lo ya calculado.

```bash
python3 main.py --nivel niveles/nivel_02_facil.txt --pasos            # consola
python3 visualizar.py --nivel niveles/nivel_02_facil.txt              # GIF
python3 main.py --nivel niveles/nivel_02_facil.txt --animar           # pygame
```

Los cuatro backends (consola, GIF, video y ventana) consumen la misma secuencia
de estados, así que se puede tirar uno y agregar otro sin tocar el motor. El
video sale de `iterar_frames`, que es un generador: para un camino de 56 000
movimientos, materializar los frames en una lista ocuparía decenas de GB.

La reproducción reusa `aplicar_accion`, la misma función que usa el generador de
sucesores, de modo que no puede quedar desincronizada con la búsqueda; además
sirve como **validación** del camino devuelto: si alguna acción no fuera
aplicable, la reproducción lanza una excepción en vez de dibujar algo inválido.

Los frames se arman con Pillow y se guardan en modo paleta, que es lo que hace
que un GIF de 400 frames pese ~2 MB en vez de ~5 MB.

---

## Resultados

Corrida completa (`python3 benchmark.py --timeout 12`, Python 3.9, macOS).
`timeout` = se cortó a los 12 s sin encontrar solución.

| Nivel | Método | Costo | Expandidos | Frontera | Tiempo (s) |
|---|---|---|---|---|---|
| `01_trivial` (1 caja) | BFS | 6 | 15 | 0 | 0.0001 |
| | DFS | 8 | 9 | 6 | 0.0001 |
| | IDDFS | 6 | 63 | 4 | 0.0003 |
| | Greedy | 6 | 15 | 1 | 0.0001 |
| | A\* | 6 | 15 | 0 | 0.0001 |
| `02_facil` (2 cajas) | BFS | 13 | 928 | 289 | 0.007 |
| | DFS | 75 | 82 | 76 | 0.0005 |
| | IDDFS | 13 | 5 000 | 21 | 0.022 |
| | Greedy | 13 | 36 | 22 | 0.0003 |
| | A\* | 13 | 293 | 184 | 0.0027 |
| `03_medio` (3 cajas) | BFS | 18 | 4 704 | 1 605 | 0.035 |
| | DFS | 1 004 | 1 405 | 933 | 0.011 |
| | IDDFS | 18 | 37 196 | 8 | 0.186 |
| | Greedy | 18 | 57 | 36 | 0.0005 |
| | A\* | 18 | 1 553 | 807 | 0.016 |
| `04_dificil` (4 cajas) | BFS | 24 | 39 939 | 19 752 | 0.313 |
| | DFS | 56 338 | 86 406 | 46 377 | 0.659 |
| | IDDFS | 24 | 334 167 | 13 | 1.632 |
| | Greedy | 24 | 88 | 40 | 0.0009 |
| | A\* | 24 | 9 356 | 4 849 | 0.098 |
| `05_seis_cajas` (6 cajas) | BFS | — | 1 050 624 | 657 634 | timeout |
| | DFS | — | 1 357 824 | 1 048 816 | timeout |
| | IDDFS | — | 2 260 992 | 23 | timeout |
| | Greedy | 32 | 123 | 100 | 0.002 |
| | A\* | **30** | 434 199 | 288 507 | 9.52 |
| `06_original` (6 cajas) | BFS | — | 1 169 408 | 110 534 | timeout |
| | DFS | 1 342 | 3 731 | 662 | 0.021 |
| | IDDFS | — | 1 978 368 | 22 | timeout |
| | Greedy | 274 | 6 388 | 300 | 0.054 |
| | A\* | — | 653 312 | 99 481 | timeout |

(Greedy y A\* con `matching`.)

**Conclusiones:**

- BFS, IDDFS y A\* **siempre coinciden en el costo**: los tres son óptimos. Es el
  chequeo de sanidad más útil del TP; si difieren, hay un bug.
- A\* con `matching` expande entre 3 y 5 veces menos nodos que BFS para
  el mismo costo óptimo, y la brecha crece con la cantidad de cajas: en
  `05_seis_cajas` es el único método que encuentra el óptimo (30 movimientos).
- Greedy es órdenes de magnitud más rápido pero **no** garantiza optimalidad: en
  `05_seis_cajas` devuelve 32 en vez de 30, y en `06_original` devuelve un camino
  de 274 movimientos que dista mucho del óptimo.
- DFS es el más barato en memoria y a veces "tiene suerte" (resuelve
  `06_original` en 21 ms), pero el costo es absurdo: 56 338 movimientos en
  `04_dificil` contra los 24 óptimos.
- IDDFS mantiene una frontera de una decena de nodos —dos a cinco órdenes de
  magnitud menos que BFS— y recupera la optimalidad, pero paga entre 7 y 10 veces
  más nodos expandidos por reexpandir los niveles superiores en cada iteración.
- El nivel 1 del Sokoban original sigue siendo duro para la búsqueda
  **óptima en movimientos**: su solución más corta ronda los 200+ movimientos y
  el factor de ramificación efectivo hace que A\* no termine en tiempo razonable.
  Es el límite natural de este enfoque, y el punto donde harían falta
  refinamientos (búsqueda sobre empujes en vez de movimientos, macro-moves,
  deadlocks más finos).
