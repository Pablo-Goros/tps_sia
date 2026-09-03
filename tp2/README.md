# ITBA - Sistemas de Inteligencia Artificial
## Trabajo Práctico 2: Algoritmos Genéticos - Compresor de Imágenes con Triángulos

Implementación de un compresor de imágenes bioinspirado mediante **Algoritmos Genéticos** puros (sin librerías externas de AG), orientado a aproximar una imagen objetivo a través de una superposición de triángulos translúcidos de color uniforme sobre un canvas.

Desarrollado siguiendo los apuntes de cátedra (`Algoritmos_Geneticos.md`) y el libro de referencia **Artificializando** (Capítulo 5, págs. 49–59).

---

## 1. Fundamentos Teóricos y Modelado

### ¿Qué es un individuo en este problema?
* **Cromosoma**: Representa una solución candidata compuesta por una secuencia ordenada de $T$ triángulos. El orden en la secuencia determina el apilamiento de capas (*z-index*) al pintar sobre el canvas con mezcla alfa (alpha blending).
* **Gen (Triángulo)**: Cada gen contiene los parámetros geométricos y de color del triángulo:
  $$\text{Gen}_k = \Big( (x_1, y_1), (x_2, y_2), (x_3, y_3), (r, g, b, \alpha) \Big)$$
  - Vértices: Coordenadas normalizadas $(x_i, y_i) \in [0.0, 1.0]$ para independencia de escala.
  - Color y Opacidad: Canales $r, g, b \in [0, 255]$ y $\alpha \in [0, 255]$ (opacidad).

### Función de Aptitud (Fitness) y Error
El renderizado del individuo produce una imagen RGB $I_{\text{ind}}$. Se calcula el **Error Cuadrático Medio (MSE)** respecto a la imagen objetivo $I_{\text{target}}$:
$$MSE = \frac{1}{3 \cdot W \cdot H} \sum_{x=1}^{W} \sum_{y=1}^{H} \sum_{c \in \{R,G,B\}} \Big( I_{\text{ind}}(x, y, c) - I_{\text{target}}(x, y, c) \Big)^2$$

Para operadores estocásticos (Ruleta, Universal, Boltzmann) que requieren un fitness estrictamente positivo y acotado, se normaliza el MSE y se define la función de aptitud monótona decreciente respecto al error:
$$Fitness(i) = \frac{1}{1 + \frac{MSE(i)}{255^2}} \in (0, 1]$$
Un fitness cercano a $1.0$ representa una aproximación fiel con error casi nulo.

---

## 2. Operadores Implementados

### A. Métodos de Selección
1. **Elite**: Selecciona los individuos más aptos repitiendo cada uno según $n(i) = \lceil \frac{K - i}{N} \rceil$.
2. **Ruleta**: Muestreo proporcional a la aptitud relativa $p(i) = \frac{f(i)}{\sum f(j)}$.
3. **Universal (SUS)**: Muestreo universal estocástico con un único punto de partida aleatorio y punteros equidistantes $r_j = \frac{r + j}{K}$.
4. **Ranking**: Pseudo-aptitud basada en el orden de aptitud $f'(i) = \frac{N - \text{rank}(i) + 1}{N}$ combinada con ruleta.
5. **Boltzmann**: Enfriamiento simulado con temperatura $T(t) = T_c + (T_0 - T_c)e^{-kt}$ y pseudo-aptitud proporcional a $e^{f(i)/T}$.
6. **Torneos Determinísticos**: Selecciona $M$ individuos al azar y conserva el de mayor fitness.
7. **Torneos Probabilísticos**: Selecciona 2 individuos al azar; con probabilidad $Threshold \in [0.5, 1.0]$ gana el más apto, de lo contrario el menos apto.

### B. Métodos de Cruce (Crossover)
1. **Cruce de Dos Puntos**: Se seleccionan dos loci de corte $P_1 \le P_2$ y se intercambia el segmento central de triángulos entre los padres.
2. **Cruce Uniforme**: Para cada locus, con probabilidad $P_{\text{swap}}$ (por defecto $0.5$) se heredan los genes de un padre o del otro.

### C. Métodos de Mutación (4 variantes)
1. **Mutación de Gen (`gen`)**: Con probabilidad $P_m$, se altera un solo gen del individuo (perturbación delta o reemplazo).
2. **Mutación Multigen Limitada (`multigen_limitada`)**: Con probabilidad $P_m$, se elige un número aleatorio $m \in [1, M]$ de genes a mutar.
3. **Mutación Multigen Uniforme (`multigen_uniforme`)**: Cada gen del cromosoma tiene una probabilidad independiente $P_m$ de mutar.
4. **Mutación No Uniforme (`no_uniforme`)**: La amplitud del delta de mutación decrece con el avance de las generaciones:
   $$\text{scale}(t) = \left(1 - \frac{t}{T_{\max}}\right)^b$$
   favoreciendo gran exploración al inicio y ajuste fino (explotación) hacia las generaciones finales.

### D. Estrategias de Supervivencia
1. **Aditiva**: Combina la población actual y los descendientes $[N + K]$ y selecciona los mejores $N$.
2. **Exclusiva**: Si $K > N$, selecciona $N$ exclusivamente de los hijos. Si $K \le N$, toma los $K$ hijos y completa los $N - K$ restantes de los padres.
3. **Brecha Generacional**: Mezcla $(1 - G) \cdot N$ individuos de los padres con $G \cdot N$ individuos de los hijos ($G \in [0, 1]$).

### E. Criterios de Parada y Seguridad
* **Tope Lógico de Triángulos**: Validación en tiempo de inicialización (`MAX_ALLOWED_TRIANGLES = 200`, configurable) para evitar colapsos de memoria o cómputo excesivo.
* **Timeout en segundos**: Permite fijar un tiempo máximo de ejecución de reloj de pared.
* **Generaciones máximas**: Cota sobre el ciclo evolutivo.
* **Solución aceptable**: Corte temprano al alcanzar un umbral de fitness o MSE.
* **Contenido**: Corte si el mejor fitness no mejora en más de $\epsilon$ durante $W$ generaciones.
* **Estructura**: Corte si la varianza de la población cae por debajo de un umbral durante $W$ generaciones.

---

## 3. Instalación y Requisitos

Requiere **Python >= 3.9** y las librerías `numpy`, `pillow`, `matplotlib` y `pytest`.

```bash
# 1. Crear y activar entorno virtual (o utilizar el existente)
python3 -m venv .venv
source .venv/bin/activate

# 2. Actualizar pip (recomendado para soportar instalaciones editables con pyproject.toml)
python3 -m pip install --upgrade pip

# 3. Instalar dependencias
pip install numpy pillow matplotlib pytest

# 4. Instalar el paquete en modo editable (desde la raíz del repo)
pip install -e tp2
```

---

## 4. Guía de Ejecución

### Ejecución básica usando `configuracion.json`
```bash
python3 -m tp2.src.cli --config tp2/configuracion.json
```

### Ejecución con flags CLI (sobreescribiendo configuración)
```bash
# Aproximar bandera de Japón con 25 triángulos, cruce uniforme, mutación no uniforme y timeout de 30s
python3 -m tp2.src.cli \
  --image tp2/ejemplos/japon.png \
  --triangles 25 \
  --generations 150 \
  --timeout 30 \
  --pop-size 40 \
  --offspring 40 \
  --parent-sel ruleta \
  --crossover uniforme \
  --mutation no_uniforme \
  --pm 0.15 \
  --output-dir tp2/salidas \
  --plot
```

### Opciones de la CLI
| Parámetro | Tipo | Descripción | Opciones |
|---|---|---|---|
| `-c`, `--config` | string | Ruta al archivo JSON de configuración | Por defecto `tp2/configuracion.json` |
| `-i`, `--image` | string | Ruta a la imagen objetivo | e.g. `tp2/ejemplos/japon.png` |
| `-t`, `--triangles` | int | Cantidad de triángulos | 1 a 200 (tope lógico) |
| `-g`, `--generations` | int | Máximo de generaciones | e.g. `200` |
| `--timeout` | float | Timeout máximo en segundos | e.g. `60.0` |
| `-n`, `--pop-size` | int | Tamaño de la población $N$ | e.g. `40` |
| `-k`, `--offspring` | int | Cantidad de hijos por generación $K$ | e.g. `40` |
| `--parent-sel` | string | Selección de padres | `elite`, `ruleta`, `universal`, `ranking`, `boltzmann`, `torneo_deterministico`, `torneo_probabilistico` |
| `--survival-sel` | string | Selección para supervivencia | (mismas que padres) |
| `--survival-strat` | string | Estrategia de reemplazo | `aditiva`, `exclusiva`, `brecha` |
| `--crossover` | string | Método de cruce | `dos_puntos`, `uniforme` |
| `--mutation` | string | Método de mutación | `gen`, `multigen_limitada`, `multigen_uniforme`, `no_uniforme` |
| `--pm` | float | Probabilidad de mutación | e.g. `0.1` |
| `-o`, `--output-dir` | string | Directorio de resultados | e.g. `salidas` |
| `--plot` | flag | Genera gráfico de curvas al terminar | - |

---

## 5. Salidas Generadas

Al finalizar la corrida, en el directorio de salida (ej. `salidas/`) se generan:
1. `mejor_imagen.png`: Imagen renderizada en resolución completa con la mejor configuración de triángulos encontrada.
2. `triangulos.json`: Estructura JSON completa con la enumeración de triángulos, coordenadas de vértices normalizadas y canales RGBA.
3. `metricas.csv`: Registro paso a paso por generación con:
   - `generacion`
   - `mejor_fitness`
   - `promedio_fitness`
   - `peor_fitness`
   - `mejor_mse`
   - `diversidad`
   - `segundos_transcurridos`
4. `snapshots/`: Capturas intermedias de la evolución de la imagen (`gen_00000.png`, `gen_00050.png`, ...).
5. `graficos_evolucion.png` (si se usa `--plot`): Gráfico de 3 paneles con las curvas de evolución de fitness, MSE y diversidad genética.

---

## 6. Ejecución de Tests

Para ejecutar la suite completa de 29 pruebas unitarias e integración:
```bash
pytest tp2/tests -v
```
