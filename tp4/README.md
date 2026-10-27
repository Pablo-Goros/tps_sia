# SIA – TP4: Métodos de Aprendizaje No Supervisado

Implementación desde cero (numpy) de **PCA**, **Red de Kohonen**, **Regla de Oja** y **Red de Hopfield**,
aplicadas a (1) la selección de creadores de contenido para campañas de marketing y (2) la recuperación
de códigos QR deteriorados.

Las decisiones de diseño, hiperparámetros e interpretaciones están en [`DECISIONES.md`](DECISIONES.md).

## Estructura

```
tp4/
├── datos/
│   ├── creadores.csv            # 50 creadores x 33 columnas
│   └── hopfield/*_qr.png        # 5 QR de 37x37
├── src/
│   ├── preprocessing.py         # z-score, min-max, unit length
│   ├── pca.py                   # PCA por autovalores de la matriz de correlaciones
│   ├── kohonen.py               # SOM: vecindario, R(i), eta(i), QE/TE, Matriz U, planos
│   ├── oja.py                   # neurona lineal con regla de Oja
│   ├── hopfield.py              # W = (1/N) K K^T (o pseudoinversa), recall sync/async, energía, ruido, ortogonalidad
│   ├── deterioros.py            # desgastes de un QR: esquina rota, mancha, rayón, borrado, tinta desvanecida
│   ├── letras.py                # abecedario 5x5 en {1,-1} y búsqueda de grupos ortogonales
│   ├── qr.py                    # PNG <-> matriz {1,-1}, decodificación con OpenCV
│   └── plots.py
├── configs/                     # hiperparámetros finales (json)
├── ej1_eda.py  ej1_pca.py  ej1_kohonen.py  ej1_kohonen_hp.py  ej1_oja.py
├── ej2_letras.py  ej2_qr.py  ej2_qr_pinv.py
├── tests/test_teorica.py        # chequeos contra los ejemplos de las diapositivas
├── run_all.sh
└── resultados/                  # se genera al correr (figuras, csv, txt)
```

## Requisitos

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`opencv-python-headless` se usa únicamente como "lector de QR" para verificar que la salida de Hopfield es
decodificable; si no está instalado, `ej2_qr.py` corre igual y reporta `<cv2 no instalado>`.

## Ejecución

```bash
./run_all.sh                     # todo, en orden (≈ 6 min; el barrido de Kohonen es lo más lento)
```

o por partes (el orden importa porque cada script usa salidas del anterior):

```bash
python3 ej1_eda.py               # -> resultados/eda/variables_seleccionadas.json, tema_dominante.csv
python3 ej1_pca.py               # -> resultados/pca/ (usa la selección del EDA)
python3 ej1_kohonen.py           # barrido (864 corridas) + modelo final; --skip-search para solo el final
python3 ej1_kohonen_hp.py        # figuras de hiperparámetros a partir del barrido (hp_*.png)
python3 ej1_oja.py               # usa el autovector PC1 de ej1_pca para comparar
python3 ej1_campanias.py         # 4 casos de campaña (short/long x fútbol/perros) con Kohonen + índice
python3 ej2_letras.py
python3 ej2_qr.py
python3 ej2_qr_pinv.py           # los 5 QR con la regla de la pseudoinversa (≈ 3 min) -> resultados/hopfield_qr_pinv/
python3 tests/test_teorica.py    # o: python3 -m pytest tests/
```

Los hiperparámetros finales se editan en `configs/*.json`.
