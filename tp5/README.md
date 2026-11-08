# TP5: autoencoder básico de caracteres

Ejercicio 1.a. Todos los comandos siguientes se ejecutan desde `tp5/`, con Python 3.10 o posterior.

## Instalación

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Inspección del dataset

```powershell
python -m src.dataset --output results/dataset
```

Produce `results/dataset/dataset.png`, con los 32 patrones etiquetados, y
`dataset.json`, con identidad SHA256, etiquetas y dimensiones. No requiere una
ventana gráfica. Para leer otra copia del header, agregar `--dataset ruta/font.h`.

`src.dataset.load_dataset(path)` expone `images` `(32,7,5)`, `vectors` flotantes
`(32,35)`, `labels`, `sha256` y `path`. El argumento es opcional y por defecto
apunta al original `data/font.h`, independientemente del directorio actual.
Lee únicamente la matriz `Font3[32][7]` sin ejecutar código ni incorporar los
comentarios. Conserva los códigos `0x60` a `0x7f`, incluido DEL, y convierte cada
fila del bit más significativo al menos significativo, de izquierda a derecha.
El aplanado conserva el orden de las filas de arriba abajo.

El original se conserva sin cambios. SHA256:
`095b431bdfa2d228eeaa337fb45cd963088cc13380c189e01436e446767502f8`.

## Verificación

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Las pruebas del flujo público verifican contenido y orientación asimétrica,
identidad del original, rechazo de datos mal formados y generación de la lámina
con su manifiesto. No entrenan modelos ni afirman que los patrones se aprendieron.
