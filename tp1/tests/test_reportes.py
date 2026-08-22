import csv
import json
from pathlib import Path

import pytest

from sokoban.experimentos import COLUMNAS_RESUMEN, cargar_configuracion
from sokoban.reportes import (
    ErrorReporte,
    NOMBRES_FIGURAS,
    SUBCARPETA_FILTRADAS,
    cargar_resumen,
    crear_series,
    filtrar_configuracion,
    filtrar_mediciones,
    generar_desde_configuracion,
    generar_figuras,
    requiere_escala_logaritmica,
    validar_resumen,
)


def crear_datos_reporte(carpeta: Path) -> tuple[Path, Path]:
    configuracion = carpeta / "configuracion.json"
    configuracion.write_text(
        json.dumps(
            {
                "semilla": 1,
                "repeticiones": 5,
                "niveles": [{"nombre": "micro", "ruta": "nivel.txt"}],
                "algoritmos": ["bfs"],
                "heuristicas": [],
                "salidas": {
                    "ejecuciones": "resultados/ejecuciones.csv",
                    "resumen": "resultados/resumen.csv",
                    "figuras": "resultados/figuras",
                },
            }
        ),
        encoding="utf-8",
    )
    (carpeta / "nivel.txt").write_text(
        "#####\n#@$.#\n#####", encoding="utf-8"
    )
    resumen = carpeta / "resultados" / "resumen.csv"
    resumen.parent.mkdir()
    with resumen.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(
            archivo, fieldnames=COLUMNAS_RESUMEN, lineterminator="\n"
        )
        escritor.writeheader()
        escritor.writerow(
            {
                "nivel": "micro",
                "algoritmo": "bfs",
                "heuristica": "",
                "repeticiones": 5,
                "exitos": 5,
                "fracasos": 0,
                "cortes": 0,
                "tasa_exito": 1.0,
                "costo_promedio": 1.0,
                "nodos_expandidos_promedio": 1.0,
                "nodos_frontera_promedio": 0.0,
                "max_nodos_frontera_promedio": 1.0,
                "tiempo_promedio_segundos": 0.001,
                "tiempo_desvio_segundos": 0.0002,
            }
        )
    return configuracion, resumen


@pytest.mark.parametrize(
    ("valores", "esperada"),
    [
        ([1.0, 99.99], False),
        ([1.0, 100.0], True),
        ([0.0, 2.0, 300.0], True),
        ([0.0, 0.0], False),
    ],
)
def test_escala_logaritmica_usa_un_umbral_determinista(
    valores: list[float], esperada: bool
) -> None:
    assert requiere_escala_logaritmica(valores) is esperada


def test_resumen_valida_que_los_estados_sumen_las_repeticiones(
    tmp_path: Path,
) -> None:
    _, resumen = crear_datos_reporte(tmp_path)
    contenido = resumen.read_text(encoding="utf-8")
    resumen.write_text(
        contenido.replace(",5,0,0,1.0,", ",4,0,0,1.0,"),
        encoding="utf-8",
    )

    with pytest.raises(ErrorReporte, match="no suman las repeticiones"):
        cargar_resumen(resumen)


def test_genera_exactamente_cuatro_png_con_rotulos_y_barras_de_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    configuracion_ruta, resumen_ruta = crear_datos_reporte(tmp_path)
    configuracion = cargar_configuracion(configuracion_ruta)
    mediciones = cargar_resumen(resumen_ruta)
    barras_error = []
    rotulos = {}
    bar_original = Axes.bar
    savefig_original = Figure.savefig

    def registrar_barra(eje, *args, **kwargs):
        if kwargs.get("yerr") is not None:
            barras_error.append(list(kwargs["yerr"]))
        return bar_original(eje, *args, **kwargs)

    def registrar_figura(figura, destino, *args, **kwargs):
        eje = figura.axes[0]
        rotulos[Path(destino).name] = (
            eje.get_title(),
            eje.get_xlabel(),
            eje.get_ylabel(),
        )
        return savefig_original(figura, destino, *args, **kwargs)

    monkeypatch.setattr(Axes, "bar", registrar_barra)
    monkeypatch.setattr(Figure, "savefig", registrar_figura)

    destinos = generar_figuras(configuracion, mediciones)

    assert tuple(destino.name for destino in destinos) == NOMBRES_FIGURAS
    assert {ruta.name for ruta in configuracion.salida_figuras.iterdir()} == set(
        NOMBRES_FIGURAS
    )
    assert all(
        destino.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        for destino in destinos
    )
    assert barras_error == [[0.0002]]
    assert rotulos["tiempo_promedio.png"] == (
        "Tiempo promedio por configuración",
        "Nivel",
        "Tiempo promedio (segundos)",
    )
    assert rotulos["costo_solucion.png"][2] == "Costo promedio (movimientos)"


def crear_datos_filtrables(carpeta: Path) -> Path:
    """Configuración con dos niveles, un método desinformado y uno informado."""
    configuracion = carpeta / "configuracion.json"
    configuracion.write_text(
        json.dumps(
            {
                "semilla": 1,
                "repeticiones": 5,
                "niveles": [
                    {"nombre": "micro", "ruta": "micro.txt"},
                    {"nombre": "mini", "ruta": "mini.txt"},
                ],
                "algoritmos": ["bfs", "astar"],
                "heuristicas": ["manhattan", "empujes_inversos"],
                "salidas": {
                    "ejecuciones": "resultados/ejecuciones.csv",
                    "resumen": "resultados/resumen.csv",
                    "figuras": "resultados/figuras",
                },
            }
        ),
        encoding="utf-8",
    )
    (carpeta / "micro.txt").write_text("#####\n#@$.#\n#####", encoding="utf-8")
    (carpeta / "mini.txt").write_text("######\n#@$ .#\n######", encoding="utf-8")

    resumen = carpeta / "resultados" / "resumen.csv"
    resumen.parent.mkdir()
    with resumen.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(
            archivo, fieldnames=COLUMNAS_RESUMEN, lineterminator="\n"
        )
        escritor.writeheader()
        for nivel in ("micro", "mini"):
            for algoritmo, heuristica in (
                ("bfs", ""),
                ("astar", "manhattan"),
                ("astar", "empujes_inversos"),
            ):
                escritor.writerow(
                    {
                        "nivel": nivel,
                        "algoritmo": algoritmo,
                        "heuristica": heuristica,
                        "repeticiones": 5,
                        "exitos": 5,
                        "fracasos": 0,
                        "cortes": 0,
                        "tasa_exito": 1.0,
                        "costo_promedio": 1.0,
                        "nodos_expandidos_promedio": 1.0,
                        "nodos_frontera_promedio": 0.0,
                        "max_nodos_frontera_promedio": 1.0,
                        "tiempo_promedio_segundos": 0.001,
                        "tiempo_desvio_segundos": 0.0002,
                    }
                )
    return configuracion


def test_filtrar_respeta_el_orden_de_la_configuracion(tmp_path: Path) -> None:
    configuracion = cargar_configuracion(crear_datos_filtrables(tmp_path))

    filtrada = filtrar_configuracion(
        configuracion,
        niveles=["mini", "micro"],
        algoritmos=["astar", "bfs"],
        heuristicas=["empujes_inversos"],
    )

    assert tuple(nivel.nombre for nivel in filtrada.niveles) == ("micro", "mini")
    assert filtrada.algoritmos == ("bfs", "astar")
    assert filtrada.heuristicas == ("empujes_inversos",)
    assert tuple(serie.clave for serie in crear_series(filtrada)) == (
        ("bfs", ""),
        ("astar", "empujes_inversos"),
    )


def test_filtrar_rechaza_lo_que_la_configuracion_no_declara(
    tmp_path: Path,
) -> None:
    configuracion = cargar_configuracion(crear_datos_filtrables(tmp_path))

    with pytest.raises(ErrorReporte, match="nivel fuera de la configuración"):
        filtrar_configuracion(configuracion, niveles=["gigante"])
    with pytest.raises(ErrorReporte, match="algoritmo fuera de la configuración"):
        filtrar_configuracion(configuracion, algoritmos=["greedy"])
    with pytest.raises(ErrorReporte, match="no puede estar vacía"):
        filtrar_configuracion(configuracion, algoritmos=["astar"], heuristicas=[])


def test_filtrar_mediciones_deja_solo_las_series_pedidas(tmp_path: Path) -> None:
    ruta = crear_datos_filtrables(tmp_path)
    configuracion = cargar_configuracion(ruta)
    mediciones = cargar_resumen(configuracion.salida_resumen)
    filtrada = filtrar_configuracion(
        configuracion, niveles=["micro"], algoritmos=["bfs"]
    )

    conservadas = filtrar_mediciones(filtrada, mediciones)

    assert len(mediciones) == 6
    assert tuple(medicion.clave for medicion in conservadas) == (("micro", "bfs", ""),)
    # El resumen sobrante no debe invalidar el subconjunto pedido.
    validar_resumen(filtrada, conservadas)


def test_las_figuras_filtradas_no_pisan_las_del_informe(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    ruta = crear_datos_filtrables(tmp_path)
    figuras_informe = tmp_path / "resultados" / "figuras"

    completas = generar_desde_configuracion(ruta)
    filtradas = generar_desde_configuracion(ruta, algoritmos=["bfs"])
    elegidas = generar_desde_configuracion(
        ruta, algoritmos=["bfs"], salida_figuras=tmp_path / "elegida"
    )

    assert {destino.parent for destino in completas} == {figuras_informe}
    assert {destino.parent for destino in filtradas} == {
        figuras_informe / SUBCARPETA_FILTRADAS
    }
    assert {destino.parent for destino in elegidas} == {tmp_path / "elegida"}
    assert {ruta.name for ruta in figuras_informe.iterdir()} == set(
        NOMBRES_FIGURAS
    ) | {SUBCARPETA_FILTRADAS}
