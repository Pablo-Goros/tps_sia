import csv
from dataclasses import replace
import json
from pathlib import Path

import pytest

from sokoban.experimentos import (
    COLUMNAS_EJECUCIONES,
    COLUMNAS_RESUMEN,
    ErrorConfiguracion,
    cargar_configuracion,
    ejecutar_desde_configuracion,
    planificar_ejecuciones,
    preparar_niveles,
    resumir_ejecuciones,
)


TABLERO_UN_MOVIMIENTO = "#####\n#@$.#\n#####"


def crear_configuracion(
    carpeta: Path,
    *,
    semilla: int = 17,
    repeticiones: int = 5,
    algoritmos: list[str] | None = None,
    heuristicas: list[str] | None = None,
) -> Path:
    niveles = carpeta / "datos"
    niveles.mkdir(parents=True)
    (niveles / "micro.txt").write_text(
        TABLERO_UN_MOVIMIENTO, encoding="utf-8"
    )
    datos = {
        "semilla": semilla,
        "repeticiones": repeticiones,
        "niveles": [{"nombre": "micro", "ruta": "datos/micro.txt"}],
        "algoritmos": algoritmos or ["bfs", "astar"],
        "heuristicas": heuristicas or ["manhattan"],
        "max_expandidos": 100,
        "salidas": {
            "ejecuciones": "salida/ejecuciones.csv",
            "resumen": "salida/resumen.csv",
        },
    }
    ruta = carpeta / "configuracion.json"
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    return ruta


def test_las_rutas_se_resuelven_desde_el_archivo_de_configuracion(
    tmp_path: Path,
) -> None:
    ruta = crear_configuracion(tmp_path)

    configuracion = cargar_configuracion(ruta)

    assert configuracion.niveles[0].ruta == (tmp_path / "datos/micro.txt").resolve()
    assert configuracion.salida_ejecuciones == (
        tmp_path / "salida/ejecuciones.csv"
    ).resolve()
    assert configuracion.salida_resumen == (
        tmp_path / "salida/resumen.csv"
    ).resolve()
    assert configuracion.repeticiones == 5


@pytest.mark.parametrize(
    ("campo", "valor", "mensaje"),
    [
        ("repeticiones", 4, "repeticiones debe valer 5"),
        ("algoritmos", ["iddfs"], "algoritmo desconocido"),
        ("max_expandidos", 0, "max_expandidos debe ser mayor que cero"),
    ],
)
def test_configuracion_rechaza_valores_invalidos(
    tmp_path: Path,
    campo: str,
    valor: object,
    mensaje: str,
) -> None:
    ruta = crear_configuracion(tmp_path)
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    datos[campo] = valor
    ruta.write_text(json.dumps(datos), encoding="utf-8")

    with pytest.raises(ErrorConfiguracion, match=mensaje):
        cargar_configuracion(ruta)


def test_el_plan_baraja_solo_el_orden_con_una_semilla_local(
    tmp_path: Path,
) -> None:
    configuracion = cargar_configuracion(crear_configuracion(tmp_path))
    niveles = preparar_niveles(configuracion)

    primero = planificar_ejecuciones(configuracion, niveles)
    segundo = planificar_ejecuciones(configuracion, niveles)
    otra_semilla = planificar_ejecuciones(
        replace(configuracion, semilla=99), niveles
    )

    def firma(plan):
        return [
            (item.nivel.nombre, item.algoritmo, item.heuristica, item.repeticion)
            for item in plan
        ]

    assert firma(primero) == firma(segundo)
    assert firma(primero) != firma(otra_semilla)
    assert sorted(firma(primero)) == sorted(firma(otra_semilla))
    assert len(primero) == 10
    assert sorted(item.repeticion for item in primero if item.algoritmo == "bfs") == [
        1,
        2,
        3,
        4,
        5,
    ]


def test_cada_corrida_se_guarda_y_las_metricas_son_reproducibles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ruta = crear_configuracion(tmp_path)
    otro_directorio = tmp_path / "otro_directorio"
    otro_directorio.mkdir()
    monkeypatch.chdir(otro_directorio)

    primera = ejecutar_desde_configuracion(ruta)
    segunda = ejecutar_desde_configuracion(ruta)

    assert len(primera.ejecuciones) == 10
    assert len(primera.resumen) == 2
    assert all(fila["estado"] == "exito" for fila in primera.ejecuciones)
    assert all(fila["costo"] == 1 for fila in primera.ejecuciones)

    def sin_tiempo(fila):
        return {
            clave: valor
            for clave, valor in fila.items()
            if clave != "tiempo_segundos"
        }

    assert [sin_tiempo(fila) for fila in primera.ejecuciones] == [
        sin_tiempo(fila) for fila in segunda.ejecuciones
    ]

    with primera.salida_ejecuciones.open(encoding="utf-8", newline="") as archivo:
        crudas = list(csv.DictReader(archivo))
    with primera.salida_resumen.open(encoding="utf-8", newline="") as archivo:
        resumen = list(csv.DictReader(archivo))

    assert tuple(crudas[0]) == COLUMNAS_EJECUCIONES
    assert tuple(resumen[0]) == COLUMNAS_RESUMEN
    assert len(crudas) == 10
    assert len(resumen) == 2
    assert {fila["repeticiones"] for fila in resumen} == {"5"}
    assert {fila["exitos"] for fila in resumen} == {"5"}


def test_resumen_separa_estados_y_promedia_segun_la_especificacion() -> None:
    filas = []
    estados = ("exito", "exito", "fracaso", "corte", "corte")
    for indice, estado in enumerate(estados, start=1):
        filas.append(
            {
                "nivel": "mixto",
                "algoritmo": "astar",
                "heuristica": "manhattan",
                "estado": estado,
                "costo": {1: 4, 2: 6}.get(indice, ""),
                "nodos_expandidos": indice,
                "nodos_frontera": indice * 2,
                "max_nodos_frontera": indice * 3,
                "tiempo_segundos": float(indice),
            }
        )

    resumen = resumir_ejecuciones(filas)[0]

    assert resumen["repeticiones"] == 5
    assert resumen["exitos"] == 2
    assert resumen["fracasos"] == 1
    assert resumen["cortes"] == 2
    assert resumen["tasa_exito"] == 0.4
    assert resumen["costo_promedio"] == 5.0
    assert resumen["nodos_expandidos_promedio"] == 3.0
    assert resumen["nodos_frontera_promedio"] == 6.0
    assert resumen["max_nodos_frontera_promedio"] == 9.0
    assert resumen["tiempo_promedio_segundos"] == 3.0
    assert resumen["tiempo_desvio_segundos"] == pytest.approx(1.58113883)


def test_costo_promedio_queda_vacio_si_no_hay_exitos() -> None:
    filas = [
        {
            "nivel": "imposible",
            "algoritmo": "bfs",
            "heuristica": "",
            "estado": "fracaso",
            "costo": "",
            "nodos_expandidos": 10,
            "nodos_frontera": 0,
            "max_nodos_frontera": 3,
            "tiempo_segundos": 0.1,
        }
        for _ in range(5)
    ]

    resumen = resumir_ejecuciones(filas)[0]

    assert resumen["costo_promedio"] == ""
    assert resumen["fracasos"] == 5
    assert resumen["cortes"] == 0
