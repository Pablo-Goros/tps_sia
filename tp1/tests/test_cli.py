import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from sokoban.reportes import NOMBRES_FIGURAS


RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
NIVEL_UN_MOVIMIENTO = RAIZ_PROYECTO / "niveles" / "nivel_01_trivial.txt"


def ejecutar_cli(*argumentos: str) -> subprocess.CompletedProcess[str]:
    entorno = os.environ.copy()
    ruta_src = str(RAIZ_PROYECTO / "src")
    pythonpath = entorno.get("PYTHONPATH")
    entorno["PYTHONPATH"] = (
        ruta_src + os.pathsep + pythonpath if pythonpath else ruta_src
    )
    return subprocess.run(
        [sys.executable, "-m", "sokoban", *argumentos],
        cwd=RAIZ_PROYECTO,
        env=entorno,
        capture_output=True,
        text=True,
        check=False,
    )


def test_ayuda_de_resolver_expone_solo_la_interfaz_requerida() -> None:
    completado = ejecutar_cli("resolver", "--help")

    assert completado.returncode == 0
    assert "--nivel RUTA" in completado.stdout
    assert "--algoritmo {bfs,dfs,greedy,astar}" in completado.stdout
    assert "--heuristica {empujes_inversos,manhattan}" in completado.stdout
    assert "--max-expandidos N" in completado.stdout
    assert "--mostrar-estados" in completado.stdout
    assert "iddfs" not in completado.stdout
    assert "gif" not in completado.stdout.lower()
    assert "pygame" not in completado.stdout.lower()


def test_resolver_usa_a_estrella_y_empujes_inversos_por_defecto() -> None:
    completado = ejecutar_cli("resolver", "--nivel", str(NIVEL_UN_MOVIMIENTO))

    assert completado.returncode == 0, completado.stderr
    assert "Estado:                  exito" in completado.stdout
    assert "Algoritmo:               A*" in completado.stdout
    assert "Heurística:              empujes_inversos" in completado.stdout
    assert "Costo:                   6" in completado.stdout
    assert "Movimientos:" in completado.stdout
    assert "Secuencia (U/D/L/R):" in completado.stdout
    assert "Nodos expandidos:" in completado.stdout
    assert "Nodos en frontera:" in completado.stdout
    assert "Máximo en frontera:" in completado.stdout
    assert "Tiempo (segundos):" in completado.stdout
    assert "Motivo:                  exito" in completado.stdout


@pytest.mark.parametrize("algoritmo", ("bfs", "dfs"))
def test_busquedas_desinformadas_rechazan_una_heuristica_sin_traceback(
    algoritmo: str,
) -> None:
    completado = ejecutar_cli(
        "resolver",
        "--nivel",
        str(NIVEL_UN_MOVIMIENTO),
        "--algoritmo",
        algoritmo,
        "--heuristica",
        "manhattan",
    )

    assert completado.returncode == 2
    assert "{} no acepta heurísticas".format(algoritmo) in completado.stderr
    assert "Traceback" not in completado.stderr


def test_limite_de_expansiones_debe_ser_positivo() -> None:
    completado = ejecutar_cli(
        "resolver",
        "--nivel",
        str(NIVEL_UN_MOVIMIENTO),
        "--max-expandidos",
        "0",
    )

    assert completado.returncode == 2
    assert "debe ser mayor que cero" in completado.stderr
    assert "Traceback" not in completado.stderr


def test_error_de_nivel_se_informa_sin_traceback(tmp_path: Path) -> None:
    nivel_invalido = tmp_path / "invalido.txt"
    nivel_invalido.write_text("###\n#@x\n###", encoding="utf-8")

    completado = ejecutar_cli("resolver", "--nivel", str(nivel_invalido))

    assert completado.returncode == 2
    assert "no se pudo cargar el nivel" in completado.stderr
    assert "caracter desconocido" in completado.stderr
    assert "Traceback" not in completado.stderr


def test_mostrar_estados_reproduce_la_solucion_en_ascii(tmp_path: Path) -> None:
    nivel = tmp_path / "un_movimiento.txt"
    nivel.write_text("#####\n#@$.#\n#####", encoding="utf-8")

    completado = ejecutar_cli(
        "resolver",
        "--nivel",
        str(nivel),
        "--mostrar-estados",
    )

    assert completado.returncode == 0, completado.stderr
    assert "Paso 0 (estado inicial)\n#####\n#@$.#\n#####" in completado.stdout
    assert "Paso 1 (derecha)\n#####\n# @*#\n#####" in completado.stdout


def test_un_corte_devuelve_estado_y_motivo_distintos_del_fracaso(
    tmp_path: Path,
) -> None:
    nivel = tmp_path / "dos_movimientos.txt"
    nivel.write_text("######\n#@ $.#\n######", encoding="utf-8")

    completado = ejecutar_cli(
        "resolver",
        "--nivel",
        str(nivel),
        "--algoritmo",
        "bfs",
        "--max-expandidos",
        "1",
    )

    assert completado.returncode == 1
    assert "Estado:                  corte" in completado.stdout
    assert "Costo:                   -" in completado.stdout
    assert "Motivo:                  limite_nodos" in completado.stdout


def test_error_de_configuracion_se_informa_sin_traceback(tmp_path: Path) -> None:
    inexistente = tmp_path / "no_existe.json"

    completado = ejecutar_cli(
        "experimentar", "--configuracion", str(inexistente)
    )

    assert completado.returncode == 2
    assert "configuración inválida" in completado.stderr
    assert "no se pudo leer" in completado.stderr
    assert "Traceback" not in completado.stderr


def test_experimentar_ejecuta_cinco_repeticiones_y_escribe_ambos_csv(
    tmp_path: Path,
) -> None:
    nivel = tmp_path / "nivel.txt"
    nivel.write_text("#####\n#@$.#\n#####", encoding="utf-8")
    configuracion = tmp_path / "configuracion.json"
    configuracion.write_text(
        json.dumps(
            {
                "semilla": 7,
                "repeticiones": 5,
                "niveles": [{"nombre": "micro", "ruta": "nivel.txt"}],
                "algoritmos": ["astar"],
                "heuristicas": ["manhattan"],
                "salidas": {
                    "ejecuciones": "resultados/ejecuciones.csv",
                    "resumen": "resultados/resumen.csv",
                },
            }
        ),
        encoding="utf-8",
    )

    completado = ejecutar_cli(
        "experimentar", "--configuracion", str(configuracion)
    )

    assert completado.returncode == 0, completado.stderr
    assert "[5/5]" in completado.stdout
    assert "Ejecuciones: 5" in completado.stdout
    assert "Resumen:     1 grupos" in completado.stdout
    assert (tmp_path / "resultados/ejecuciones.csv").is_file()
    assert (tmp_path / "resultados/resumen.csv").is_file()


def test_graficar_informa_errores_sin_traceback(tmp_path: Path) -> None:
    inexistente = tmp_path / "no_existe.json"

    completado = ejecutar_cli("graficar", "--configuracion", str(inexistente))

    assert completado.returncode == 2
    assert "no se pudieron generar las figuras" in completado.stderr
    assert "no se pudo leer" in completado.stderr
    assert "Traceback" not in completado.stderr


def test_ayuda_de_graficar_expone_los_filtros() -> None:
    completado = ejecutar_cli("graficar", "--ayuda")

    assert completado.returncode == 0
    for opcion in ("--niveles", "--algoritmos", "--heuristicas", "--salida-figuras"):
        assert opcion in completado.stdout
    assert "iddfs" not in completado.stdout


def test_graficar_filtrado_escribe_donde_se_le_pide(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    destino = tmp_path / "figuras_elegidas"

    completado = ejecutar_cli(
        "graficar",
        "--algoritmos",
        "bfs",
        "--niveles",
        "facil",
        "medio",
        "--salida-figuras",
        str(destino),
    )

    assert completado.returncode == 0, completado.stderr
    assert {ruta.name for ruta in destino.iterdir()} == set(NOMBRES_FIGURAS)


def test_graficar_rechaza_un_nivel_ausente_de_la_configuracion() -> None:
    completado = ejecutar_cli("graficar", "--niveles", "inexistente")

    assert completado.returncode == 2
    assert "nivel fuera de la configuración" in completado.stderr
    assert "Traceback" not in completado.stderr


def test_ayuda_de_animar_expone_las_opciones_de_salida() -> None:
    completado = ejecutar_cli("animar", "--ayuda")

    assert completado.returncode == 0
    for opcion in ("--nivel RUTA", "--salida RUTA", "--formato {gif,mp4}",
                   "--fps N", "--max-frames N", "--frames CARPETA"):
        assert opcion in completado.stdout


def test_animar_genera_un_gif_en_la_ruta_pedida(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    destino = tmp_path / "solucion.gif"

    completado = ejecutar_cli(
        "animar",
        "--nivel",
        str(NIVEL_UN_MOVIMIENTO),
        "--salida",
        str(destino),
    )

    assert completado.returncode == 0, completado.stderr
    assert destino.read_bytes().startswith(b"GIF89a")
    assert str(destino) in completado.stdout


def test_animar_informa_cuando_no_hay_solucion_que_mostrar() -> None:
    sin_solucion = RAIZ_PROYECTO / "niveles" / "nivel_00_sin_solucion.txt"

    completado = ejecutar_cli("animar", "--nivel", str(sin_solucion))

    assert completado.returncode == 2
    assert "no hay solución que animar" in completado.stderr
    assert "Traceback" not in completado.stderr


def test_animar_usa_resultados_animaciones_cuando_no_se_da_salida(
    tmp_path: Path,
) -> None:
    pytest.importorskip("PIL")
    nivel = tmp_path / "micro.txt"
    nivel.write_text("#####\n#@$.#\n#####\n", encoding="utf-8")
    destino = RAIZ_PROYECTO / "resultados" / "animaciones" / "micro_astar.gif"
    destino.unlink(missing_ok=True)

    try:
        completado = ejecutar_cli("animar", "--nivel", str(nivel))

        assert completado.returncode == 0, completado.stderr
        assert destino.read_bytes().startswith(b"GIF89a")
    finally:
        destino.unlink(missing_ok=True)


def test_animar_rechaza_una_heuristica_en_un_metodo_desinformado() -> None:
    completado = ejecutar_cli(
        "animar",
        "--nivel",
        str(NIVEL_UN_MOVIMIENTO),
        "--algoritmo",
        "bfs",
        "--heuristica",
        "manhattan",
    )

    assert completado.returncode == 2
    assert "no acepta heurísticas" in completado.stderr
