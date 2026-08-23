from pathlib import Path

import pytest

from sokoban.animacion import (
    CARPETA_ANIMACIONES,
    ErrorAnimacion,
    MAX_FRAMES_GIF,
    dibujar_frame,
    dimensiones,
    duracion_formateada,
    es_video,
    estados_de_la_solucion,
    fps_sugerido,
    generar_gif,
    generar_video,
    guardar_frames,
    hay_ffmpeg,
    indices_de_frames,
    submuestreo_efectivo,
)
from sokoban.nivel import parsear_tablero
from sokoban.problema import ProblemaSokoban


PASILLO = "########\n#@$   .#\n########\n"


def crear_problema() -> ProblemaSokoban:
    mapa, estado = parsear_tablero(PASILLO)
    return ProblemaSokoban(mapa, estado)


def test_los_indices_conservan_el_primero_y_el_ultimo_paso() -> None:
    assert indices_de_frames(5) == [0, 1, 2, 3, 4]
    assert indices_de_frames(10, submuestreo=3) == [0, 3, 6, 9]
    # El ultimo estado se agrega aunque el submuestreo no caiga justo.
    assert indices_de_frames(11, submuestreo=3) == [0, 3, 6, 9, 10]


def test_el_tope_de_frames_acota_los_caminos_largos() -> None:
    indices = indices_de_frames(10_000, max_frames=100)

    assert len(indices) <= 101
    assert indices[0] == 0
    assert indices[-1] == 9_999
    assert submuestreo_efectivo(10_000, max_frames=100) == 100


@pytest.mark.parametrize(
    ("submuestreo", "max_frames"),
    [(0, None), (-1, None), (1, 1), (1, 0)],
)
def test_los_indices_rechazan_parametros_invalidos(
    submuestreo: int, max_frames: int | None
) -> None:
    with pytest.raises(ErrorAnimacion):
        indices_de_frames(10, submuestreo, max_frames)


def test_los_estados_se_reproducen_con_las_reglas_del_problema() -> None:
    problema = crear_problema()

    estados = estados_de_la_solucion(problema, ("derecha", "derecha"))

    assert len(estados) == 3
    assert estados[0] is problema.estado_inicial
    assert estados[-1].jugador == (3, 1)
    assert estados[-1].cajas == frozenset({(4, 1)})


def test_un_movimiento_ilegal_no_se_anima() -> None:
    problema = crear_problema()

    with pytest.raises(ErrorAnimacion, match="el movimiento 1 de la solucion"):
        estados_de_la_solucion(problema, ("arriba",))


def test_el_frame_mide_lo_que_declara_dimensiones() -> None:
    problema = crear_problema()

    con_encabezado = dibujar_frame(
        problema.estado_inicial, problema.mapa, titulo="t", subtitulo="s"
    )
    sin_encabezado = dibujar_frame(
        problema.estado_inicial, problema.mapa, encabezado=False
    )

    assert con_encabezado.size == dimensiones(problema.mapa)
    assert sin_encabezado.size == dimensiones(problema.mapa, encabezado=False)
    assert con_encabezado.size[1] > sin_encabezado.size[1]


def test_el_gif_guarda_un_frame_por_paso(tmp_path: Path) -> None:
    from PIL import Image

    problema = crear_problema()
    movimientos = ("derecha", "derecha", "derecha", "derecha")
    estados = estados_de_la_solucion(problema, movimientos)
    destino = tmp_path / "animacion.gif"

    generar_gif(estados, problema.mapa, destino, movimientos)

    with Image.open(destino) as imagen:
        assert imagen.format == "GIF"
        assert imagen.n_frames == len(estados)


def test_el_gif_saltea_pasos_cuando_el_camino_es_largo(tmp_path: Path) -> None:
    from PIL import Image

    problema = crear_problema()
    estados = [problema.estado_inicial] * 1_000
    destino = tmp_path / "larga.gif"

    generar_gif(estados, problema.mapa, destino, max_frames=25)

    with Image.open(destino) as imagen:
        assert imagen.n_frames <= 26
    # El tope es una cota, no un objetivo: se salta de a pasos enteros.
    assert len(indices_de_frames(1_000, max_frames=MAX_FRAMES_GIF)) <= MAX_FRAMES_GIF
    assert submuestreo_efectivo(1_000, max_frames=MAX_FRAMES_GIF) == 3


def test_los_png_sueltos_se_numeran_por_paso(tmp_path: Path) -> None:
    problema = crear_problema()
    movimientos = ("derecha", "derecha")
    estados = estados_de_la_solucion(problema, movimientos)

    rutas = guardar_frames(estados, problema.mapa, tmp_path / "frames", movimientos)

    assert [ruta.name for ruta in rutas] == ["paso_0.png", "paso_1.png", "paso_2.png"]
    assert all(ruta.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for ruta in rutas)


@pytest.mark.parametrize(
    ("cantidad", "fps", "esperado"),
    [(12, 12, "1.0 s"), (600, 60, "10.0 s"), (7104, 60, "1:58.4 min")],
)
def test_la_duracion_se_formatea_para_leerla_de_un_vistazo(
    cantidad: int, fps: int, esperado: str
) -> None:
    assert duracion_formateada(cantidad, fps) == esperado


def test_los_caminos_largos_sugieren_mas_cuadros_por_segundo() -> None:
    assert fps_sugerido(50) == 12
    assert fps_sugerido(5_000) == 60


@pytest.mark.parametrize(
    ("nombre", "esperado"),
    [("a.mp4", True), ("a.MOV", True), ("a.gif", False), ("a", False)],
)
def test_la_extension_decide_el_formato(nombre: str, esperado: bool) -> None:
    assert es_video(nombre) is esperado


@pytest.mark.skipif(not hay_ffmpeg(), reason="ffmpeg no esta en el PATH")
def test_el_video_se_escribe_por_streaming(tmp_path: Path) -> None:
    problema = crear_problema()
    movimientos = ("derecha", "derecha", "derecha", "derecha")
    estados = estados_de_la_solucion(problema, movimientos)
    destino = tmp_path / "animacion.mp4"
    avisos = []

    generar_video(
        estados,
        problema.mapa,
        destino,
        movimientos,
        fps=12,
        progreso=lambda escritos, total: avisos.append((escritos, total)),
    )

    assert destino.stat().st_size > 0
    assert avisos[-1] == (len(estados), len(estados))


def test_el_destino_por_defecto_vive_junto_a_los_resultados() -> None:
    assert CARPETA_ANIMACIONES == Path("resultados/animaciones")
