"""Animacion de una solucion como GIF, video o secuencia de PNG.

Los caminos largos son el caso interesante: DFS devuelve soluciones de miles de
movimientos. Por eso los frames se generan de a uno (nunca se arma la lista
completa en memoria) y el video se escribe por streaming hacia ffmpeg.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

from .estado import Direccion, Estado, Mapa
from .problema import ProblemaSokoban

CELDA = 44
MARGEN = 14
ALTO_ENCABEZADO = 46
ALTO_PIE = 26

MAX_FRAMES_GIF = 400
MS_POR_PASO = 300

FPS_POR_DEFECTO = 12
FPS_CAMINOS_LARGOS = 60
UMBRAL_CAMINO_LARGO = 600

EXTENSIONES_VIDEO: Tuple[str, ...] = (".mp4", ".mov", ".mkv", ".webm")

CARPETA_ANIMACIONES = Path("resultados/animaciones")

FONDO = (24, 26, 32)
PISO = (54, 58, 70)
PARED = (96, 102, 120)
OBJETIVO = (196, 154, 62)
CAJA = (200, 118, 70)
CAJA_LISTA = (94, 168, 106)
JUGADOR = (86, 142, 214)
TEXTO = (232, 234, 240)
TEXTO_TENUE = (150, 156, 172)
BARRA = (86, 142, 214)
BARRA_FONDO = (54, 58, 70)

Progreso = Callable[[int, int], None]


class ErrorAnimacion(ValueError):
    """Indica que no se puede construir la animacion pedida."""


class FfmpegNoDisponible(ErrorAnimacion):
    """ffmpeg no esta instalado o no esta en el PATH."""


def _pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:  # pragma: no cover - depende del entorno
        raise ErrorAnimacion(
            "Pillow no esta instalado; ejecute "
            "'python -m pip install -e .[visual]'"
        ) from error
    return Image, ImageDraw, ImageFont


def estados_de_la_solucion(
    problema: ProblemaSokoban,
    movimientos: Sequence[Direccion],
) -> List[Estado]:
    """Reproduce la solucion con las mismas reglas que uso la busqueda."""
    estados = [problema.estado_inicial]
    actual = problema.estado_inicial
    for paso, movimiento in enumerate(movimientos, start=1):
        siguiente = problema.aplicar_accion(actual, movimiento)
        if siguiente is None:
            raise ErrorAnimacion(
                "el movimiento {} de la solucion es ilegal: {}".format(
                    paso, movimiento
                )
            )
        estados.append(siguiente)
        actual = siguiente
    return estados


def indices_de_frames(
    cantidad_estados: int,
    submuestreo: int = 1,
    max_frames: Optional[int] = None,
) -> List[int]:
    """Elige que pasos dibujar, conservando siempre el primero y el ultimo."""
    if cantidad_estados <= 0:
        raise ErrorAnimacion("no hay estados para animar")
    if submuestreo < 1:
        raise ErrorAnimacion("el submuestreo debe ser mayor o igual que uno")
    if max_frames is not None and max_frames < 2:
        raise ErrorAnimacion("el tope de frames debe ser al menos dos")

    paso = submuestreo
    if max_frames is not None:
        necesario = -(-cantidad_estados // max_frames)
        paso = max(paso, necesario)

    indices = list(range(0, cantidad_estados, paso))
    if indices[-1] != cantidad_estados - 1:
        indices.append(cantidad_estados - 1)
    return indices


def submuestreo_efectivo(
    cantidad_estados: int,
    submuestreo: int = 1,
    max_frames: Optional[int] = None,
) -> int:
    """Cada cuantos pasos se dibuja realmente un frame."""
    indices = indices_de_frames(cantidad_estados, submuestreo, max_frames)
    return -(-cantidad_estados // len(indices))


def fps_sugerido(cantidad_frames: int) -> int:
    """Los caminos largos se ven a mas cuadros por segundo o duran una eternidad."""
    if cantidad_frames >= UMBRAL_CAMINO_LARGO:
        return FPS_CAMINOS_LARGOS
    return FPS_POR_DEFECTO


def duracion_formateada(cantidad_frames: int, fps: int) -> str:
    if fps <= 0:
        raise ErrorAnimacion("los cuadros por segundo deben ser positivos")
    segundos = cantidad_frames / fps
    if segundos < 60:
        return "{:.1f} s".format(segundos)
    return "{:d}:{:04.1f} min".format(int(segundos // 60), segundos % 60)


def hay_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def es_video(ruta: os.PathLike | str) -> bool:
    return Path(ruta).suffix.lower() in EXTENSIONES_VIDEO


def dimensiones(mapa: Mapa, celda: int = CELDA, encabezado: bool = True) -> Tuple[int, int]:
    ancho = mapa.ancho * celda + 2 * MARGEN
    alto = mapa.alto * celda + 2 * MARGEN
    if encabezado:
        alto += ALTO_ENCABEZADO + ALTO_PIE
    return ancho, alto


def dibujar_frame(
    estado: Estado,
    mapa: Mapa,
    paso: int = 0,
    total: int = 0,
    accion: Optional[Direccion] = None,
    titulo: str = "",
    subtitulo: str = "",
    celda: int = CELDA,
    encabezado: bool = True,
):
    """Dibuja un unico estado y devuelve la imagen de Pillow."""
    Image, ImageDraw, _ = _pillow()
    ancho, alto = dimensiones(mapa, celda, encabezado)
    imagen = Image.new("RGB", (ancho, alto), FONDO)
    dibujo = ImageDraw.Draw(imagen)

    desplazamiento_y = MARGEN + (ALTO_ENCABEZADO if encabezado else 0)
    if encabezado:
        _dibujar_encabezado(dibujo, ancho, titulo, paso, total, accion)

    for y in range(mapa.alto):
        for x in range(mapa.ancho):
            posicion = (x, y)
            if posicion not in mapa.pisos and posicion not in mapa.paredes:
                continue
            izquierda = MARGEN + x * celda
            arriba = desplazamiento_y + y * celda
            caja = (izquierda, arriba, izquierda + celda - 2, arriba + celda - 2)

            if posicion in mapa.paredes:
                dibujo.rectangle(caja, fill=PARED)
                continue

            dibujo.rectangle(caja, fill=PISO)
            if posicion in mapa.objetivos:
                centro = (
                    izquierda + celda // 2 - 1,
                    arriba + celda // 2 - 1,
                )
                radio = max(3, celda // 8)
                dibujo.ellipse(
                    (
                        centro[0] - radio,
                        centro[1] - radio,
                        centro[0] + radio,
                        centro[1] + radio,
                    ),
                    fill=OBJETIVO,
                )

    borde = max(3, celda // 10)
    for posicion in estado.cajas:
        x, y = posicion
        izquierda = MARGEN + x * celda + borde
        arriba = desplazamiento_y + y * celda + borde
        dibujo.rectangle(
            (
                izquierda,
                arriba,
                izquierda + celda - 2 * borde - 2,
                arriba + celda - 2 * borde - 2,
            ),
            fill=CAJA_LISTA if posicion in mapa.objetivos else CAJA,
        )

    x, y = estado.jugador
    centro_x = MARGEN + x * celda + celda // 2 - 1
    centro_y = desplazamiento_y + y * celda + celda // 2 - 1
    radio = max(4, celda // 3)
    dibujo.ellipse(
        (centro_x - radio, centro_y - radio, centro_x + radio, centro_y + radio),
        fill=JUGADOR,
    )

    if encabezado and subtitulo:
        fuente = _fuente(13)
        ancho_texto = dibujo.textlength(subtitulo, font=fuente)
        dibujo.text(
            ((ancho - ancho_texto) / 2, alto - ALTO_PIE + 4),
            subtitulo,
            font=fuente,
            fill=TEXTO_TENUE,
        )

    return imagen


def iterar_frames(
    estados: Sequence[Estado],
    mapa: Mapa,
    movimientos: Sequence[Direccion] = (),
    titulo: str = "",
    subtitulo: str = "",
    celda: int = CELDA,
    encabezado: bool = True,
    submuestreo: int = 1,
    max_frames: Optional[int] = None,
) -> Iterator:
    """Genera los frames de a uno: el consumidor decide si los acumula."""
    indices = indices_de_frames(len(estados), submuestreo, max_frames)
    total = len(estados) - 1
    for indice in indices:
        accion = movimientos[indice - 1] if 0 < indice <= len(movimientos) else None
        yield dibujar_frame(
            estados[indice],
            mapa,
            paso=indice,
            total=total,
            accion=accion,
            titulo=titulo,
            subtitulo=subtitulo,
            celda=celda,
            encabezado=encabezado,
        )


def generar_gif(
    estados: Sequence[Estado],
    mapa: Mapa,
    destino: os.PathLike | str,
    movimientos: Sequence[Direccion] = (),
    ms_por_paso: int = MS_POR_PASO,
    titulo: str = "",
    subtitulo: str = "",
    celda: int = CELDA,
    encabezado: bool = True,
    submuestreo: int = 1,
    max_frames: Optional[int] = MAX_FRAMES_GIF,
) -> Path:
    """Guarda un GIF animado, en modo paleta para que no pese de mas."""
    ruta = Path(destino)
    _asegurar_directorio(ruta)
    frames = iterar_frames(
        estados,
        mapa,
        movimientos,
        titulo=titulo,
        subtitulo=subtitulo,
        celda=celda,
        encabezado=encabezado,
        submuestreo=submuestreo,
        max_frames=max_frames,
    )
    primero = next(frames)
    paleta = [imagen.convert("P", palette=1) for imagen in frames]
    primero.convert("P", palette=1).save(
        ruta,
        save_all=True,
        append_images=paleta,
        duration=ms_por_paso,
        loop=0,
        optimize=True,
    )
    return ruta


def generar_video(
    estados: Sequence[Estado],
    mapa: Mapa,
    destino: os.PathLike | str,
    movimientos: Sequence[Direccion] = (),
    fps: Optional[int] = None,
    titulo: str = "",
    subtitulo: str = "",
    celda: int = CELDA,
    encabezado: bool = True,
    submuestreo: int = 1,
    max_frames: Optional[int] = None,
    progreso: Optional[Progreso] = None,
) -> Path:
    """Escribe un video H.264 mandando los frames crudos a ffmpeg.

    Sin tope de frames: es la salida pensada para las soluciones de miles de
    movimientos, donde un GIF tendria que saltear pasos.
    """
    if not hay_ffmpeg():
        raise FfmpegNoDisponible(
            "ffmpeg no esta en el PATH; instalelo o genere un GIF"
        )

    ruta = Path(destino)
    _asegurar_directorio(ruta)
    cantidad = len(indices_de_frames(len(estados), submuestreo, max_frames))
    cuadros = fps or fps_sugerido(cantidad)
    ancho, alto = dimensiones(mapa, celda, encabezado)

    comando = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "{}x{}".format(ancho, alto),
        "-r",
        str(cuadros),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        # H.264 exige dimensiones pares.
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(ruta),
    ]

    proceso = subprocess.Popen(comando, stdin=subprocess.PIPE)
    escritos = 0
    try:
        for imagen in iterar_frames(
            estados,
            mapa,
            movimientos,
            titulo=titulo,
            subtitulo=subtitulo,
            celda=celda,
            encabezado=encabezado,
            submuestreo=submuestreo,
            max_frames=max_frames,
        ):
            proceso.stdin.write(imagen.tobytes())
            escritos += 1
            if progreso is not None and escritos % 200 == 0:
                progreso(escritos, cantidad)
    finally:
        if proceso.stdin is not None:
            proceso.stdin.close()
        codigo = proceso.wait()

    if progreso is not None:
        progreso(escritos, cantidad)
    if codigo != 0:
        raise ErrorAnimacion("ffmpeg termino con codigo {}".format(codigo))
    return ruta


def guardar_frames(
    estados: Sequence[Estado],
    mapa: Mapa,
    carpeta: os.PathLike | str,
    movimientos: Sequence[Direccion] = (),
    titulo: str = "",
    subtitulo: str = "",
    celda: int = CELDA,
    encabezado: bool = True,
    submuestreo: int = 1,
    max_frames: Optional[int] = None,
) -> Tuple[Path, ...]:
    """Guarda un PNG por frame, para elegir cuadros sueltos en la presentacion."""
    destino = Path(carpeta)
    destino.mkdir(parents=True, exist_ok=True)
    indices = indices_de_frames(len(estados), submuestreo, max_frames)
    ancho_nombre = len(str(indices[-1]))
    rutas = []
    frames = iterar_frames(
        estados,
        mapa,
        movimientos,
        titulo=titulo,
        subtitulo=subtitulo,
        celda=celda,
        encabezado=encabezado,
        submuestreo=submuestreo,
        max_frames=max_frames,
    )
    for indice, imagen in zip(indices, frames):
        archivo = destino / "paso_{}.png".format(str(indice).zfill(ancho_nombre))
        imagen.save(archivo)
        rutas.append(archivo)
    return tuple(rutas)


def _dibujar_encabezado(
    dibujo,
    ancho: int,
    titulo: str,
    paso: int,
    total: int,
    accion: Optional[Direccion],
) -> None:
    fuente_titulo = _fuente(16)
    fuente_contador = _fuente(13)

    contador = "paso {} / {}".format(paso, total)
    if accion:
        contador += "  ·  {}".format(accion)
    ancho_contador = dibujo.textlength(contador, font=fuente_contador)
    dibujo.text(
        (ancho - MARGEN - ancho_contador, MARGEN - 2),
        contador,
        font=fuente_contador,
        fill=TEXTO_TENUE,
    )

    disponible = ancho - 2 * MARGEN - ancho_contador - 12
    dibujo.text(
        (MARGEN, MARGEN - 4),
        _recortar(dibujo, titulo, fuente_titulo, disponible),
        font=fuente_titulo,
        fill=TEXTO,
    )

    y_barra = MARGEN + 24
    dibujo.rectangle(
        (MARGEN, y_barra, ancho - MARGEN, y_barra + 4), fill=BARRA_FONDO
    )
    if total > 0:
        avance = (ancho - 2 * MARGEN) * paso / total
        if avance > 0:
            dibujo.rectangle(
                (MARGEN, y_barra, MARGEN + avance, y_barra + 4), fill=BARRA
            )


def _recortar(dibujo, texto: str, fuente, ancho_maximo: float) -> str:
    if ancho_maximo <= 0 or dibujo.textlength(texto, font=fuente) <= ancho_maximo:
        return texto
    recortado = texto
    while recortado and dibujo.textlength(recortado + "…", font=fuente) > ancho_maximo:
        recortado = recortado[:-1]
    return recortado + "…"


def _fuente(tamanio: int):
    _, _, ImageFont = _pillow()
    for nombre in (
        "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(nombre, tamanio)
        except OSError:
            continue
    return ImageFont.load_default()


def _asegurar_directorio(ruta: Path) -> None:
    carpeta = ruta.parent
    if carpeta and not carpeta.is_dir():
        carpeta.mkdir(parents=True, exist_ok=True)


__all__ = [
    "CARPETA_ANIMACIONES",
    "CELDA",
    "ErrorAnimacion",
    "EXTENSIONES_VIDEO",
    "FfmpegNoDisponible",
    "MAX_FRAMES_GIF",
    "MS_POR_PASO",
    "dibujar_frame",
    "dimensiones",
    "duracion_formateada",
    "es_video",
    "estados_de_la_solucion",
    "fps_sugerido",
    "generar_gif",
    "generar_video",
    "guardar_frames",
    "hay_ffmpeg",
    "indices_de_frames",
    "iterar_frames",
    "submuestreo_efectivo",
]
