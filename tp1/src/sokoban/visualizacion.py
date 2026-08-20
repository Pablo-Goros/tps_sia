"""Salidas visuales: consola, GIF, video y ventana. Solo consume Estado y Mapa."""

from __future__ import annotations

import itertools
import os
from typing import Callable, Iterator, List, Optional, Sequence

from .estado import (
    CAJA,
    CAJA_EN_OBJETIVO,
    JUGADOR,
    JUGADOR_EN_OBJETIVO,
    OBJETIVO,
    PARED,
    PISO,
    Estado,
    Mapa,
    aplicar_accion,
)

CELDA = 44
MARGEN = 14
ALTO_ENCABEZADO = 46
ALTO_PIE = 26

COLORES = {
    "fondo": (20, 22, 28),
    "panel": (29, 33, 43),
    "pared": (58, 65, 82),
    "pared_luz": (78, 87, 108),
    "piso_a": (233, 231, 224),
    "piso_b": (225, 223, 214),
    "objetivo": (217, 164, 65),
    "caja": (192, 138, 78),
    "caja_borde": (138, 95, 49),
    "caja_ok": (90, 169, 106),
    "caja_ok_borde": (61, 122, 76),
    "resaltado": (247, 208, 90),
    "jugador": (74, 134, 232),
    "jugador_borde": (38, 84, 163),
    "texto": (238, 238, 240),
    "texto_tenue": (150, 157, 173),
    "barra": (74, 134, 232),
    "barra_fondo": (52, 58, 72),
}

# Si no hay ninguna, Pillow cae a su fuente bitmap por defecto.
_RUTAS_FUENTE = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/SFNSRounded.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)

MAX_FRAMES = 400

FPS_POR_DEFECTO = 12
FPS_CAMINOS_LARGOS = 60
UMBRAL_CAMINO_LARGO = 600

EXTENSIONES_VIDEO = (".mp4", ".mkv", ".webm", ".mov", ".avi")


# --- Consola ---


def render_texto(estado: Estado, mapa: Mapa) -> str:
    """Vuelve a la notacion XSB del nivel."""
    filas = []
    for y in range(mapa.alto):
        fila = []
        for x in range(mapa.ancho):
            pos = (x, y)
            if pos in mapa.paredes:
                fila.append(PARED)
            elif pos == estado.jugador:
                fila.append(JUGADOR_EN_OBJETIVO if pos in mapa.objetivos else JUGADOR)
            elif pos in estado.cajas:
                fila.append(CAJA_EN_OBJETIVO if pos in mapa.objetivos else CAJA)
            elif pos in mapa.objetivos:
                fila.append(OBJETIVO)
            else:
                fila.append(PISO)
        filas.append("".join(fila).rstrip())
    return "\n".join(filas)


def estados_desde_camino(
    estado_inicial: Estado, mapa: Mapa, camino: Sequence[str]
) -> List[Estado]:
    """Reproduce las acciones con `aplicar_accion`, lo que valida el camino."""
    estados = [estado_inicial]
    actual = estado_inicial
    for i, accion in enumerate(camino):
        siguiente = aplicar_accion(actual, mapa, accion)
        if siguiente is None:
            raise ValueError(
                "el camino es invalido: la accion {} ({}) no se puede aplicar".format(
                    i + 1, accion
                )
            )
        actual = siguiente
        estados.append(actual)
    return estados


def reproducir_en_consola(
    estados: Sequence[Estado],
    mapa: Mapa,
    camino: Sequence[str] = (),
    limpiar: bool = False,
    pausa: float = 0.0,
) -> None:
    import time

    for i, estado in enumerate(estados):
        if limpiar:
            os.system("cls" if os.name == "nt" else "clear")
        if i == 0:
            print("Estado inicial:")
        else:
            print("Paso {}/{}: {}".format(i, len(estados) - 1, camino[i - 1]))
        print(render_texto(estado, mapa))
        print()
        if pausa:
            time.sleep(pausa)


# --- Dibujo de frames (Pillow) ---


def _fuente(tamano: int):
    from PIL import ImageFont

    for ruta in _RUTAS_FUENTE:
        if os.path.isfile(ruta):
            try:
                return ImageFont.truetype(ruta, tamano)
            except OSError:
                continue
    return ImageFont.load_default()


def dibujar_frame(
    estado: Estado,
    mapa: Mapa,
    titulo: str = "",
    subtitulo: str = "",
    paso: Optional[int] = None,
    total: Optional[int] = None,
    accion: Optional[str] = None,
    resaltar: Optional[Sequence] = None,
    celda: int = CELDA,
    encabezado: bool = True,
):
    """`resaltar` marca la caja recien empujada, para seguir la accion."""
    from PIL import Image, ImageDraw

    hay_encabezado = encabezado and paso is not None
    alto_arriba = ALTO_ENCABEZADO if hay_encabezado else MARGEN
    alto_abajo = ALTO_PIE if hay_encabezado else MARGEN

    ancho = mapa.ancho * celda + 2 * MARGEN
    alto = mapa.alto * celda + alto_arriba + alto_abajo
    imagen = Image.new("RGB", (ancho, alto), COLORES["fondo"])
    dibujo = ImageDraw.Draw(imagen)
    resaltar = set(resaltar or ())

    def caja_de(pos, margen=0):
        x, y = pos
        return [
            MARGEN + x * celda + margen,
            alto_arriba + y * celda + margen,
            MARGEN + (x + 1) * celda - 1 - margen,
            alto_arriba + (y + 1) * celda - 1 - margen,
        ]

    for y in range(mapa.alto):
        for x in range(mapa.ancho):
            pos = (x, y)
            rect = caja_de(pos)
            if pos in mapa.paredes:
                dibujo.rectangle(rect, fill=COLORES["pared"], outline=COLORES["fondo"])
                dibujo.line(
                    [rect[0] + 2, rect[1] + 1, rect[2] - 2, rect[1] + 1],
                    fill=COLORES["pared_luz"],
                )
            else:
                color = (
                    COLORES["piso_a"]
                    if (x + y) % 2 == 0
                    else COLORES["piso_b"]
                )
                dibujo.rectangle(rect, fill=color)

    for objetivo in mapa.objetivos:
        x_inicial, y_inicial, x_final, y_final = caja_de(
            objetivo, margen=celda // 3
        )
        centro_x = (x_inicial + x_final) / 2
        centro_y = (y_inicial + y_final) / 2
        radio = (x_final - x_inicial) / 2
        dibujo.polygon(
            [
                (centro_x, centro_y - radio),
                (centro_x + radio, centro_y),
                (centro_x, centro_y + radio),
                (centro_x - radio, centro_y),
            ],
            outline=COLORES["objetivo"],
            width=max(2, celda // 20),
        )

    for pos in estado.cajas:
        en_objetivo = pos in mapa.objetivos
        relleno = COLORES["caja_ok"] if en_objetivo else COLORES["caja"]
        borde = COLORES["caja_ok_borde"] if en_objetivo else COLORES["caja_borde"]
        if pos in resaltar:
            borde = COLORES["resaltado"]
        dibujo.rounded_rectangle(
            caja_de(pos, margen=max(3, celda // 10)),
            radius=max(2, celda // 10),
            fill=relleno,
            outline=borde,
            width=max(2, celda // 16),
        )
        dibujo.rectangle(
            caja_de(pos, margen=max(3, celda // 5)), outline=borde, width=1
        )

    dibujo.ellipse(
        caja_de(estado.jugador, margen=max(5, celda // 6)),
        fill=COLORES["jugador"],
        outline=COLORES["jugador_borde"],
        width=max(2, celda // 20),
    )

    if hay_encabezado:
        _dibujar_encabezado(
            dibujo,
            imagen.width,
            alto,
            titulo,
            subtitulo,
            paso,
            total,
            accion,
            estado,
            mapa,
        )

    return imagen


def _recortar(dibujo, texto: str, fuente, ancho_max: float) -> str:
    if ancho_max <= 0 or dibujo.textlength(texto, font=fuente) <= ancho_max:
        return texto
    recortado = texto
    while recortado and dibujo.textlength(recortado + "...", font=fuente) > ancho_max:
        recortado = recortado[:-1]
    return recortado.rstrip() + "..."


def _dibujar_encabezado(
    dibujo, ancho, alto, titulo, subtitulo, paso, total, accion, estado, mapa
) -> None:
    fuente = _fuente(13)
    fuente_chica = _fuente(11)

    dibujo.rectangle([0, 0, ancho, ALTO_ENCABEZADO - 8], fill=COLORES["panel"])

    etiqueta = "paso {} / {}".format(paso, total)
    ancho_etiqueta = dibujo.textlength(etiqueta, font=fuente_chica)
    dibujo.text(
        (ancho - MARGEN, 10),
        etiqueta,
        font=fuente_chica,
        fill=COLORES["texto_tenue"],
        anchor="ra",
    )

    if titulo:
        disponible = ancho - 2 * MARGEN - ancho_etiqueta - 12
        dibujo.text(
            (MARGEN, 9),
            _recortar(dibujo, titulo, fuente, disponible),
            font=fuente,
            fill=COLORES["texto"],
        )

    y_barra = ALTO_ENCABEZADO - 13
    dibujo.rectangle(
        [MARGEN, y_barra, ancho - MARGEN, y_barra + 3], fill=COLORES["barra_fondo"]
    )
    if total:
        avance = (ancho - 2 * MARGEN) * paso / total
        if avance > 0:
            dibujo.rectangle(
                [MARGEN, y_barra, MARGEN + avance, y_barra + 3], fill=COLORES["barra"]
            )

    y_pie = alto - ALTO_PIE + 7
    izquierda = "accion: {}".format(accion) if accion else "estado inicial"
    derecha = "cajas en objetivo: {}/{}".format(
        len(estado.cajas & mapa.objetivos), len(mapa.objetivos)
    )
    dibujo.text(
        (MARGEN, y_pie), izquierda, font=fuente_chica, fill=COLORES["texto_tenue"]
    )
    dibujo.text(
        (ancho - MARGEN, y_pie),
        derecha,
        font=fuente_chica,
        fill=COLORES["texto_tenue"],
        anchor="ra",
    )

    if subtitulo:
        usado = (
            dibujo.textlength(izquierda, font=fuente_chica)
            + dibujo.textlength(derecha, font=fuente_chica)
            + 4 * MARGEN
        )
        if dibujo.textlength(subtitulo, font=fuente_chica) <= ancho - usado:
            dibujo.text(
                (ancho / 2, y_pie),
                subtitulo,
                font=fuente_chica,
                fill=COLORES["texto_tenue"],
                anchor="ma",
            )


def submuestreo_efectivo(
    cantidad_estados: int, submuestreo: int = 1, max_frames: Optional[int] = MAX_FRAMES
) -> int:
    """Cada cuantos pasos dibujar para no pasarse de `max_frames`."""
    paso = max(1, submuestreo)
    if max_frames and cantidad_estados / paso > max_frames:
        paso = -(-cantidad_estados // max_frames)
    return paso


def indices_de_frames(
    cantidad_estados: int,
    submuestreo: int = 1,
    max_frames: Optional[int] = MAX_FRAMES,
) -> List[int]:
    if cantidad_estados <= 0:
        raise ValueError("no hay estados para dibujar")
    paso = submuestreo_efectivo(cantidad_estados, submuestreo, max_frames)
    indices = list(range(0, cantidad_estados, paso))
    if indices[-1] != cantidad_estados - 1:
        indices.append(cantidad_estados - 1)
    return indices


def iterar_frames(
    estados: Sequence[Estado],
    mapa: Mapa,
    camino: Sequence[str] = (),
    titulo: str = "",
    celda: int = CELDA,
    encabezado: bool = True,
    submuestreo: int = 1,
    subtitulo: str = "",
    max_frames: Optional[int] = MAX_FRAMES,
) -> Iterator:
    """Generador y no lista: un camino de DFS puede tener decenas de miles de frames."""
    for i in indices_de_frames(len(estados), submuestreo, max_frames):
        anterior = estados[i - 1] if i > 0 else None
        movidas = (estados[i].cajas - anterior.cajas) if anterior else frozenset()
        yield dibujar_frame(
            estados[i],
            mapa,
            titulo=titulo,
            subtitulo=subtitulo,
            paso=i,
            total=len(estados) - 1,
            accion=camino[i - 1] if 0 < i <= len(camino) else None,
            resaltar=movidas,
            celda=celda,
            encabezado=encabezado,
        )


def frames_de_solucion(
    estados: Sequence[Estado],
    mapa: Mapa,
    camino: Sequence[str] = (),
    titulo: str = "",
    celda: int = CELDA,
    encabezado: bool = True,
    submuestreo: int = 1,
    subtitulo: str = "",
    max_frames: Optional[int] = MAX_FRAMES,
) -> List:
    return list(
        iterar_frames(
            estados,
            mapa,
            camino,
            titulo,
            celda,
            encabezado,
            submuestreo,
            subtitulo,
            max_frames,
        )
    )


# --- GIF ---


def generar_gif(
    estados: Sequence[Estado],
    mapa: Mapa,
    ruta: str,
    camino: Sequence[str] = (),
    ms_por_paso: int = 300,
    titulo: str = "",
    celda: int = CELDA,
    encabezado: bool = True,
    submuestreo: int = 1,
    subtitulo: str = "",
    max_frames: Optional[int] = MAX_FRAMES,
) -> str:
    """Guarda la solucion como GIF animado. Requiere Pillow."""
    frames = frames_de_solucion(
        estados,
        mapa,
        camino,
        titulo,
        celda,
        encabezado,
        submuestreo,
        subtitulo,
        max_frames,
    )

    duraciones = [ms_por_paso] * len(frames)
    duraciones[0] = max(ms_por_paso, 900)
    duraciones[-1] = max(ms_por_paso, 1800)

    # Pocos colores: pasar a paleta achica el archivo mas de la mitad.
    paleta = [frame.convert("P", palette=1, colors=64) for frame in frames]
    paleta[0].save(
        ruta,
        save_all=True,
        append_images=paleta[1:],
        duration=duraciones,
        loop=0,
        optimize=True,
    )
    return ruta


def guardar_frames(
    estados: Sequence[Estado],
    mapa: Mapa,
    carpeta: str,
    camino: Sequence[str] = (),
    titulo: str = "",
    celda: int = CELDA,
    encabezado: bool = True,
    subtitulo: str = "",
    max_frames: Optional[int] = MAX_FRAMES,
) -> List[str]:
    """Un PNG por paso."""
    os.makedirs(carpeta, exist_ok=True)
    rutas = []
    for i, frame in enumerate(
        iterar_frames(
            estados, mapa, camino, titulo, celda, encabezado, 1, subtitulo, max_frames
        )
    ):
        destino = os.path.join(carpeta, "paso_{:03d}.png".format(i))
        frame.save(destino)
        rutas.append(destino)
    return rutas


# --- Video (ffmpeg) ---


class FfmpegNoDisponible(RuntimeError):
    pass


def hay_ffmpeg() -> bool:
    import shutil

    return shutil.which("ffmpeg") is not None


def fps_sugerido(cantidad_frames: int) -> int:
    return (
        FPS_POR_DEFECTO
        if cantidad_frames <= UMBRAL_CAMINO_LARGO
        else FPS_CAMINOS_LARGOS
    )


def duracion_formateada(cantidad_frames: int, fps: int) -> str:
    segundos = cantidad_frames / float(fps)
    minutos, resto = divmod(segundos, 60)
    return "{:d}:{:04.1f} min".format(int(minutos), resto)


def generar_video(
    estados: Sequence[Estado],
    mapa: Mapa,
    ruta: str,
    camino: Sequence[str] = (),
    fps: Optional[int] = None,
    titulo: str = "",
    subtitulo: str = "",
    celda: int = CELDA,
    encabezado: bool = True,
    submuestreo: int = 1,
    max_frames: Optional[int] = None,
    progreso: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Video H.264, un frame por movimiento y sin tope por defecto.

    Los frames se escriben de a uno al stdin de ffmpeg, asi que la memoria es
    constante aunque el camino tenga decenas de miles de movimientos.
    """
    import subprocess

    if not hay_ffmpeg():
        raise FfmpegNoDisponible(
            "ffmpeg no esta instalado: no se puede generar el video. "
            "Instalalo con 'brew install ffmpeg' (macOS) o "
            "'apt install ffmpeg' (Linux), o genera un GIF en su lugar."
        )

    total = len(indices_de_frames(len(estados), submuestreo, max_frames))
    frames = iterar_frames(
        estados,
        mapa,
        camino,
        titulo,
        celda,
        encabezado,
        submuestreo,
        subtitulo,
        max_frames,
    )

    primero = next(frames)
    ancho, alto = primero.size
    fps = fps or fps_sugerido(total)

    comando = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", "{}x{}".format(ancho, alto),
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # H.264 necesita lados pares
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        ruta,
    ]

    proceso = subprocess.Popen(comando, stdin=subprocess.PIPE)
    try:
        for i, frame in enumerate(itertools.chain([primero], frames), start=1):
            proceso.stdin.write(frame.tobytes())
            if progreso is not None and (i % 2000 == 0 or i == total):
                progreso(i, total)
    except BrokenPipeError as error:
        raise RuntimeError("ffmpeg corto la escritura del video") from error
    finally:
        if proceso.stdin:
            proceso.stdin.close()
        codigo = proceso.wait()

    if codigo != 0:
        raise RuntimeError("ffmpeg termino con codigo {}".format(codigo))
    return ruta


# --- Ventana animada (pygame, opcional) ---


def animar(
    estados: Sequence[Estado],
    mapa: Mapa,
    ms_por_paso: int = 300,
    titulo: str = "Sokoban",
) -> None:
    try:
        import pygame
    except ImportError as error:
        raise SystemExit(
            "pygame no esta instalado. Instalalo con 'pip install pygame' "
            "o usa --gif / --pasos."
        ) from error

    pygame.init()
    pantalla = pygame.display.set_mode((mapa.ancho * CELDA, mapa.alto * CELDA))
    pygame.display.set_caption(titulo)
    reloj = pygame.time.Clock()

    def dibujar(estado: Estado) -> None:
        pantalla.fill(COLORES["fondo"])
        for y in range(mapa.alto):
            for x in range(mapa.ancho):
                rect = pygame.Rect(x * CELDA, y * CELDA, CELDA, CELDA)
                if (x, y) in mapa.paredes:
                    pygame.draw.rect(pantalla, COLORES["pared"], rect)
                else:
                    pygame.draw.rect(pantalla, COLORES["piso_a"], rect)
        for objetivo in mapa.objetivos:
            centro = (
                objetivo[0] * CELDA + CELDA // 2,
                objetivo[1] * CELDA + CELDA // 2,
            )
            pygame.draw.circle(pantalla, COLORES["objetivo"], centro, CELDA // 6)
        for caja in estado.cajas:
            color = COLORES["caja_ok"] if caja in mapa.objetivos else COLORES["caja"]
            rect = pygame.Rect(
                caja[0] * CELDA + 5,
                caja[1] * CELDA + 5,
                CELDA - 10,
                CELDA - 10,
            )
            pygame.draw.rect(pantalla, color, rect)
        centro = (
            estado.jugador[0] * CELDA + CELDA // 2,
            estado.jugador[1] * CELDA + CELDA // 2,
        )
        pygame.draw.circle(pantalla, COLORES["jugador"], centro, CELDA // 3)
        pygame.display.flip()

    for estado in estados:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                return
        dibujar(estado)
        pygame.time.wait(ms_por_paso)
        reloj.tick(60)

    pygame.time.wait(1500)
    pygame.quit()
