"""Generación de las figuras comparativas del trabajo práctico."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from os import PathLike
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .busqueda import ALGORITMOS
from .experimentos import (
    COLUMNAS_RESUMEN,
    ConfiguracionExperimentos,
    cargar_configuracion,
)


NOMBRES_FIGURAS: Tuple[str, ...] = (
    "tiempo_promedio.png",
    "nodos_expandidos.png",
    "max_nodos_frontera.png",
    "costo_solucion.png",
)
UMBRAL_ESCALA_LOGARITMICA = 100.0
SUBCARPETA_FILTRADAS = "filtradas"

COLORES: Tuple[str, ...] = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#ECA82C",
    "#B279A2",
    "#E45756",
)


class ErrorReporte(ValueError):
    """Indica que los datos no permiten construir un reporte válido."""


@dataclass(frozen=True)
class MedicionResumen:
    nivel: str
    algoritmo: str
    heuristica: str
    repeticiones: int
    exitos: int
    fracasos: int
    cortes: int
    tasa_exito: float
    costo_promedio: Optional[float]
    nodos_expandidos_promedio: float
    nodos_frontera_promedio: float
    max_nodos_frontera_promedio: float
    tiempo_promedio_segundos: float
    tiempo_desvio_segundos: float

    @property
    def clave(self) -> Tuple[str, str, str]:
        return self.nivel, self.algoritmo, self.heuristica


@dataclass(frozen=True)
class Serie:
    algoritmo: str
    heuristica: str
    etiqueta: str
    color: str

    @property
    def clave(self) -> Tuple[str, str]:
        return self.algoritmo, self.heuristica


@dataclass(frozen=True)
class EspecificacionFigura:
    nombre: str
    campo: str
    titulo: str
    etiqueta_y: str
    barras_error: Optional[str] = None
    escala_logaritmica_nodos: bool = False


ESPECIFICACIONES: Tuple[EspecificacionFigura, ...] = (
    EspecificacionFigura(
        nombre="tiempo_promedio.png",
        campo="tiempo_promedio_segundos",
        titulo="Tiempo promedio por configuración",
        etiqueta_y="Tiempo promedio (segundos)",
        barras_error="tiempo_desvio_segundos",
    ),
    EspecificacionFigura(
        nombre="nodos_expandidos.png",
        campo="nodos_expandidos_promedio",
        titulo="Nodos expandidos por configuración",
        etiqueta_y="Nodos expandidos promedio",
        escala_logaritmica_nodos=True,
    ),
    EspecificacionFigura(
        nombre="max_nodos_frontera.png",
        campo="max_nodos_frontera_promedio",
        titulo="Máximo de nodos en frontera por configuración",
        etiqueta_y="Máximo de nodos en frontera promedio",
        escala_logaritmica_nodos=True,
    ),
    EspecificacionFigura(
        nombre="costo_solucion.png",
        campo="costo_promedio",
        titulo="Costo promedio de las soluciones",
        etiqueta_y="Costo promedio (movimientos)",
    ),
)


def cargar_resumen(ruta: Path) -> Tuple[MedicionResumen, ...]:
    """Carga el CSV resumido y valida columnas, tipos y estados."""
    try:
        with ruta.open("r", encoding="utf-8", newline="") as archivo:
            lector = csv.DictReader(archivo)
            if tuple(lector.fieldnames or ()) != COLUMNAS_RESUMEN:
                raise ErrorReporte(
                    "{} no tiene las columnas esperadas".format(ruta)
                )
            filas = tuple(
                _convertir_fila(fila, numero_fila)
                for numero_fila, fila in enumerate(lector, start=2)
            )
    except OSError as error:
        raise ErrorReporte("no se pudo leer {}: {}".format(ruta, error)) from error
    except (UnicodeError, csv.Error) as error:
        raise ErrorReporte("{} no es un CSV válido: {}".format(ruta, error)) from error

    if not filas:
        raise ErrorReporte("{} no contiene mediciones".format(ruta))
    return filas


def validar_resumen(
    configuracion: ConfiguracionExperimentos,
    mediciones: Sequence[MedicionResumen],
) -> None:
    """Comprueba que el resumen corresponde exactamente a la configuración."""
    por_clave: Dict[Tuple[str, str, str], MedicionResumen] = {}
    for medicion in mediciones:
        if medicion.clave in por_clave:
            raise ErrorReporte(
                "el resumen contiene una configuración repetida: {}".format(
                    medicion.clave
                )
            )
        por_clave[medicion.clave] = medicion

    esperadas = {
        (nivel.nombre, serie.algoritmo, serie.heuristica)
        for nivel in configuracion.niveles
        for serie in crear_series(configuracion)
    }
    recibidas = set(por_clave)
    if esperadas != recibidas:
        faltantes = sorted(esperadas - recibidas)
        adicionales = sorted(recibidas - esperadas)
        partes = []
        if faltantes:
            partes.append("faltan {}".format(faltantes))
        if adicionales:
            partes.append("sobran {}".format(adicionales))
        raise ErrorReporte(
            "el resumen no coincide con la configuración: " + "; ".join(partes)
        )

    for medicion in mediciones:
        if medicion.repeticiones != configuracion.repeticiones:
            raise ErrorReporte(
                "{} tiene {} repeticiones; se esperaban {}".format(
                    medicion.clave,
                    medicion.repeticiones,
                    configuracion.repeticiones,
                )
            )


def crear_series(configuracion: ConfiguracionExperimentos) -> Tuple[Serie, ...]:
    """Crea las series en un orden estable derivado de la configuración."""
    series = []
    for algoritmo in configuracion.algoritmos:
        if ALGORITMOS[algoritmo].usa_heuristica:
            heuristicas: Iterable[str] = configuracion.heuristicas
        else:
            heuristicas = ("",)
        for heuristica in heuristicas:
            etiqueta = ALGORITMOS[algoritmo].nombre
            if heuristica:
                etiqueta += " · " + _titulo(heuristica)
            series.append(
                Serie(
                    algoritmo=algoritmo,
                    heuristica=heuristica,
                    etiqueta=etiqueta,
                    color=COLORES[len(series) % len(COLORES)],
                )
            )
    return tuple(series)


def filtrar_configuracion(
    configuracion: ConfiguracionExperimentos,
    niveles: Optional[Sequence[str]] = None,
    algoritmos: Optional[Sequence[str]] = None,
    heuristicas: Optional[Sequence[str]] = None,
) -> ConfiguracionExperimentos:
    """Restringe la configuración a un subconjunto de sus propias opciones.

    Cada selección se ordena como en la configuración, no como en la línea de
    comandos, para que dos corridas con el mismo subconjunto produzcan figuras
    idénticas.
    """
    nombres_niveles = tuple(nivel.nombre for nivel in configuracion.niveles)
    elegidos = _seleccionar(niveles, nombres_niveles, "nivel")
    seleccion_algoritmos = _seleccionar(
        algoritmos, configuracion.algoritmos, "algoritmo"
    )
    seleccion_heuristicas = _seleccionar(
        heuristicas, configuracion.heuristicas, "heurística"
    )

    return replace(
        configuracion,
        niveles=tuple(
            nivel for nivel in configuracion.niveles if nivel.nombre in elegidos
        ),
        algoritmos=seleccion_algoritmos,
        heuristicas=seleccion_heuristicas,
    )


def filtrar_mediciones(
    configuracion: ConfiguracionExperimentos,
    mediciones: Sequence[MedicionResumen],
) -> Tuple[MedicionResumen, ...]:
    """Conserva únicamente las mediciones que la configuración describe."""
    claves = {
        (nivel.nombre, serie.algoritmo, serie.heuristica)
        for nivel in configuracion.niveles
        for serie in crear_series(configuracion)
    }
    return tuple(medicion for medicion in mediciones if medicion.clave in claves)


def requiere_escala_logaritmica(valores: Sequence[float]) -> bool:
    """Decide la escala únicamente a partir de una razón numérica estable."""
    positivos = [valor for valor in valores if valor > 0 and math.isfinite(valor)]
    if len(positivos) < 2:
        return False
    return max(positivos) / min(positivos) >= UMBRAL_ESCALA_LOGARITMICA


def generar_figuras(
    configuracion: ConfiguracionExperimentos,
    mediciones: Sequence[MedicionResumen],
) -> Tuple[Path, ...]:
    """Genera exactamente las cuatro figuras requeridas por el informe."""
    validar_resumen(configuracion, mediciones)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ErrorReporte(
            "Matplotlib no está instalado; ejecute "
            "'python -m pip install -e .[analysis]'"
        ) from error

    try:
        configuracion.salida_figuras.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ErrorReporte(
            "no se pudo crear {}: {}".format(configuracion.salida_figuras, error)
        ) from error

    niveles = tuple(nivel.nombre for nivel in configuracion.niveles)
    series = crear_series(configuracion)
    por_clave = {medicion.clave: medicion for medicion in mediciones}
    destinos = []

    for especificacion in ESPECIFICACIONES:
        figura, eje = plt.subplots(figsize=(12.5, 7.2))
        todos_los_valores = _dibujar_barras(
            eje,
            niveles,
            series,
            por_clave,
            especificacion,
        )
        if (
            especificacion.escala_logaritmica_nodos
            and requiere_escala_logaritmica(todos_los_valores)
        ):
            eje.set_yscale("log")
            etiqueta_y = especificacion.etiqueta_y + " (escala logarítmica)"
        else:
            eje.set_ylim(bottom=0)
            etiqueta_y = especificacion.etiqueta_y

        eje.set_title(especificacion.titulo, fontsize=15, pad=14)
        eje.set_xlabel("Nivel")
        eje.set_ylabel(etiqueta_y)
        eje.set_xticks(range(len(niveles)))
        eje.set_xticklabels(_titulo(nivel) for nivel in niveles)
        eje.grid(axis="y", alpha=0.25, linestyle="--")
        eje.set_axisbelow(True)
        figura.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(3, len(series)),
            title="Configuración",
            frameon=False,
        )
        if especificacion.barras_error:
            figura.text(
                0.5,
                0.135,
                "Barras de error: desvío estándar de cinco repeticiones.",
                ha="center",
                fontsize=9,
            )
        margen_inferior = 0.19 if especificacion.barras_error else 0.13
        figura.tight_layout(rect=(0.03, margen_inferior, 0.99, 0.98))

        destino = configuracion.salida_figuras / especificacion.nombre
        try:
            figura.savefig(
                destino,
                dpi=160,
                bbox_inches="tight",
                metadata={"Software": "sokoban-sia"},
            )
        except OSError as error:
            raise ErrorReporte(
                "no se pudo escribir {}: {}".format(destino, error)
            ) from error
        finally:
            plt.close(figura)
        destinos.append(destino)

    return tuple(destinos)


def generar_desde_configuracion(
    ruta: str | PathLike[str],
    niveles: Optional[Sequence[str]] = None,
    algoritmos: Optional[Sequence[str]] = None,
    heuristicas: Optional[Sequence[str]] = None,
    salida_figuras: Optional[str | PathLike[str]] = None,
) -> Tuple[Path, ...]:
    """Genera las figuras, opcionalmente restringidas a un subconjunto."""
    configuracion = cargar_configuracion(ruta)
    mediciones = cargar_resumen(configuracion.salida_resumen)
    filtrada = filtrar_configuracion(
        configuracion,
        niveles=niveles,
        algoritmos=algoritmos,
        heuristicas=heuristicas,
    )
    filtrada = replace(
        filtrada,
        salida_figuras=destino_figuras(configuracion, filtrada, salida_figuras),
    )
    return generar_figuras(filtrada, filtrar_mediciones(filtrada, mediciones))


def destino_figuras(
    configuracion: ConfiguracionExperimentos,
    filtrada: ConfiguracionExperimentos,
    salida_figuras: Optional[str | PathLike[str]] = None,
) -> Path:
    """Elige dónde escribir las figuras sin pisar las del informe.

    Una corrida filtrada no describe la configuración completa, así que sus
    figuras van a una subcarpeta salvo que se pida un destino explícito.
    """
    if salida_figuras is not None:
        return Path(salida_figuras).expanduser().resolve()
    if (
        filtrada.niveles == configuracion.niveles
        and filtrada.algoritmos == configuracion.algoritmos
        and filtrada.heuristicas == configuracion.heuristicas
    ):
        return configuracion.salida_figuras
    return configuracion.salida_figuras / SUBCARPETA_FILTRADAS


def _seleccionar(
    pedidos: Optional[Sequence[str]],
    disponibles: Sequence[str],
    etiqueta: str,
) -> Tuple[str, ...]:
    """Valida lo pedido contra lo disponible y respeta el orden de origen."""
    if pedidos is None:
        return tuple(disponibles)
    elegidos = set()
    for pedido in pedidos:
        if pedido not in disponibles:
            raise ErrorReporte(
                "{} fuera de la configuración: {!r}; disponibles: {}".format(
                    etiqueta, pedido, ", ".join(disponibles) or "ninguno"
                )
            )
        elegidos.add(pedido)
    if not elegidos:
        raise ErrorReporte(
            "la selección de {}s no puede estar vacía".format(etiqueta)
        )
    return tuple(opcion for opcion in disponibles if opcion in elegidos)


def _dibujar_barras(
    eje,
    niveles: Sequence[str],
    series: Sequence[Serie],
    por_clave: Dict[Tuple[str, str, str], MedicionResumen],
    especificacion: EspecificacionFigura,
) -> List[float]:
    ancho = min(0.8 / len(series), 0.18)
    valores_totales = []
    for indice, serie in enumerate(series):
        desplazamiento = (indice - (len(series) - 1) / 2) * ancho
        posiciones = [nivel + desplazamiento for nivel in range(len(niveles))]
        mediciones = [
            por_clave[(nivel, serie.algoritmo, serie.heuristica)]
            for nivel in niveles
        ]
        valores = [
            _valor_graficable(getattr(medicion, especificacion.campo))
            for medicion in mediciones
        ]
        errores = None
        if especificacion.barras_error:
            errores = [
                float(getattr(medicion, especificacion.barras_error))
                for medicion in mediciones
            ]
        eje.bar(
            posiciones,
            valores,
            width=ancho * 0.9,
            label=serie.etiqueta,
            color=serie.color,
            yerr=errores,
            capsize=3 if errores is not None else 0,
            error_kw={"elinewidth": 1, "capthick": 1},
        )
        valores_totales.extend(
            valor for valor in valores if math.isfinite(valor)
        )
    return valores_totales


def _convertir_fila(
    fila: Dict[str, str], numero_fila: int
) -> MedicionResumen:
    contexto = "fila {}".format(numero_fila)
    repeticiones = _entero_no_negativo(fila["repeticiones"], contexto)
    exitos = _entero_no_negativo(fila["exitos"], contexto)
    fracasos = _entero_no_negativo(fila["fracasos"], contexto)
    cortes = _entero_no_negativo(fila["cortes"], contexto)
    if exitos + fracasos + cortes != repeticiones:
        raise ErrorReporte(
            "{}: éxitos, fracasos y cortes no suman las repeticiones".format(
                contexto
            )
        )
    tasa_exito = _decimal_no_negativo(fila["tasa_exito"], contexto)
    tasa_esperada = exitos / repeticiones if repeticiones else 0.0
    if tasa_exito > 1 or not math.isclose(tasa_exito, tasa_esperada):
        raise ErrorReporte(
            "{}: tasa_exito no coincide con los éxitos y repeticiones".format(
                contexto
            )
        )

    costo_texto = fila["costo_promedio"].strip()
    costo = _decimal_no_negativo(costo_texto, contexto) if costo_texto else None
    if (exitos == 0) != (costo is None):
        raise ErrorReporte(
            "{}: costo_promedio debe existir únicamente si hubo éxitos".format(
                contexto
            )
        )

    return MedicionResumen(
        nivel=_texto(fila["nivel"], contexto),
        algoritmo=_texto(fila["algoritmo"], contexto),
        heuristica=fila["heuristica"].strip(),
        repeticiones=repeticiones,
        exitos=exitos,
        fracasos=fracasos,
        cortes=cortes,
        tasa_exito=tasa_exito,
        costo_promedio=costo,
        nodos_expandidos_promedio=_decimal_no_negativo(
            fila["nodos_expandidos_promedio"], contexto
        ),
        nodos_frontera_promedio=_decimal_no_negativo(
            fila["nodos_frontera_promedio"], contexto
        ),
        max_nodos_frontera_promedio=_decimal_no_negativo(
            fila["max_nodos_frontera_promedio"], contexto
        ),
        tiempo_promedio_segundos=_decimal_no_negativo(
            fila["tiempo_promedio_segundos"], contexto
        ),
        tiempo_desvio_segundos=_decimal_no_negativo(
            fila["tiempo_desvio_segundos"], contexto
        ),
    )


def _entero_no_negativo(texto: str, contexto: str) -> int:
    try:
        valor = int(texto)
    except ValueError as error:
        raise ErrorReporte(
            "{}: se esperaba un entero, se recibió {!r}".format(contexto, texto)
        ) from error
    if valor < 0:
        raise ErrorReporte("{}: los conteos no pueden ser negativos".format(contexto))
    return valor


def _decimal_no_negativo(texto: str, contexto: str) -> float:
    try:
        valor = float(texto)
    except ValueError as error:
        raise ErrorReporte(
            "{}: se esperaba un número, se recibió {!r}".format(contexto, texto)
        ) from error
    if not math.isfinite(valor) or valor < 0:
        raise ErrorReporte(
            "{}: los valores numéricos deben ser finitos y no negativos".format(
                contexto
            )
        )
    return valor


def _texto(texto: str, contexto: str) -> str:
    valor = texto.strip()
    if not valor:
        raise ErrorReporte("{}: hay un texto obligatorio vacío".format(contexto))
    return valor


def _valor_graficable(valor: Optional[float]) -> float:
    return math.nan if valor is None else valor


def _titulo(texto: str) -> str:
    return texto.replace("_", " ").capitalize()


__all__ = [
    "ErrorReporte",
    "MedicionResumen",
    "NOMBRES_FIGURAS",
    "SUBCARPETA_FILTRADAS",
    "Serie",
    "UMBRAL_ESCALA_LOGARITMICA",
    "cargar_resumen",
    "crear_series",
    "destino_figuras",
    "filtrar_configuracion",
    "filtrar_mediciones",
    "generar_desde_configuracion",
    "generar_figuras",
    "requiere_escala_logaritmica",
    "validar_resumen",
]
