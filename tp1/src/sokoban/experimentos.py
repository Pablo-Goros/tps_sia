"""Ejecución reproducible y resumen de experimentos de Sokoban."""

from __future__ import annotations

import csv
import json
import random
import statistics
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .busqueda import ALGORITMOS, Resultado, ejecutar_busqueda
from .estado import ABREVIATURAS
from .heuristicas import HEURISTICAS, HEURISTICA_POR_DEFECTO
from .nivel import cargar_nivel
from .problema import ProblemaSokoban


REPETICIONES_REQUERIDAS = 5

COLUMNAS_EJECUCIONES: Tuple[str, ...] = (
    "orden_ejecucion",
    "repeticion",
    "nivel",
    "algoritmo",
    "heuristica",
    "estado",
    "motivo",
    "costo",
    "movimientos",
    "nodos_expandidos",
    "nodos_frontera",
    "max_nodos_frontera",
    "tiempo_segundos",
)

COLUMNAS_RESUMEN: Tuple[str, ...] = (
    "nivel",
    "algoritmo",
    "heuristica",
    "repeticiones",
    "exitos",
    "fracasos",
    "cortes",
    "tasa_exito",
    "costo_promedio",
    "nodos_expandidos_promedio",
    "nodos_frontera_promedio",
    "max_nodos_frontera_promedio",
    "tiempo_promedio_segundos",
    "tiempo_desvio_segundos",
)

Fila = Dict[str, object]
Informador = Callable[[str], None]


class ErrorConfiguracion(ValueError):
    """Indica que la configuración no puede ejecutarse de forma segura."""


@dataclass(frozen=True)
class NivelConfigurado:
    nombre: str
    ruta: Path


@dataclass(frozen=True)
class ConfiguracionExperimentos:
    ruta: Path
    semilla: int
    repeticiones: int
    niveles: Tuple[NivelConfigurado, ...]
    algoritmos: Tuple[str, ...]
    heuristicas: Tuple[str, ...]
    max_expandidos: Optional[int]
    salida_ejecuciones: Path
    salida_resumen: Path
    salida_figuras: Path


@dataclass(frozen=True)
class NivelPreparado:
    nombre: str
    ruta: Path
    problema: ProblemaSokoban


@dataclass(frozen=True)
class EjecucionPlanificada:
    nivel: NivelPreparado
    algoritmo: str
    heuristica: Optional[str]
    repeticion: int


@dataclass(frozen=True)
class ResultadoExperimentos:
    ejecuciones: Tuple[Fila, ...]
    resumen: Tuple[Fila, ...]
    salida_ejecuciones: Path
    salida_resumen: Path


def cargar_configuracion(
    ruta: str | PathLike[str],
) -> ConfiguracionExperimentos:
    """Carga y valida una configuración, resolviendo sus rutas relativas."""
    ruta_configuracion = Path(ruta).expanduser().resolve()
    try:
        with ruta_configuracion.open("r", encoding="utf-8") as archivo:
            datos = json.load(archivo)
    except OSError as error:
        raise ErrorConfiguracion(
            "no se pudo leer {}: {}".format(ruta_configuracion, error)
        ) from error
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ErrorConfiguracion(
            "{} no contiene JSON válido: {}".format(ruta_configuracion, error)
        ) from error

    objeto = _validar_objeto(
        datos,
        "la configuración",
        requeridas={
            "semilla",
            "repeticiones",
            "niveles",
            "algoritmos",
            "heuristicas",
            "salidas",
        },
        opcionales={"max_expandidos"},
    )
    semilla = _validar_entero(objeto["semilla"], "semilla")
    repeticiones = _validar_entero_positivo(
        objeto["repeticiones"], "repeticiones"
    )
    if repeticiones != REPETICIONES_REQUERIDAS:
        raise ErrorConfiguracion(
            "repeticiones debe valer {}".format(REPETICIONES_REQUERIDAS)
        )

    base = ruta_configuracion.parent
    niveles = _validar_niveles(objeto["niveles"], base)
    algoritmos = _validar_opciones(
        objeto["algoritmos"], "algoritmos", ALGORITMOS
    )
    heuristicas = _validar_opciones(
        objeto["heuristicas"], "heuristicas", HEURISTICAS
    )
    if any(ALGORITMOS[algoritmo].usa_heuristica for algoritmo in algoritmos):
        if not heuristicas:
            raise ErrorConfiguracion(
                "heuristicas no puede estar vacía si hay búsquedas informadas"
            )

    max_expandidos_dato = objeto.get("max_expandidos")
    max_expandidos = (
        None
        if max_expandidos_dato is None
        else _validar_entero_positivo(max_expandidos_dato, "max_expandidos")
    )

    salidas = _validar_objeto(
        objeto["salidas"],
        "salidas",
        requeridas={"ejecuciones", "resumen"},
        opcionales={"figuras"},
    )
    salida_ejecuciones = _resolver_ruta(
        salidas["ejecuciones"], base, "salidas.ejecuciones"
    )
    salida_resumen = _resolver_ruta(
        salidas["resumen"], base, "salidas.resumen"
    )
    salida_figuras = _resolver_ruta(
        salidas.get("figuras", "resultados/figuras"),
        base,
        "salidas.figuras",
    )
    for nombre, salida in (
        ("salidas.ejecuciones", salida_ejecuciones),
        ("salidas.resumen", salida_resumen),
    ):
        if salida.suffix.lower() != ".csv":
            raise ErrorConfiguracion("{} debe terminar en .csv".format(nombre))
        if salida == ruta_configuracion:
            raise ErrorConfiguracion(
                "{} no puede sobrescribir la configuración".format(nombre)
            )
    if salida_ejecuciones == salida_resumen:
        raise ErrorConfiguracion("las dos salidas CSV deben ser diferentes")
    if salida_figuras in (salida_ejecuciones, salida_resumen):
        raise ErrorConfiguracion(
            "salidas.figuras debe ser diferente de las salidas CSV"
        )
    if salida_figuras == ruta_configuracion:
        raise ErrorConfiguracion(
            "salidas.figuras no puede sobrescribir la configuración"
        )

    return ConfiguracionExperimentos(
        ruta=ruta_configuracion,
        semilla=semilla,
        repeticiones=repeticiones,
        niveles=niveles,
        algoritmos=algoritmos,
        heuristicas=heuristicas,
        max_expandidos=max_expandidos,
        salida_ejecuciones=salida_ejecuciones,
        salida_resumen=salida_resumen,
        salida_figuras=salida_figuras,
    )


def preparar_niveles(
    configuracion: ConfiguracionExperimentos,
) -> Tuple[NivelPreparado, ...]:
    """Carga mapas y precalcula distancias antes de medir las búsquedas."""
    preparados = []
    rutas_niveles = {nivel.ruta for nivel in configuracion.niveles}
    for salida in (
        configuracion.salida_ejecuciones,
        configuracion.salida_resumen,
        configuracion.salida_figuras,
    ):
        if salida in rutas_niveles:
            raise ErrorConfiguracion(
                "una salida CSV no puede sobrescribir un archivo de nivel"
            )

    for nivel in configuracion.niveles:
        try:
            mapa, estado_inicial = cargar_nivel(nivel.ruta)
            problema = ProblemaSokoban(mapa, estado_inicial)
        except (OSError, ValueError) as error:
            raise ErrorConfiguracion(
                "no se pudo preparar el nivel {} ({}): {}".format(
                    nivel.nombre, nivel.ruta, error
                )
            ) from error
        preparados.append(NivelPreparado(nivel.nombre, nivel.ruta, problema))
    return tuple(preparados)


def planificar_ejecuciones(
    configuracion: ConfiguracionExperimentos,
    niveles: Sequence[NivelPreparado],
) -> List[EjecucionPlanificada]:
    """Construye todas las repeticiones y baraja únicamente su orden."""
    ejecuciones = []
    for nivel in niveles:
        for algoritmo in configuracion.algoritmos:
            heuristicas: Iterable[Optional[str]]
            if ALGORITMOS[algoritmo].usa_heuristica:
                heuristicas = configuracion.heuristicas
            else:
                heuristicas = (None,)
            for heuristica in heuristicas:
                for repeticion in range(1, configuracion.repeticiones + 1):
                    ejecuciones.append(
                        EjecucionPlanificada(
                            nivel=nivel,
                            algoritmo=algoritmo,
                            heuristica=heuristica,
                            repeticion=repeticion,
                        )
                    )

    generador = random.Random(configuracion.semilla)
    generador.shuffle(ejecuciones)
    return ejecuciones


def ejecutar_plan(
    configuracion: ConfiguracionExperimentos,
    plan: Sequence[EjecucionPlanificada],
    informar: Optional[Informador] = None,
) -> List[Fila]:
    """Ejecuta un plan creando bookkeeping y cachés nuevos en cada corrida."""
    filas = []
    total = len(plan)
    for orden, ejecucion in enumerate(plan, start=1):
        problema = ejecucion.nivel.problema
        resultado = ejecutar_busqueda(
            ejecucion.algoritmo,
            problema.estado_inicial,
            problema.mapa,
            heuristica=ejecucion.heuristica or HEURISTICA_POR_DEFECTO,
            max_nodos=configuracion.max_expandidos,
            problema=problema,
        )
        filas.append(_crear_fila(orden, ejecucion, resultado))
        if informar is not None:
            informar(
                "[{}/{}] {} / {} / {} / repetición {}: {}".format(
                    orden,
                    total,
                    ejecucion.nivel.nombre,
                    ejecucion.algoritmo,
                    ejecucion.heuristica or "-",
                    ejecucion.repeticion,
                    resultado.estado,
                )
            )
    return filas


def resumir_ejecuciones(ejecuciones: Sequence[Mapping[str, object]]) -> List[Fila]:
    """Agrupa resultados aplicando las semánticas estadísticas del TP."""
    grupos: Dict[Tuple[str, str, str], List[Mapping[str, object]]] = {}
    for fila in ejecuciones:
        clave = (
            str(fila["nivel"]),
            str(fila["algoritmo"]),
            str(fila["heuristica"]),
        )
        grupos.setdefault(clave, []).append(fila)

    resumen = []
    for (nivel, algoritmo, heuristica), filas in sorted(grupos.items()):
        exitosas = [fila for fila in filas if fila["estado"] == "exito"]
        fracasos = sum(fila["estado"] == "fracaso" for fila in filas)
        cortes = sum(fila["estado"] == "corte" for fila in filas)
        tiempos = [_numero(fila["tiempo_segundos"]) for fila in filas]
        costos = [_numero(fila["costo"]) for fila in exitosas]
        resumen.append(
            {
                "nivel": nivel,
                "algoritmo": algoritmo,
                "heuristica": heuristica,
                "repeticiones": len(filas),
                "exitos": len(exitosas),
                "fracasos": fracasos,
                "cortes": cortes,
                "tasa_exito": _redondear(len(exitosas) / len(filas)),
                "costo_promedio": (
                    _redondear(statistics.fmean(costos)) if costos else ""
                ),
                "nodos_expandidos_promedio": _promedio(
                    filas, "nodos_expandidos"
                ),
                "nodos_frontera_promedio": _promedio(filas, "nodos_frontera"),
                "max_nodos_frontera_promedio": _promedio(
                    filas, "max_nodos_frontera"
                ),
                "tiempo_promedio_segundos": _redondear(statistics.fmean(tiempos)),
                "tiempo_desvio_segundos": _redondear(
                    statistics.stdev(tiempos) if len(tiempos) > 1 else 0.0
                ),
            }
        )
    return resumen


def guardar_csv(
    ruta: Path,
    columnas: Sequence[str],
    filas: Sequence[Mapping[str, object]],
) -> None:
    try:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with ruta.open("w", encoding="utf-8", newline="") as archivo:
            escritor = csv.DictWriter(
                archivo, fieldnames=columnas, lineterminator="\n"
            )
            escritor.writeheader()
            escritor.writerows(filas)
    except OSError as error:
        raise ErrorConfiguracion(
            "no se pudo escribir {}: {}".format(ruta, error)
        ) from error


def ejecutar_desde_configuracion(
    ruta: str | PathLike[str],
    informar: Optional[Informador] = None,
) -> ResultadoExperimentos:
    """Ejecuta el flujo completo y escribe los CSV crudo y resumido."""
    configuracion = cargar_configuracion(ruta)
    niveles = preparar_niveles(configuracion)
    plan = planificar_ejecuciones(configuracion, niveles)
    ejecuciones = ejecutar_plan(configuracion, plan, informar)
    resumen = resumir_ejecuciones(ejecuciones)
    guardar_csv(
        configuracion.salida_ejecuciones,
        COLUMNAS_EJECUCIONES,
        ejecuciones,
    )
    guardar_csv(configuracion.salida_resumen, COLUMNAS_RESUMEN, resumen)
    return ResultadoExperimentos(
        ejecuciones=tuple(ejecuciones),
        resumen=tuple(resumen),
        salida_ejecuciones=configuracion.salida_ejecuciones,
        salida_resumen=configuracion.salida_resumen,
    )


def _crear_fila(
    orden: int,
    ejecucion: EjecucionPlanificada,
    resultado: Resultado,
) -> Fila:
    return {
        "orden_ejecucion": orden,
        "repeticion": ejecucion.repeticion,
        "nivel": ejecucion.nivel.nombre,
        "algoritmo": ejecucion.algoritmo,
        "heuristica": ejecucion.heuristica or "",
        "estado": resultado.estado,
        "motivo": resultado.motivo or "",
        "costo": resultado.costo if resultado.costo is not None else "",
        "movimientos": "".join(
            ABREVIATURAS[movimiento] for movimiento in resultado.movimientos
        ),
        "nodos_expandidos": resultado.nodos_expandidos,
        "nodos_frontera": resultado.nodos_frontera,
        "max_nodos_frontera": resultado.max_nodos_frontera,
        "tiempo_segundos": resultado.tiempo_segundos,
    }


def _validar_objeto(
    valor: object,
    contexto: str,
    requeridas: set[str],
    opcionales: Optional[set[str]] = None,
) -> Mapping[str, object]:
    if not isinstance(valor, dict):
        raise ErrorConfiguracion("{} debe ser un objeto JSON".format(contexto))
    opcionales = opcionales or set()
    faltantes = requeridas - set(valor)
    desconocidas = set(valor) - requeridas - opcionales
    if faltantes:
        raise ErrorConfiguracion(
            "faltan campos en {}: {}".format(contexto, ", ".join(sorted(faltantes)))
        )
    if desconocidas:
        raise ErrorConfiguracion(
            "campos desconocidos en {}: {}".format(
                contexto, ", ".join(sorted(desconocidas))
            )
        )
    return valor


def _validar_entero(valor: object, nombre: str) -> int:
    if isinstance(valor, bool) or not isinstance(valor, int):
        raise ErrorConfiguracion("{} debe ser un número entero".format(nombre))
    return valor


def _validar_entero_positivo(valor: object, nombre: str) -> int:
    entero = _validar_entero(valor, nombre)
    if entero <= 0:
        raise ErrorConfiguracion("{} debe ser mayor que cero".format(nombre))
    return entero


def _validar_texto(valor: object, nombre: str) -> str:
    if not isinstance(valor, str) or not valor.strip():
        raise ErrorConfiguracion("{} debe ser texto no vacío".format(nombre))
    return valor


def _resolver_ruta(valor: object, base: Path, nombre: str) -> Path:
    texto = _validar_texto(valor, nombre)
    ruta = Path(texto).expanduser()
    if not ruta.is_absolute():
        ruta = base / ruta
    return ruta.resolve()


def _validar_niveles(valor: object, base: Path) -> Tuple[NivelConfigurado, ...]:
    if not isinstance(valor, list) or not valor:
        raise ErrorConfiguracion("niveles debe ser una lista no vacía")
    niveles = []
    nombres = set()
    rutas = set()
    for indice, dato in enumerate(valor):
        contexto = "niveles[{}]".format(indice)
        nivel = _validar_objeto(
            dato, contexto, requeridas={"nombre", "ruta"}
        )
        nombre = _validar_texto(nivel["nombre"], "{}.nombre".format(contexto))
        ruta = _resolver_ruta(nivel["ruta"], base, "{}.ruta".format(contexto))
        if nombre in nombres:
            raise ErrorConfiguracion("nombre de nivel repetido: {}".format(nombre))
        if ruta in rutas:
            raise ErrorConfiguracion("ruta de nivel repetida: {}".format(ruta))
        nombres.add(nombre)
        rutas.add(ruta)
        niveles.append(NivelConfigurado(nombre, ruta))
    return tuple(niveles)


def _validar_opciones(
    valor: object,
    nombre: str,
    disponibles: Mapping[str, object],
) -> Tuple[str, ...]:
    if not isinstance(valor, list):
        raise ErrorConfiguracion("{} debe ser una lista".format(nombre))
    opciones = []
    for indice, dato in enumerate(valor):
        opcion = _validar_texto(dato, "{}[{}]".format(nombre, indice))
        if opcion not in disponibles:
            raise ErrorConfiguracion(
                "{} desconocido: {!r}; opciones: {}".format(
                    nombre[:-1], opcion, ", ".join(disponibles)
                )
            )
        if opcion in opciones:
            raise ErrorConfiguracion(
                "{} no puede contener valores repetidos: {}".format(nombre, opcion)
            )
        opciones.append(opcion)
    if nombre == "algoritmos" and not opciones:
        raise ErrorConfiguracion("algoritmos debe ser una lista no vacía")
    return tuple(opciones)


def _numero(valor: object) -> float:
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        raise ValueError(
            "se esperaba un valor numérico, se recibió {!r}".format(valor)
        )
    return float(valor)


def _promedio(filas: Sequence[Mapping[str, object]], columna: str) -> float:
    return _redondear(statistics.fmean(_numero(fila[columna]) for fila in filas))


def _redondear(valor: float) -> float:
    return round(valor, 9)


__all__ = [
    "COLUMNAS_EJECUCIONES",
    "COLUMNAS_RESUMEN",
    "ConfiguracionExperimentos",
    "EjecucionPlanificada",
    "ErrorConfiguracion",
    "NivelConfigurado",
    "NivelPreparado",
    "REPETICIONES_REQUERIDAS",
    "ResultadoExperimentos",
    "cargar_configuracion",
    "ejecutar_desde_configuracion",
    "ejecutar_plan",
    "guardar_csv",
    "planificar_ejecuciones",
    "preparar_niveles",
    "resumir_ejecuciones",
]
