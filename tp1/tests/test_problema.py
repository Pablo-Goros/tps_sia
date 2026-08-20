import pytest

from sokoban import ProblemaSokoban, parsear_tablero, resolver
from sokoban.estado import Estado, Mapa


def test_sucesores_respetan_el_orden_configurado_y_cuestan_uno() -> None:
    mapa, estado = parsear_tablero(
        """\
#######
# .   #
#     #
#  @  #
#  $  #
#     #
#######
"""
    )
    orden = ("derecha", "izquierda", "abajo", "arriba")
    problema = ProblemaSokoban(mapa, estado, orden)

    sucesores = problema.generar_sucesores(estado, podar_deadlocks=False)

    assert [accion for _, accion, _ in sucesores] == list(orden)
    assert {costo for _, _, costo in sucesores} == {1}


def test_generar_sucesores_no_muta_el_estado_recibido() -> None:
    mapa, estado = parsear_tablero(
        """\
#####
# . #
# $ #
# @ #
#####
"""
    )
    problema = ProblemaSokoban(mapa, estado)
    copia = Estado(estado.jugador, estado.cajas)

    sucesores = problema.generar_sucesores(estado)

    assert estado == copia
    assert estado.cajas is copia.cajas
    assert all(sucesor is not estado for sucesor, _, _ in sucesores)
    assert all(isinstance(sucesor.cajas, frozenset) for sucesor, _, _ in sucesores)


def test_distancias_y_celdas_muertas_quedan_dentro_de_los_pisos() -> None:
    mapa, estado = parsear_tablero(
        """\
#######
#  .  #
#  $  #
#  @  #
#######
"""
    )
    problema = ProblemaSokoban(mapa, estado)

    assert problema.celdas_muertas <= mapa.pisos
    assert set(problema.distancias_empujes) == mapa.objetivos
    assert all(
        set(distancias) <= mapa.pisos
        for distancias in problema.distancias_empujes.values()
    )

    objetivo = next(iter(mapa.objetivos))
    with pytest.raises(TypeError):
        problema.distancias_empujes[objetivo][objetivo] = 99


def test_poda_estatica_descarta_un_empuje_a_una_esquina_muerta() -> None:
    mapa, estado = parsear_tablero(
        """\
######
# .  #
#    #
# $@ #
######
"""
    )
    problema = ProblemaSokoban(mapa, estado)

    sin_poda = problema.generar_sucesores(estado, podar_deadlocks=False)
    con_poda = problema.generar_sucesores(estado, podar_deadlocks=True)

    assert "izquierda" in [accion for _, accion, _ in sin_poda]
    assert "izquierda" not in [accion for _, accion, _ in con_poda]
    assert (1, 3) in problema.celdas_muertas


def test_una_esquina_objetivo_no_se_poda() -> None:
    mapa, estado = parsear_tablero(
        """\
######
#    #
#    #
#.$@ #
######
"""
    )
    problema = ProblemaSokoban(mapa, estado)

    sucesores = problema.generar_sucesores(estado)
    estado_final = next(
        sucesor for sucesor, accion, _ in sucesores if accion == "izquierda"
    )

    assert estado_final.es_objetivo(mapa)
    assert (1, 3) not in problema.celdas_muertas


def _mapa_para_bloque_2x2(objetivos) -> Mapa:
    paredes = frozenset({(1, 0), (2, 0)})
    pisos = frozenset(
        (x, y)
        for y in range(3)
        for x in range(5)
        if (x, y) not in paredes
    )
    return Mapa(
        ancho=5,
        alto=3,
        pisos=pisos,
        paredes=paredes,
        objetivos=frozenset(objetivos),
    )


def test_bloque_2x2_fuera_de_objetivo_es_deadlock() -> None:
    mapa = _mapa_para_bloque_2x2({(1, 1), (3, 1)})
    anterior = Estado(jugador=(4, 1), cajas=frozenset({(1, 1), (3, 1)}))
    problema = ProblemaSokoban(mapa, anterior)

    nuevo = problema.aplicar_accion(anterior, "izquierda")

    assert nuevo is not None
    assert (2, 1) not in problema.celdas_muertas
    assert problema.es_deadlock(nuevo, anterior)


def test_bloque_2x2_con_todas_las_cajas_en_objetivos_no_se_poda() -> None:
    mapa = _mapa_para_bloque_2x2({(1, 1), (2, 1)})
    anterior = Estado(jugador=(4, 1), cajas=frozenset({(1, 1), (3, 1)}))
    problema = ProblemaSokoban(mapa, anterior)

    sucesores = problema.generar_sucesores(anterior)
    nuevo = next(
        sucesor for sucesor, accion, _ in sucesores if accion == "izquierda"
    )

    assert nuevo.es_objetivo(mapa)
    assert not problema.es_deadlock(nuevo, anterior)


@pytest.mark.parametrize("algoritmo", ["bfs", "dfs", "greedy", "astar"])
def test_todos_los_algoritmos_usan_los_sucesores_del_problema(
    algoritmo: str, monkeypatch
) -> None:
    mapa, estado = parsear_tablero("#####\n#@$.#\n#####")
    problema = ProblemaSokoban(mapa, estado)
    original = ProblemaSokoban.generar_sucesores
    llamadas = 0

    def generar_controlado(self, estado, podar_deadlocks=True):
        nonlocal llamadas
        llamadas += 1
        return original(self, estado, podar_deadlocks)

    monkeypatch.setattr(ProblemaSokoban, "generar_sucesores", generar_controlado)

    resultado = resolver(problema, algoritmo)

    assert resultado.exito
    assert resultado.costo == 1
    assert llamadas >= 1
