from pathlib import Path

from sokoban.reportes import NOMBRES_FIGURAS


RAIZ = Path(__file__).resolve().parents[1]


def test_readme_documenta_la_interfaz_y_los_artefactos_actuales() -> None:
    readme = (RAIZ / "README.md").read_text(encoding="utf-8")

    for comando in ("resolver", "experimentar", "graficar"):
        assert "python -m sokoban {}".format(comando) in readme
    for nombre in NOMBRES_FIGURAS:
        assert nombre in readme
    assert "Éxitos | Fracasos | Cortes" in readme
    assert "Los tiempos no son invariantes" in readme
    assert "PowerPoint se arma fuera del repositorio" in readme

    for referencia_obsoleta in (
        "main.py",
        "visualizar.py",
        "benchmark.py",
        "IDDFS",
        "Pygame",
        "manhattan_jugador",
    ):
        assert referencia_obsoleta not in readme


def test_notas_de_presentacion_tienen_exactamente_siete_secciones() -> None:
    notas = (RAIZ / "docs" / "notas_presentacion.md").read_text(
        encoding="utf-8"
    )
    secciones = [linea for linea in notas.splitlines() if linea.startswith("## ")]

    assert len(secciones) == 7
    assert all(
        seccion.startswith("## {}.".format(indice))
        for indice, seccion in enumerate(secciones, start=1)
    )
    assert "Éxitos | Fracasos | Cortes" in notas
    assert "fuera del repositorio" in notas
