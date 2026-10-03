"""Compatibilidad: los optimizadores se implementan en el núcleo común."""
from tps_sia.tp3.shared.optimizers import Optimizador, SGD, construir_optimizador

__all__ = ["Optimizador", "SGD", "construir_optimizador"]
