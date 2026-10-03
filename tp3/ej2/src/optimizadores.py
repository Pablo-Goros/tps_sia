"""Compatibilidad: los optimizadores se implementan en el núcleo común."""
from tps_sia.tp3.shared.optimizers import Adam, Momentum, Optimizador, SGD, construir_optimizador

__all__ = ["Optimizador", "SGD", "Momentum", "Adam", "construir_optimizador"]
