"""TP5 interface over the TP3 MLP; all parameter updates remain in TP3."""
from pathlib import Path
import sys
import numpy as np

# Resolve the sibling package once, independent of the caller's directory.
_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from tp3.shared.mlp import MLP
from tp3.shared.optimizers import construir_optimizador


def validate_architecture(architecture):
    """Shared contract for trainable and persisted TP5 architectures."""
    if (not isinstance(architecture, list) or len(architecture) < 3
            or len(architecture) % 2 != 1
            or any(type(width) is not int or width < 1 for width in architecture)
            or architecture[0] != 35 or architecture[-1] != 35
            or architecture[len(architecture)//2] != 2):
        raise ValueError('Architecture must have positive integer widths, 35 inputs/outputs and a central two-unit code')


class Autoencoder:
    def __init__(self, architecture=None, seed=42, learning_rate=0.001,
                 batch_size=32, optimizer=None):
        architecture = [35, 20, 2, 20, 35] if architecture is None else architecture
        validate_architecture(architecture)
        config = optimizer or {'name': 'adam', 'learning_rate': learning_rate}
        self._mlp = MLP(architecture, activacion='tanh', salida='logistica',
                        semilla=seed, tamano_lote=batch_size,
                        optimizador=construir_optimizador(config))

    @staticmethod
    def _input(values, width):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != width or not np.isfinite(values).all():
            raise ValueError(f'Expected finite nonempty (N,{width}) data')
        return values

    def encode(self, x):
        values = self._input(x, 35)
        for w, b in list(zip(self._mlp.pesos, self._mlp.biases))[:len(self._mlp.pesos)//2]:
            values = self._mlp.activacion.theta(values @ w + b)
        return values

    def decode(self, z):
        values = self._input(z, 2)
        layers = list(zip(self._mlp.pesos, self._mlp.biases))
        for index in range(len(layers)//2, len(layers)):
            w, b = layers[index]
            activation = self._mlp.activacion_salida if index == len(layers)-1 else self._mlp.activacion
            values = activation.theta(values @ w + b)
        return values

    def reconstruct(self, x):
        return self._mlp.predecir(self._input(x, 35))

    def train(self, x, epochs):
        x = self._input(x, 35)
        return self._mlp.entrenar(x, x, epocas=epochs, epsilon=0)

    def set_learning_rate(self, learning_rate):
        """Change the next update's rate without rebuilding optimizer state."""
        optimizer = self._mlp.optimizador
        config = optimizer.configuracion()
        key = 'eta' if 'nombre' in config else 'learning_rate'
        config[key] = learning_rate
        construir_optimizador(config)  # Validate before mutating the live optimizer.
        setattr(optimizer, key, learning_rate)

    @property
    def configuration(self):
        return {'architecture': list(self._mlp.arquitectura),
                'activation': self._mlp.activacion.nombre, 'output': self._mlp.salida,
                'beta': self._mlp.beta, 'batch_size': self._mlp.tamano_lote,
                'optimizer': self._mlp.optimizador.configuracion(),
                'initialization': self._mlp.inicializacion, 'l2': self._mlp.l2,
                'augmentation': self._mlp.augmentation, 'batch_norm': self._mlp.batch_norm}

    @property
    def history(self):
        return self._mlp.historia.to_dict()

    def save(self, path):
        self._mlp.guardar(path)

    @classmethod
    def load(cls, path):
        model = cls.__new__(cls)
        model._mlp = MLP.cargar(path)
        validate_architecture(model._mlp.arquitectura)
        if model._mlp.activacion.nombre != 'tanh' or model._mlp.salida != 'logistica' or model._mlp.batch_norm:
            raise ValueError('Expected a tanh/logistic autoencoder without batch normalization')
        return model
