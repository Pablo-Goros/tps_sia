"""MLP matricial con mini-batches, SGD y persistencia sin pickle.

Filas = muestras. W[l] tiene forma (entrada, salida), b[l] forma (salida,).
Softmax usa cross-entropy media por muestra. Logística usa medio error
cuadrático sumado sobre las salidas y promediado sobre las muestras.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time

import numpy as np

from ...ej1.src.activaciones import construir_activacion
from .optimizadores import Optimizador, SGD, construir_optimizador


@dataclass
class Historia:
    costo: list[float] = field(default_factory=list)
    mse: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)
    costo_validacion: list[float] = field(default_factory=list)
    mse_validacion: list[float] = field(default_factory=list)
    accuracy_validacion: list[float] = field(default_factory=list)
    norma_gradiente: list[float] = field(default_factory=list)
    epocas_corridas: int = 0
    tiempo_segundos: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class MLP:
    def __init__(self, arquitectura: list[int], activacion: str = "tanh",
                 salida: str = "softmax", eta: float = 0.01, beta: float = 1.0,
                 tamano_lote: int | None = 32, inicializacion: str = "auto",
                 semilla: int = 0, optimizador: Optimizador | None = None) -> None:
        if (len(arquitectura) < 2 or any(isinstance(n, (bool, np.bool_)) or
                not isinstance(n, (int, np.integer)) or n <= 0 for n in arquitectura)):
            raise ValueError("La arquitectura debe contener al menos dos tamaños enteros positivos")
        if activacion not in ("tanh", "relu"):
            raise ValueError("Las capas ocultas admiten tanh o relu")
        if salida not in ("softmax", "logistica"):
            raise ValueError("La salida debe ser softmax o logistica")
        if salida == "softmax" and arquitectura[-1] < 2:
            raise ValueError("Softmax necesita al menos dos salidas; para una use logistica")
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("beta debe ser positiva y finita")
        if tamano_lote is not None and (isinstance(tamano_lote, bool) or
                not isinstance(tamano_lote, (int, np.integer)) or tamano_lote <= 0):
            raise ValueError("tamano_lote debe ser un entero positivo o None (batch)")
        if inicializacion not in ("auto", "xavier", "he"):
            raise ValueError("La inicialización debe ser auto, xavier o he")
        self.arquitectura = [int(n) for n in arquitectura]
        self.activacion = construir_activacion(activacion, beta)
        self.salida = salida
        self.activacion_salida = construir_activacion("logistica", beta)
        self.beta = float(beta)
        self.tamano_lote = None if tamano_lote is None else int(tamano_lote)
        self.inicializacion = inicializacion
        self.optimizador = optimizador if optimizador is not None else SGD(eta)
        self.rng = np.random.default_rng(semilla)
        self.pesos: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for i, (n_in, n_out) in enumerate(zip(self.arquitectura, self.arquitectura[1:])):
            metodo = inicializacion
            if metodo == "auto":
                metodo = "he" if activacion == "relu" and i < len(arquitectura) - 2 else "xavier"
            escala = np.sqrt(2.0 / n_in) if metodo == "he" else np.sqrt(2.0 / (n_in + n_out))
            self.pesos.append(self.rng.normal(0, escala, size=(n_in, n_out)))
            self.biases.append(np.zeros(n_out))
        self.historia = Historia()

    @property
    def parametros(self) -> list[np.ndarray]:
        """Orden fijo para optimizadores y gradient checks: pesos, luego biases."""
        return self.pesos + self.biases

    def _entradas(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != self.arquitectura[0]:
            raise ValueError("X debe tener forma (N, n_entradas), con N > 0")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contiene valores no finitos")
        return X

    def _objetivos(self, y: np.ndarray, n: int) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if y.ndim == 1 and self.arquitectura[-1] == 1:
            y = y[:, None]
        if y.shape != (n, self.arquitectura[-1]):
            raise ValueError("y debe tener forma (N, n_salidas)")
        if not np.all(np.isfinite(y)) or np.any((y < 0) | (y > 1)):
            raise ValueError("Los objetivos deben ser finitos y estar en [0, 1]")
        if self.salida == "softmax" and not np.allclose(y.sum(axis=1), 1, rtol=0, atol=1e-7):
            raise ValueError("Cada fila de y debe sumar 1 para softmax (one-hot o distribución)")
        return y

    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activaciones, nets = [X], []
        for i, (w, b) in enumerate(zip(self.pesos, self.biases)):
            h = activaciones[-1] @ w + b
            nets.append(h)
            if i < len(self.pesos) - 1:
                o = self.activacion.theta(h)
            elif self.salida == "logistica":
                o = self.activacion_salida.theta(h)
            else:
                exp = np.exp(h - h.max(axis=1, keepdims=True))
                o = exp / exp.sum(axis=1, keepdims=True)
            activaciones.append(o)
        return activaciones, nets

    def predecir(self, X: np.ndarray) -> np.ndarray:
        """Devuelve las salidas con forma (N, n_salidas), incluso para una salida."""
        return self._forward(self._entradas(X))[0][-1]

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.predecir(X)

    def predecir_clases(self, X: np.ndarray) -> np.ndarray:
        o = self.predecir(X)
        return (o[:, 0] >= 0.5).astype(int) if o.shape[1] == 1 else o.argmax(axis=1)

    def _costo(self, y: np.ndarray, o: np.ndarray, h: np.ndarray) -> float:
        if self.salida == "logistica":
            return float(0.5 * np.mean(np.sum((o - y) ** 2, axis=1)))
        # Log-softmax estable: no se recortan probabilidades ni se altera su gradiente.
        z = h - h.max(axis=1, keepdims=True)
        log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-np.mean(np.sum(y * log_p, axis=1)))

    def costo(self, X: np.ndarray, y: np.ndarray) -> float:
        X = self._entradas(X)
        y = self._objetivos(y, len(X))
        a, h = self._forward(X)
        return self._costo(y, a[-1], h[-1])

    def _gradientes(self, X: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
        a, h = self._forward(X)
        delta = (a[-1] - y) / len(X)
        if self.salida == "logistica":
            delta *= self.activacion_salida.dtheta(h[-1], a[-1])
        dw, db = [None] * len(self.pesos), [None] * len(self.biases)
        for i in range(len(self.pesos) - 1, -1, -1):
            dw[i] = a[i].T @ delta
            db[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.pesos[i].T) * self.activacion.dtheta(h[i - 1], a[i])
        return dw + db

    def backprop(self, X: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
        """Gradientes del costo medio, sin actualizar parámetros ni estado."""
        X = self._entradas(X)
        return self._gradientes(X, self._objetivos(y, len(X)))

    @staticmethod
    def _accuracy(y: np.ndarray, o: np.ndarray) -> float:
        if o.shape[1] == 1:
            return float(np.mean((o[:, 0] >= 0.5) == (y[:, 0] >= 0.5)))
        return float(np.mean(o.argmax(axis=1) == y.argmax(axis=1)))

    def entrenar(self, X: np.ndarray, y: np.ndarray, epocas: int = 200,
                 epsilon: float = 0.0, mezclar: bool = True, verbose: int = 0,
                 X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> Historia:
        """Continúa el modelo; devuelve y almacena la historia de esta llamada.

        epsilon detiene por costo de entrenamiento. Validación sólo se evalúa.
        norma_gradiente registra el promedio de las normas de los mini-batches.
        """
        X = self._entradas(X)
        y = self._objetivos(y, len(X))
        if isinstance(epocas, bool) or not isinstance(epocas, (int, np.integer)) or epocas <= 0:
            raise ValueError("epocas debe ser un entero positivo")
        if not np.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon debe ser no negativo y finito")
        if isinstance(verbose, bool) or not isinstance(verbose, (int, np.integer)) or verbose < 0:
            raise ValueError("verbose debe ser un entero no negativo")
        if (X_val is None) != (y_val is None):
            raise ValueError("La validación requiere X_val e y_val juntos")
        if X_val is not None:
            X_val = self._entradas(X_val)
            y_val = self._objetivos(y_val, len(X_val))
        hist = Historia()
        inicio = time.perf_counter()
        lote = len(X) if self.tamano_lote is None else self.tamano_lote
        for epoca in range(epocas):
            orden = self.rng.permutation(len(X)) if mezclar else np.arange(len(X))
            normas = []
            for j in range(0, len(X), lote):
                idx = orden[j:j + lote]
                gradientes = self._gradientes(X[idx], y[idx])
                normas.append(float(np.sqrt(sum(np.sum(g ** 2) for g in gradientes))))
                self.optimizador.paso(self.parametros, gradientes)
            a, h = self._forward(X)
            hist.costo.append(self._costo(y, a[-1], h[-1]))
            hist.mse.append(float(np.mean((a[-1] - y) ** 2)))
            hist.accuracy.append(self._accuracy(y, a[-1]))
            hist.norma_gradiente.append(float(np.mean(normas)))
            if X_val is not None:
                av, hv = self._forward(X_val)
                hist.costo_validacion.append(self._costo(y_val, av[-1], hv[-1]))
                hist.mse_validacion.append(float(np.mean((av[-1] - y_val) ** 2)))
                hist.accuracy_validacion.append(self._accuracy(y_val, av[-1]))
            hist.epocas_corridas = epoca + 1
            if verbose and (epoca % verbose == 0 or epoca == epocas - 1):
                print(f"época {epoca + 1}/{epocas} costo={hist.costo[-1]:.6f} accuracy={hist.accuracy[-1]:.4f}")
            if epsilon > 0 and hist.costo[-1] < epsilon:
                break
        hist.tiempo_segundos = time.perf_counter() - inicio
        self.historia = hist
        return hist

    def guardar(self, ruta: str | Path) -> None:
        config = {
            "version": 1, "arquitectura": self.arquitectura,
            "activacion": self.activacion.nombre, "salida": self.salida,
            "beta": self.beta, "tamano_lote": self.tamano_lote,
            "inicializacion": self.inicializacion,
            "optimizador": self.optimizador.configuracion(),
            "rng": self.rng.bit_generator.state, "historia": self.historia.to_dict(),
        }
        arrays = {f"w_{i}": w for i, w in enumerate(self.pesos)}
        arrays.update({f"b_{i}": b for i, b in enumerate(self.biases)})
        # Respetar la ruta exacta, sin agregar automáticamente la extensión .npz.
        with open(ruta, "wb") as archivo:
            np.savez_compressed(archivo, configuracion=json.dumps(config), **arrays)

    @classmethod
    def cargar(cls, ruta: str | Path) -> MLP:
        with np.load(ruta, allow_pickle=False) as datos:
            config = json.loads(str(datos["configuracion"]))
            if config.pop("version") != 1:
                raise ValueError("Versión de modelo no compatible")
            rng = config.pop("rng")
            historia = config.pop("historia")
            config["optimizador"] = construir_optimizador(config["optimizador"])
            modelo = cls(**config)
            for i, (w, b) in enumerate(zip(modelo.pesos, modelo.biases)):
                for clave, destino in ((f"w_{i}", w), (f"b_{i}", b)):
                    valor = datos[clave]
                    if valor.shape != destino.shape or not np.all(np.isfinite(valor)):
                        raise ValueError(f"Parámetro inválido en el modelo: {clave}")
                    destino[:] = valor
        modelo.rng.bit_generator.state = rng
        modelo.historia = Historia(**historia)
        return modelo
