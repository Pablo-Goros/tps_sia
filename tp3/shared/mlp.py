"""MLP matricial con mini-batches, optimizadores y persistencia sin pickle.

Filas = muestras. W[l] tiene forma (entrada, salida), b[l] forma (salida,).
Softmax usa cross-entropy media por muestra. Logística usa medio error
cuadrático sumado sobre las salidas y promediado sobre las muestras.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import copy
import os
import tempfile
from pathlib import Path
import time

import numpy as np

from .activations import construir_activacion
from .augmentation import translate_images, rotate_images, validate_augmentation
from .atomic_files import replace
from .schedulers import validate_scheduler, plateau_step
from .batch_norm import validate_batch_norm, backward as batch_norm_backward
from .optimizers import Optimizador, SGD, construir_optimizador


_UNSET = object()


@dataclass
class Historia:
    costo: list[float] = field(default_factory=list)
    mse: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)
    costo_validacion: list[float] = field(default_factory=list)
    mse_validacion: list[float] = field(default_factory=list)
    accuracy_validacion: list[float] = field(default_factory=list)
    norma_gradiente: list[float] = field(default_factory=list)
    validation_epochs: list[int] = field(default_factory=list)
    epochs: list[int] = field(default_factory=list)
    updates: list[int] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    l2_penalty: list[float] = field(default_factory=list)
    training_objective: list[float] = field(default_factory=list)
    stop_reason: str | None = None
    epocas_corridas: int = 0
    tiempo_segundos: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class MLP:
    def __init__(self, arquitectura: list[int], activacion: str = "tanh",
                 salida: str = "softmax", eta: float = 0.01, beta: float = 1.0,
                 tamano_lote: int | None = 32, inicializacion: str = "auto",
                 semilla: int = 0, optimizador: Optimizador | None = None,
                 l2: float = 0.0, augmentation: dict | None = None,
                 lr_scheduler: dict | None = None, batch_norm: dict | None = None) -> None:
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
        if (isinstance(l2, (bool, np.bool_)) or not isinstance(l2, (int, float))
                or not np.isfinite(l2) or l2 < 0):
            raise ValueError('L2 debe ser finito y no negativo')
        self.l2 = float(l2)
        self.augmentation = validate_augmentation(augmentation, self.arquitectura[0])
        self.activacion = construir_activacion(activacion, beta)
        self.salida = salida
        self.activacion_salida = construir_activacion("logistica", beta)
        self.beta = float(beta)
        self.tamano_lote = None if tamano_lote is None else int(tamano_lote)
        self.inicializacion = inicializacion
        self.optimizador = optimizador if optimizador is not None else SGD(eta)
        self.lr_scheduler = validate_scheduler(lr_scheduler)
        initial_config = self.optimizador.configuracion()
        initial_rate = initial_config.get('learning_rate', initial_config.get('eta'))
        if self.lr_scheduler and self.lr_scheduler['min_lr'] > initial_rate:
            raise ValueError('Scheduler min_lr cannot exceed initial learning rate.')
        self.scheduler_state = {'best': None, 'bad_epochs': 0, 'reductions': 0,
                                'current_lr': initial_rate}
        self.seed = int(semilla)
        self.initialization_layers = []
        self.preprocessing = {"name": "identity"}
        self.training_config = {}
        self.training_calls = []
        self.early_stopping = {"best_value": None, "best_epoch": None, "reference_value": None, "bad_epochs": 0}
        self.best_state = None
        self.weight_logging = {"path": None, "every_updates": 1, "selected_weights": []}
        self.rng = np.random.default_rng(semilla)
        self.pesos: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for i, (n_in, n_out) in enumerate(zip(self.arquitectura, self.arquitectura[1:])):
            metodo = inicializacion
            if metodo == "auto":
                metodo = "he" if activacion == "relu" and i < len(arquitectura) - 2 else "xavier"
            escala = np.sqrt(2.0 / n_in) if metodo == "he" else np.sqrt(2.0 / (n_in + n_out))
            self.initialization_layers.append({"method": metodo, "scale": float(escala)})
            self.pesos.append(self.rng.normal(0, escala, size=(n_in, n_out)))
            self.biases.append(np.zeros(n_out))
        self.historia = Historia()
        self.batch_norm = validate_batch_norm(batch_norm)
        if self.batch_norm and (len(self.arquitectura) < 3 or self.tamano_lote == 1):
            raise ValueError('BatchNorm requires a hidden layer and batches of at least2 samples.')
        hidden = self.arquitectura[1:-1] if self.batch_norm else []
        self.bn_gamma = [np.ones(width) for width in hidden]
        self.bn_beta = [np.zeros(width) for width in hidden]
        self.bn_state = {'running_mean': [np.zeros(width) for width in hidden],
                         'running_var': [np.ones(width) for width in hidden],
                         'batches_tracked': [0 for _ in hidden]}

    @property
    def parametros(self) -> list[np.ndarray]:
        """Orden fijo para optimizadores y gradient checks: pesos, luego biases."""
        return self.pesos + self.biases + self.bn_gamma + self.bn_beta

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

    def _forward(self, X: np.ndarray, *, training=False, update_running=False,
                 bn_cache=None) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activaciones, nets = [X], []
        for i, (w, b) in enumerate(zip(self.pesos, self.biases)):
            h = activaciones[-1] @ w + b
            if self.batch_norm and i < len(self.pesos)-1:
                if training:
                    if len(X) < 2:
                        raise ValueError('BatchNorm training requires at least2 samples.')
                    mean, variance = h.mean(axis=0), h.var(axis=0)
                    if update_running:
                        momentum = self.batch_norm['momentum']
                        self.bn_state['running_mean'][i] *= 1-momentum
                        self.bn_state['running_mean'][i] += momentum*mean
                        self.bn_state['running_var'][i] *= 1-momentum
                        self.bn_state['running_var'][i] += momentum*variance*len(X)/(len(X)-1)
                        self.bn_state['batches_tracked'][i] += 1
                else:
                    mean, variance = self.bn_state['running_mean'][i], self.bn_state['running_var'][i]
                inverse = 1/np.sqrt(variance+self.batch_norm['epsilon'])
                normalized = (h-mean)*inverse
                if bn_cache is not None:
                    bn_cache.append((normalized, inverse))
                h = self.bn_gamma[i]*normalized + self.bn_beta[i]
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

    def _gradientes(self, X: np.ndarray, y: np.ndarray, *, update_running=False) -> list[np.ndarray]:
        cache = []
        a, h = self._forward(X, training=bool(self.batch_norm),
                             update_running=update_running, bn_cache=cache)
        delta = (a[-1] - y) / len(X)
        if self.salida == "logistica":
            delta *= self.activacion_salida.dtheta(h[-1], a[-1])
        dw, db = [None] * len(self.pesos), [None] * len(self.biases)
        dg, dt = [None]*len(self.bn_gamma), [None]*len(self.bn_beta)
        for i in range(len(self.pesos) - 1, -1, -1):
            if self.batch_norm and i < len(self.pesos)-1:
                delta, dg[i], dt[i] = batch_norm_backward(delta, *cache[i], self.bn_gamma[i])
            dw[i] = a[i].T @ delta
            if self.l2:
                dw[i] += self.l2 * self.pesos[i]
            db[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.pesos[i].T) * self.activacion.dtheta(h[i - 1], a[i])
        return dw + db + dg + dt

    def backprop(self, X: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
        """Gradientes de loss media + L2, sin penalizar biases ni modificar estado."""
        X = self._entradas(X)
        return self._gradientes(X, self._objetivos(y, len(X)))

    def l2_penalty(self) -> float:
        """lambda/2 * sum(||W||²); lambda is independent of batch/sample count."""
        return self.l2 / 2 * sum(float(np.sum(w ** 2)) for w in self.pesos) if self.l2 else 0.0

    def training_objective(self, X: np.ndarray, y: np.ndarray) -> float:
        """Objective differentiated by backprop; costo remains predictive loss."""
        if self.batch_norm:
            X = self._entradas(X)
            y = self._objetivos(y, len(X))
            a, h = self._forward(X, training=True)
            return self._costo(y, a[-1], h[-1]) + self.l2_penalty()
        return self.costo(X, y) + self.l2_penalty()

    @staticmethod
    def _accuracy(y: np.ndarray, o: np.ndarray) -> float:
        if o.shape[1] == 1:
            return float(np.mean((o[:, 0] >= 0.5) == (y[:, 0] >= 0.5)))
        return float(np.mean(o.argmax(axis=1) == y.argmax(axis=1)))

    @property
    def epochs_completed(self) -> int:
        return self.historia.epocas_corridas

    @property
    def updates_completed(self) -> int:
        return self.optimizador.export_state()["updates"]

    def _snapshot(self) -> dict:
        return copy.deepcopy({
            "parameters": self.parametros, "optimizer": self.optimizador.export_state(),
            "rng": self.rng.bit_generator.state, "history": self.historia.to_dict(),
            "early_stopping": self.early_stopping,
            "scheduler_state": self.scheduler_state,
            "bn_state": self.bn_state,
        })

    def _restore(self, state: dict) -> None:
        parameters = state["parameters"]
        if len(parameters) != len(self.parametros) or any(
                p.shape != value.shape or not np.all(np.isfinite(value))
                for p, value in zip(self.parametros, parameters)):
            raise ValueError("Parámetros inválidos en el estado de entrenamiento")
        optimizer = construir_optimizador(state['optimizer']['config'])
        optimizer.restore_state(state["optimizer"], parameters)
        for p, value in zip(self.parametros, parameters):
            p[:] = value
        self.optimizador = optimizer
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.historia = Historia(**copy.deepcopy(state["history"]))
        self.early_stopping = copy.deepcopy(state["early_stopping"])
        if 'scheduler_state' in state:
            self.scheduler_state = copy.deepcopy(state['scheduler_state'])
        if 'bn_state' in state:
            self._restore_bn_state(state['bn_state'])

    def _restore_bn_state(self, state):
        if not isinstance(state, dict) or set(state) != {'running_mean', 'running_var', 'batches_tracked'}:
            raise ValueError('Invalid BatchNorm state.')
        n = len(self.bn_gamma)
        if any(len(state[key]) != n for key in state):
            raise ValueError('BatchNorm layer count mismatch.')
        for i, gamma in enumerate(self.bn_gamma):
            for key in ('running_mean', 'running_var'):
                value = np.asarray(state[key][i])
                if value.shape != gamma.shape or not np.all(np.isfinite(value)):
                    raise ValueError('Invalid BatchNorm statistics.')
            if np.any(state['running_var'][i] < 0) or type(state['batches_tracked'][i]) is not int or state['batches_tracked'][i] < 0:
                raise ValueError('Invalid BatchNorm variance or count.')
        self.bn_state = copy.deepcopy(state)

    def guardar_best(self, ruta: str | Path) -> None:
        """Guarda la mejor época con su optimizador/RNG, sin cambiar el último estado."""
        if self.best_state is None:
            raise ValueError("No hay una mejor época de validación")
        model = copy.deepcopy(self)
        model._restore(self.best_state)
        model.best_state = copy.deepcopy(self.best_state)
        model.guardar(ruta)

    def entrenar(self, X: np.ndarray, y: np.ndarray, epocas: int = 200,
                 epsilon: float = _UNSET, mezclar: bool = _UNSET, verbose: int = 0,
                 X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
                 *, patience: int | None = _UNSET, min_delta: float = _UNSET,
                 monitor: str = _UNSET, checkpoint_path: str | Path | None = None,
                 checkpoint_every: int = 1, best_model_path: str | Path | None = None,
                 pause_after: int | None = None, weight_log_path: str | Path | None = None,
                 log_every_updates: int = _UNSET,
                 selected_weights: list[tuple[int, int, int]] | None = None) -> Historia:
        """Continúa desde la última época y devuelve la historia acumulada.

        epocas es el presupuesto adicional. La tasa es constante; epsilon mide
        costo de train. Early stopping minimiza loss o maximiza accuracy de
        validación. Nunca restaura automáticamente la mejor época sobre el último
        estado: guardar_best la exporta por separado. Pausa/rollback en límite de
        época; una interrupción a mitad de época la repite al continuar.
        """
        epsilon = self.training_config.get("epsilon", 0.0) if epsilon is _UNSET else epsilon
        mezclar = self.training_config.get("shuffle", True) if mezclar is _UNSET else mezclar
        patience = self.training_config.get("patience") if patience is _UNSET else patience
        min_delta = self.training_config.get("min_delta", 0.0) if min_delta is _UNSET else min_delta
        monitor = self.training_config.get("monitor", "validation_loss") if monitor is _UNSET else monitor
        log_every_updates = self.weight_logging["every_updates"] if log_every_updates is _UNSET else log_every_updates
        if (checkpoint_path is not None and best_model_path is not None
                and Path(checkpoint_path).resolve() == Path(best_model_path).resolve()):
            raise ValueError("El checkpoint y el mejor modelo requieren rutas diferentes")
        effective_log_path = weight_log_path if weight_log_path is not None else self.weight_logging["path"]
        for path in (checkpoint_path, best_model_path):
            if path is not None and effective_log_path is not None and Path(path).resolve() == Path(effective_log_path).resolve():
                raise ValueError("El registro de pesos requiere un archivo aparte")
        X = self._entradas(X)
        y = self._objetivos(y, len(X))
        def positive_integer(value, name):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} debe ser un entero positivo")
        for value, name in ((epocas, "epocas"), (checkpoint_every, "checkpoint_every"),
                            (log_every_updates, "log_every_updates")):
            positive_integer(value, name)
        for value, name in ((patience, "patience"), (pause_after, "pause_after")):
            if value is not None:
                positive_integer(value, name)
        if not np.isfinite(epsilon) or epsilon < 0 or not np.isfinite(min_delta) or min_delta < 0:
            raise ValueError("epsilon y min_delta deben ser no negativos y finitos")
        if isinstance(verbose, bool) or not isinstance(verbose, (int, np.integer)) or verbose < 0:
            raise ValueError("verbose debe ser un entero no negativo")
        if not isinstance(mezclar, (bool, np.bool_)):
            raise ValueError("mezclar debe ser booleano")
        if (X_val is None) != (y_val is None):
            raise ValueError("La validación requiere X_val e y_val juntos")
        if monitor not in ("validation_loss", "validation_accuracy"):
            raise ValueError("monitor debe ser validation_loss o validation_accuracy")
        if X_val is None and (patience is not None or best_model_path is not None):
            raise ValueError("La selección y parada temprana requieren validación")
        if X_val is not None:
            X_val = self._entradas(X_val)
            y_val = self._objetivos(y_val, len(X_val))
        selected = selected_weights if selected_weights is not None else self.weight_logging["selected_weights"]
        for indices in selected:
            if (len(indices) != 3 or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in indices)):
                raise ValueError("Cada peso seleccionado requiere (layer, row, column)")
            layer, row, column = indices
            if not (0 <= layer < len(self.pesos) and 0 <= row < self.pesos[layer].shape[0]
                    and 0 <= column < self.pesos[layer].shape[1]):
                raise ValueError("Índice de peso fuera de rango")
        patience = int(patience) if patience is not None else None
        log_every_updates = int(log_every_updates)
        selected = [tuple(int(i) for i in index) for index in selected]
        config = {"shuffle": bool(mezclar), "epsilon": float(epsilon), "epsilon_quantity": "training_loss",
                  "patience": patience, "min_delta": float(min_delta), "monitor": monitor,
                  "loss": "cross_entropy" if self.salida == "softmax" else "half_squared_error",
                  "gradient_reduction": "mean_per_sample", "learning_rate_schedule": "constant",
                  "strategy": "batch" if self.tamano_lote is None else "online" if self.tamano_lote == 1 else "mini_batch",
                  "batch_size": self.tamano_lote, "validation": X_val is not None}
        if self.l2:
            config['l2'] = self.l2
            config['objective'] = 'mean_predictive_loss + l2/2 * sum_squared_weights; biases excluded'
        if self.augmentation:
            config['augmentation'] = self.augmentation
        if self.batch_norm:
            if len(X) < 2:
                raise ValueError('BatchNorm training requires at least2 samples.')
            config['batch_norm'] = self.batch_norm
        if self.lr_scheduler:
            if X_val is None:
                raise ValueError('Plateau scheduler requires validation data.')
            config['learning_rate_schedule'] = self.lr_scheduler
        if self.training_config and config != self.training_config:
            # Un cambio de política empieza un nuevo seguimiento de selección.
            self.early_stopping = {"best_value": None, "best_epoch": None, "reference_value": None, "bad_epochs": 0}
            self.best_state = None
        self.training_config = config
        self.training_calls.append({"start_epoch": self.epochs_completed, "max_epochs": int(epocas),
                                    "checkpoint_every": int(checkpoint_every),
                                    "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
                                    "best_model_path": str(best_model_path) if best_model_path is not None else None,
                                    "pause_after": int(pause_after) if pause_after is not None else None})
        self.weight_logging = {"path": str(weight_log_path) if weight_log_path is not None else self.weight_logging["path"],
                               "every_updates": log_every_updates, "selected_weights": [list(i) for i in selected]}
        inicio = time.perf_counter()
        previous_time = self.historia.tiempo_segundos
        lote = len(X) if self.tamano_lote is None else self.tamano_lote
        # Old checkpoints have no objective history; their penalty was zero.
        if not self.historia.l2_penalty and self.historia.epocas_corridas:
            self.historia.l2_penalty = [0.0] * self.historia.epocas_corridas
            self.historia.training_objective = list(self.historia.costo)
        self.historia.stop_reason = None
        for local_epoch in range(epocas):
            rollback = self._snapshot()
            records = []
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    orden = self.rng.permutation(len(X)) if mezclar else np.arange(len(X))
                    normas = []
                    lr_config = self.optimizador.configuracion()
                    rate = lr_config.get("learning_rate", lr_config.get("eta"))
                    starts = list(range(0, len(X), lote))
                    if self.batch_norm and len(starts)>1 and len(X)-starts[-1] == 1:
                        starts.pop()  # Merge singleton tail into the preceding mini-batch.
                    for batch_index, j in enumerate(starts):
                        end = starts[batch_index+1] if batch_index+1 < len(starts) else len(X)
                        idx = orden[j:end]
                        batch = X[idx]
                        if self.augmentation:
                            batch = translate_images(batch, self.augmentation['max_shift'], self.rng)
                            if self.augmentation['name'] == 'translation_rotation':
                                batch = rotate_images(batch, self.augmentation['max_angle_degrees'], self.rng)
                        gradients = (self._gradientes(batch, y[idx], update_running=True)
                                     if self.batch_norm else self._gradientes(batch, y[idx]))
                        if not all(np.all(np.isfinite(g)) for g in gradients):
                            raise FloatingPointError("Gradiente no finito")
                        normas.append(float(np.sqrt(sum(np.sum(g ** 2) for g in gradients))))
                        before = [p.copy() for p in self.parametros] if self.weight_logging["path"] else None
                        try:
                            self.optimizador.paso(self.parametros, gradients)
                        except ValueError as exc:
                            raise FloatingPointError(str(exc)) from exc
                        if before is not None and self.updates_completed % log_every_updates == 0:
                            layers = len(self.pesos)
                            records.append({"epoch": self.epochs_completed + 1, "update": self.updates_completed,
                                "learning_rate": rate, "layers": [
                                    {"weight_norm": float(np.linalg.norm(w)), "bias_norm": float(np.linalg.norm(self.biases[i])),
                                     "gradient_norm": float(np.sqrt(np.sum(gradients[i] ** 2) + np.sum(gradients[layers+i] ** 2))),
                                     "update_norm": float(np.sqrt(np.sum((w-before[i]) ** 2) + np.sum((self.biases[i]-before[layers+i]) ** 2)))}
                                    for i, w in enumerate(self.pesos)],
                                "selected_weights": [{"index": list(index), "value": float(self.pesos[index[0]][index[1], index[2]]),
                                    "update": float(self.pesos[index[0]][index[1], index[2]] - before[index[0]][index[1], index[2]])} for index in selected]})
                    a, h = self._forward(X)
                    metrics = [self._costo(y, a[-1], h[-1]), float(np.mean((a[-1]-y)**2)), self._accuracy(y, a[-1])]
                    penalty = self.l2_penalty()
                    val_metrics = []
                    if X_val is not None:
                        av, hv = self._forward(X_val)
                        val_metrics = [self._costo(y_val, av[-1], hv[-1]), float(np.mean((av[-1]-y_val)**2)), self._accuracy(y_val, av[-1])]
                    if not np.all(np.isfinite(metrics + val_metrics + [penalty])):
                        raise FloatingPointError("Métricas no finitas")
            except (KeyboardInterrupt, FloatingPointError) as exc:
                self._restore(rollback)
                self.historia.stop_reason = "interrupted" if isinstance(exc, KeyboardInterrupt) else "numerical_failure"
                break
            hist = self.historia
            for name, value in zip(("costo", "mse", "accuracy"), metrics):
                getattr(hist, name).append(value)
            hist.l2_penalty.append(penalty)
            hist.training_objective.append(metrics[0] + penalty)
            for name, value in zip(("costo_validacion", "mse_validacion", "accuracy_validacion"), val_metrics):
                getattr(hist, name).append(value)
            if val_metrics:
                hist.validation_epochs.append(hist.epocas_corridas + 1)
            hist.norma_gradiente.append(float(np.mean(normas)))
            hist.epocas_corridas += 1
            hist.epochs.append(hist.epocas_corridas)
            hist.updates.append(self.updates_completed)
            hist.learning_rate.append(rate)
            if self.lr_scheduler:
                next_rate = plateau_step(self.lr_scheduler, self.scheduler_state, val_metrics[0], rate)
                if hasattr(self.optimizador, 'learning_rate'):
                    self.optimizador.learning_rate = next_rate
                else:
                    self.optimizador.eta = next_rate
            if val_metrics:
                value = val_metrics[0] if monitor == "validation_loss" else val_metrics[2]
                old = self.early_stopping["best_value"]
                reference = self.early_stopping["reference_value"]
                gain = lambda previous: previous-value if monitor == "validation_loss" else value-previous
                significant = reference is None or gain(reference) > min_delta
                if significant:
                    self.early_stopping["reference_value"] = value
                    self.early_stopping["bad_epochs"] = 0
                else:
                    self.early_stopping["bad_epochs"] += 1
                # min_delta controla paciencia, sin descartar la mejor métrica observada.
                if old is None or gain(old) > 0:
                    self.early_stopping["best_value"] = value
                    self.early_stopping["best_epoch"] = hist.epocas_corridas
                    hist.tiempo_segundos = previous_time + time.perf_counter() - inicio
                    self.best_state = self._snapshot()
                    if best_model_path is not None:
                        self.guardar_best(best_model_path)
            if self.weight_logging["path"]:
                path = Path(self.weight_logging["path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as file:
                    for record in records:
                        file.write(json.dumps({**record, "training_config": config,
                            "model_config": {"architecture": self.arquitectura, "activation": self.activacion.nombre,
                                             "beta": self.beta, "output": self.salida, "seed": self.seed,
                                             "initialization_layers": self.initialization_layers,
                                             "optimizer": self.optimizador.configuracion(), "preprocessing": self.preprocessing}}, allow_nan=False) + "\n")
            if epsilon > 0 and metrics[0] < epsilon:
                hist.stop_reason = "training_epsilon"
            elif patience is not None and self.early_stopping["bad_epochs"] >= patience:
                hist.stop_reason = "early_stopping"
            elif pause_after is not None and local_epoch + 1 >= pause_after:
                hist.stop_reason = "interrupted"
            elif local_epoch + 1 == epocas:
                hist.stop_reason = "max_epochs"
            if verbose and (hist.epocas_corridas % verbose == 0 or hist.stop_reason):
                print(f"época {hist.epocas_corridas} costo={metrics[0]:.6f} accuracy={metrics[2]:.4f}")
            hist.tiempo_segundos = previous_time + time.perf_counter() - inicio
            if checkpoint_path is not None and hist.stop_reason is None and hist.epocas_corridas % checkpoint_every == 0:
                self.guardar(checkpoint_path)
            if hist.stop_reason:
                break
        self.historia.tiempo_segundos = previous_time + time.perf_counter() - inicio
        if checkpoint_path is not None:
            self.guardar(checkpoint_path)
        if best_model_path is not None and self.best_state is not None:
            self.guardar_best(best_model_path)
        return self.historia

    def guardar(self, ruta: str | Path) -> None:
        config = {
            "version": 3, "arquitectura": self.arquitectura,
            "activacion": self.activacion.nombre, "salida": self.salida,
            "beta": self.beta, "tamano_lote": self.tamano_lote,
            "inicializacion": self.inicializacion,
            "l2": self.l2,
            "augmentation": self.augmentation,
            "lr_scheduler": self.lr_scheduler,
            "batch_norm": self.batch_norm,
            "optimizador": self.optimizador.configuracion(),
            "rng": self.rng.bit_generator.state, "historia": self.historia.to_dict(),
        }
        def pack(value):
            if isinstance(value, np.ndarray):
                name = f"training_tensor_{len(arrays)}"
                arrays[name] = value
                return {"array": name}
            if isinstance(value, dict):
                return {key: pack(item) for key, item in value.items()}
            if isinstance(value, list):
                return [pack(item) for item in value]
            return value
        arrays = {f"w_{i}": w for i, w in enumerate(self.pesos)}
        arrays.update({f"b_{i}": b for i, b in enumerate(self.biases)})
        arrays.update({f'bn_gamma_{i}': gamma for i, gamma in enumerate(self.bn_gamma)})
        arrays.update({f'bn_beta_{i}': beta for i, beta in enumerate(self.bn_beta)})
        state = self.optimizador.export_state()
        # Los momentos se guardan como arrays NPZ, sin pickle ni listas JSON
        # de millones de números; la metadata conserva su orden explícito.
        for key, values in list(state.items()):
            if isinstance(values, list):
                names = [f"optimizer_{key}_{i}" for i in range(len(values))]
                arrays.update(zip(names, values))
                state[key] = names
        config["optimizer_state"] = state
        config["training_state"] = pack({
            "seed": self.seed, "initialization_layers": self.initialization_layers,
            "preprocessing": self.preprocessing, "training_config": self.training_config,
            "training_calls": self.training_calls,
            "early_stopping": self.early_stopping, "best_state": self.best_state,
            "weight_logging": self.weight_logging,
            "scheduler_state": self.scheduler_state,
            "bn_state": self.bn_state,
        })
        # Respetar la ruta exacta, sin agregar automáticamente la extensión .npz.
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=ruta.parent, prefix=ruta.name + ".", suffix=".tmp", delete=False) as archivo:
                temporary = Path(archivo.name)
                np.savez_compressed(archivo, configuracion=json.dumps(config, allow_nan=False), **arrays)
                archivo.flush()
                os.fsync(archivo.fileno())
            replace(temporary, ruta)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @classmethod
    def cargar(cls, ruta: str | Path) -> MLP:
        with np.load(ruta, allow_pickle=False) as datos:
            config = json.loads(str(datos["configuracion"]))
            version = config.pop("version")
            if version not in (1, 2, 3):
                raise ValueError("Versión de modelo no compatible")
            if version >= 2:
                state = config.pop("optimizer_state", None)
                if not isinstance(state, dict):
                    raise ValueError("El modelo versión 2 requiere estado del optimizador")
                for key, names in list(state.items()):
                    if isinstance(names, list):
                        if not all(isinstance(name, str) and name in datos.files for name in names):
                            raise ValueError("Tensor del optimizador ausente o inválido")
                        state[key] = [datos[name] for name in names]
            def unpack(value):
                if isinstance(value, dict):
                    if set(value) == {"array"}:
                        return datos[value["array"]].copy()
                    return {key: unpack(item) for key, item in value.items()}
                if isinstance(value, list):
                    return [unpack(item) for item in value]
                return value
            if version == 3 and not isinstance(config.get("training_state"), dict):
                raise ValueError("El formato versión 3 requiere estado de entrenamiento")
            training_state = unpack(config.pop("training_state", {}))
            rng = config.pop("rng")
            historia = config.pop("historia")
            config["optimizador"] = construir_optimizador(config["optimizador"])
            if version == 1 and not isinstance(config["optimizador"], SGD):
                raise ValueError("El formato versión 1 sólo conserva el estado de SGD")
            modelo = cls(**config)
            for i, (w, b) in enumerate(zip(modelo.pesos, modelo.biases)):
                for clave, destino in ((f"w_{i}", w), (f"b_{i}", b)):
                    valor = datos[clave]
                    if valor.shape != destino.shape or not np.all(np.isfinite(valor)):
                        raise ValueError(f"Parámetro inválido en el modelo: {clave}")
                    destino[:] = valor
            for prefix, parameters in (('bn_gamma', modelo.bn_gamma), ('bn_beta', modelo.bn_beta)):
                for i, parameter in enumerate(parameters):
                    value = datos[f'{prefix}_{i}']
                    if value.shape != parameter.shape or not np.all(np.isfinite(value)):
                        raise ValueError('Invalid BatchNorm affine parameter.')
                    parameter[:] = value
            if version >= 2:
                modelo.optimizador.restore_state(state, modelo.parametros)
        modelo.rng.bit_generator.state = rng
        modelo.historia = Historia(**historia)
        for key, value in training_state.items():
            if key not in {"seed", "initialization_layers", "preprocessing", "training_config", "training_calls", "early_stopping", "best_state", "weight_logging", "scheduler_state", "bn_state"}:
                raise ValueError("Campo desconocido en el estado de entrenamiento")
            setattr(modelo, key, value)
        modelo._restore_bn_state(modelo.bn_state)
        if version == 3:
            hist = modelo.historia
            if (hist.epocas_corridas != len(hist.costo)
                    or hist.epochs != list(range(1, hist.epocas_corridas + 1))
                    or any(len(values) != hist.epocas_corridas for values in
                           (hist.mse, hist.accuracy, hist.norma_gradiente, hist.learning_rate, hist.updates))
                    or not (len(hist.validation_epochs) == len(hist.costo_validacion)
                            == len(hist.mse_validacion) == len(hist.accuracy_validacion))
                    or (hist.updates and hist.updates[-1] != modelo.updates_completed)):
                raise ValueError("Historia y contadores incompatibles")
            if modelo.best_state is not None:
                probe = copy.deepcopy(modelo)
                probe._restore(modelo.best_state)
        if version < 3:
            modelo.seed = None  # Los formatos anteriores no guardaban la semilla inicial.
            modelo.historia.updates = [None] * modelo.epochs_completed
            if modelo.historia.updates:
                modelo.historia.updates[-1] = modelo.updates_completed
            modelo.historia.learning_rate = [None] * modelo.epochs_completed
            modelo.historia.validation_epochs = list(range(1, len(modelo.historia.costo_validacion) + 1))
            modelo.historia.epochs = list(range(1, modelo.epochs_completed + 1))
        return modelo
