"""Optimizadores con actualizaciones atómicas y estado independiente del MLP.

Cada paso recibe gradientes ya reducidos por el modelo. El orden de los
tensores (pesos, luego biases en MLP.parametros) también ordena el estado.
"""
from __future__ import annotations

from numbers import Real
from typing import Protocol, Sequence

import numpy as np


class Optimizador(Protocol):
    def paso(self, parametros: Sequence[np.ndarray],
             gradientes: Sequence[np.ndarray]) -> None: ...

    def configuracion(self) -> dict: ...

    def export_state(self) -> dict: ...

    def restore_state(self, state: dict, parameters: Sequence[np.ndarray]) -> None: ...


def _number(value: float, name: str, *, unit_interval: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} debe ser un número real")
    value = float(value)
    invalid = not 0 <= value < 1 if unit_interval else value <= 0
    if not np.isfinite(value) or invalid:
        bounds = "en [0, 1)" if unit_interval else "positivo"
        raise ValueError(f"{name} debe ser finito y {bounds}")
    return value


def _parameters(parameters: Sequence[np.ndarray]) -> None:
    if not len(parameters):
        raise ValueError("Se requiere al menos un parámetro")
    for parameter in parameters:
        if (not isinstance(parameter, np.ndarray) or parameter.dtype.kind != "f"
                or not parameter.flags.writeable or not np.all(np.isfinite(parameter))):
            raise ValueError("Los parámetros deben ser arrays flotantes, finitos y escribibles")


def _gradients(parameters: Sequence[np.ndarray], gradients: Sequence[np.ndarray]) -> None:
    _parameters(parameters)
    if len(parameters) != len(gradients):
        raise ValueError("Debe haber un gradiente por parámetro")
    for parameter, gradient in zip(parameters, gradients):
        if not isinstance(gradient, np.ndarray) or gradient.dtype.kind not in "fiu":
            raise ValueError("Los gradientes deben ser arrays numéricos reales")
        if parameter.shape != gradient.shape:
            raise ValueError("La forma del gradiente no coincide con el parámetro")
        if not np.all(np.isfinite(gradient)):
            raise ValueError("El gradiente contiene valores no finitos")


def _tensors(values: list, parameters: Sequence[np.ndarray], *,
             nonnegative: bool = False, copy: bool = True) -> list:
    if not isinstance(values, list) or len(values) != len(parameters):
        raise ValueError("Debe haber un tensor de estado por parámetro")
    result = []
    for value, parameter in zip(values, parameters):
        if (not isinstance(value, np.ndarray) or value.dtype.kind != "f"
                or value.shape != parameter.shape or not np.all(np.isfinite(value))
                or (nonnegative and np.any(value < 0))):
            raise ValueError("Tensor de estado inválido: forma, tipo o valores")
        result.append(value.copy() if copy else value)
    return result


class _Optimizer:
    _state_keys: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.updates = 0
        self._state: dict[str, list[np.ndarray]] = {key: [] for key in self._state_keys}

    def export_state(self) -> dict:
        """Devuelve copias: modificar el resultado no altera el optimizador."""
        return {
            "version": 1, "config": self.configuracion(), "updates": self.updates,
            **{key: [value.copy() for value in values] for key, values in self._state.items()},
        }

    def restore_state(self, state: dict, parameters: Sequence[np.ndarray]) -> None:
        """Restaura sólo tras validar todo; tampoco modifica los parámetros."""
        _parameters(parameters)
        required = {"version", "config", "updates", *self._state_keys}
        if (not isinstance(state, dict) or set(state) != required
                or type(state["version"]) is not int or state["version"] != 1):
            raise ValueError("Formato de estado del optimizador incompatible")
        config = construir_optimizador(state["config"]).configuracion()
        if config != self.configuracion():
            raise ValueError("La configuración del estado no coincide con el optimizador")
        updates = state["updates"]
        if isinstance(updates, (bool, np.bool_)) or not isinstance(updates, (int, np.integer)) or updates < 0:
            raise ValueError("updates debe ser un entero no negativo")
        restored = {}
        for key in self._state_keys:
            if updates == 0:
                if not isinstance(state[key], list) or state[key]:
                    raise ValueError("Un optimizador sin actualizaciones debe tener estado vacío")
                restored[key] = []
            else:
                restored[key] = _tensors(state[key], parameters, nonnegative=key == "second_moment")
        self._state = restored
        self.updates = int(updates)

    def paso(self, parametros: Sequence[np.ndarray],
             gradientes: Sequence[np.ndarray]) -> None:
        self.configuracion()  # También verifica hiperparámetros modificados entre pasos.
        _gradients(parametros, gradientes)
        for key in self._state_keys:
            if self.updates:
                _tensors(self._state[key], parametros, nonnegative=key == "second_moment", copy=False)
        # Preparar todo antes de escribir: un overflow en el último bias no
        # deja pesos, momentos ni contador parcialmente actualizados.
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                values, state = self._step(parametros, gradientes)
                values = [np.asarray(value, dtype=parameter.dtype)
                          for value, parameter in zip(values, parametros)]
                if not all(np.all(np.isfinite(value)) for value in values):
                    raise ValueError("La actualización produce parámetros no finitos")
                for key in self._state_keys:
                    _tensors(state[key], parametros, nonnegative=key == "second_moment", copy=False)
        except FloatingPointError as exc:
            raise ValueError("La actualización produce valores no finitos") from exc
        for parameter, value in zip(parametros, values):
            np.copyto(parameter, value, casting="no")
        self._state = state
        self.updates += 1


class SGD(_Optimizer):
    def __init__(self, eta: float = 0.01) -> None:
        super().__init__()
        self.eta = _number(eta, "eta")

    def _step(self, parameters, gradients):
        return [p - self.eta * g for p, g in zip(parameters, gradients)], {}

    def configuracion(self) -> dict:
        # Configuración histórica, para modelos SGD versión 1.
        return {"nombre": "sgd", "eta": _number(self.eta, "eta")}


class Momentum(_Optimizer):
    """Momentum clásico: v_t = momentum*v_(t-1) + g_t; p -= learning_rate*v_t."""

    _state_keys = ("velocity",)

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9) -> None:
        super().__init__()
        self.learning_rate = _number(learning_rate, "learning_rate")
        self.momentum = _number(momentum, "momentum", unit_interval=True)

    def _step(self, parameters, gradients):
        previous = self._state["velocity"] if self.updates else [np.zeros_like(p) for p in parameters]
        velocity = [self.momentum * v + g for v, g in zip(previous, gradients)]
        return ([p - self.learning_rate * v for p, v in zip(parameters, velocity)],
                {"velocity": velocity})

    def configuracion(self) -> dict:
        return {"name": "momentum", "learning_rate": _number(self.learning_rate, "learning_rate"),
                "momentum": _number(self.momentum, "momentum", unit_interval=True)}


class Adam(_Optimizer):
    """Adam con corrección de sesgo y epsilon fuera de la raíz cuadrada.

    optimizer_epsilon estabiliza la división; no es el epsilon de parada.
    updates cuenta lotes, no épocas, y se incrementa una vez por paso completo.
    """

    _state_keys = ("first_moment", "second_moment")

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, optimizer_epsilon: float = 1e-8) -> None:
        super().__init__()
        self.learning_rate = _number(learning_rate, "learning_rate")
        self.beta1 = _number(beta1, "beta1", unit_interval=True)
        self.beta2 = _number(beta2, "beta2", unit_interval=True)
        self.optimizer_epsilon = _number(optimizer_epsilon, "optimizer_epsilon")

    def _step(self, parameters, gradients):
        first = self._state["first_moment"] if self.updates else [np.zeros_like(p) for p in parameters]
        second = self._state["second_moment"] if self.updates else [np.zeros_like(p) for p in parameters]
        first = [self.beta1 * m + (1 - self.beta1) * g for m, g in zip(first, gradients)]
        second = [self.beta2 * v + (1 - self.beta2) * np.square(g.astype(float))
                  for v, g in zip(second, gradients)]
        t = self.updates + 1
        values = [p - self.learning_rate * (m / (1 - self.beta1 ** t)) /
                  (np.sqrt(v / (1 - self.beta2 ** t)) + self.optimizer_epsilon)
                  for p, m, v in zip(parameters, first, second)]
        return values, {"first_moment": first, "second_moment": second}

    def configuracion(self) -> dict:
        return {"name": "adam", "learning_rate": _number(self.learning_rate, "learning_rate"),
                "beta1": _number(self.beta1, "beta1", unit_interval=True),
                "beta2": _number(self.beta2, "beta2", unit_interval=True),
                "optimizer_epsilon": _number(self.optimizer_epsilon, "optimizer_epsilon")}


class RMSProp(_Optimizer):
    """RMSProp sin corrección de sesgo, con estado inicial cero:
    s_t = rho*s_(t-1) + (1-rho)*g_t**2; p -= learning_rate*g_t/(sqrt(s_t)+optimizer_epsilon).
    """

    _state_keys = ("second_moment",)

    def __init__(self, learning_rate: float = 0.001, rho: float = 0.9,
                 optimizer_epsilon: float = 1e-8) -> None:
        super().__init__()
        self.learning_rate = _number(learning_rate, "learning_rate")
        self.rho = _number(rho, "rho", unit_interval=True)
        self.optimizer_epsilon = _number(optimizer_epsilon, "optimizer_epsilon")

    def _step(self, parameters, gradients):
        previous = self._state["second_moment"] if self.updates else [np.zeros_like(p) for p in parameters]
        second = [self.rho * s + (1 - self.rho) * np.square(g.astype(float))
                  for s, g in zip(previous, gradients)]
        values = [p - self.learning_rate * g / (np.sqrt(s) + self.optimizer_epsilon)
                  for p, g, s in zip(parameters, gradients, second)]
        return values, {"second_moment": second}

    def configuracion(self) -> dict:
        return {"name": "rmsprop", "learning_rate": _number(self.learning_rate, "learning_rate"),
                "rho": _number(self.rho, "rho", unit_interval=True),
                "optimizer_epsilon": _number(self.optimizer_epsilon, "optimizer_epsilon")}


def construir_optimizador(configuracion: dict) -> Optimizador:
    """Construye SGD (incluida su configuración previa), momentum, RMSProp o Adam."""
    if not isinstance(configuracion, dict):
        raise ValueError("La configuración del optimizador debe ser un mapping")
    if "nombre" in configuracion:
        if set(configuracion) != {"nombre", "eta"} or configuracion["nombre"] != "sgd":
            raise ValueError("Configuración histórica de SGD inválida")
        return SGD(eta=configuracion["eta"])
    config = configuracion.copy()
    name = config.pop("name", None)
    if not isinstance(name, str) or name not in ("sgd", "momentum", "rmsprop", "adam"):
        raise ValueError(f"Optimizador desconocido: {name!r}")
    allowed = {"learning_rate"}
    if name == "momentum":
        allowed.add("momentum")
    elif name == "rmsprop":
        allowed.update(("rho", "optimizer_epsilon"))
    elif name == "adam":
        allowed.update(("beta1", "beta2", "optimizer_epsilon"))
    if set(config) - allowed:
        raise ValueError("Campos desconocidos en la configuración del optimizador")
    if name == "sgd":
        return SGD(eta=config.get("learning_rate", 0.01))
    return {"momentum": Momentum, "rmsprop": RMSProp, "adam": Adam}[name](**config)
