"""Perceptrón simple (lineal / no lineal) - TP3 SIA.

Implementa el algoritmo de la Clase 10.2:

    h^mu   = sum_i x_i^mu * w_i          (con x_0 = 1 para el bias)
    O^mu   = theta(h^mu)
    Dw     = eta * (zeta^mu - O^mu) * theta'(h^mu) * x^mu
    w      = w + Dw

El parámetro ``tamano_lote`` controla el régimen de entrenamiento:
    1    -> ONLINE (incremental, el de la cátedra)
    None -> BATCH  (acumula el gradiente sobre todas las muestras)
    k    -> MINI-BATCH de k muestras

Todas las operaciones son matriciales (numpy), como pide el enunciado.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .activaciones import Activacion, construir_activacion
from .metricas import error_cuadratico_medio


@dataclass
class Historia:
    """Registro por época de una corrida de entrenamiento."""
    mse: List[float] = field(default_factory=list)
    costo: List[float] = field(default_factory=list)
    norma_gradiente: List[float] = field(default_factory=list)
    derivada_media: List[float] = field(default_factory=list)   # media de theta'(h)
    fraccion_saturada: List[float] = field(default_factory=list)  # % con theta'(h) chico
    mse_validacion: List[float] = field(default_factory=list)  # sólo si se pasa X_val
    epocas_corridas: int = 0
    mejor_mse: float = float("inf")
    mejores_pesos: Optional[np.ndarray] = None
    tiempo_segundos: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "mse": self.mse,
            "costo": self.costo,
            "norma_gradiente": self.norma_gradiente,
            "derivada_media": self.derivada_media,
            "fraccion_saturada": self.fraccion_saturada,
            "mse_validacion": self.mse_validacion,
            "epocas_corridas": self.epocas_corridas,
            "mejor_mse": self.mejor_mse,
            "tiempo_segundos": self.tiempo_segundos,
        }


class PerceptronSimple:
    def __init__(
        self,
        n_entradas: int,
        activacion: str = "lineal",
        eta: float = 0.01,
        beta: float = 0.5,
        tamano_lote: Optional[int] = 1,
        escala_init: float = 0.01,
        semilla: int = 0,
        umbral_saturacion: float = 0.05,
    ) -> None:
        self.n_entradas = n_entradas
        self.eta = eta
        self.beta = beta
        self.tamano_lote = tamano_lote
        self.umbral_saturacion = umbral_saturacion
        self.activacion: Activacion = construir_activacion(activacion, beta)
        self.rng = np.random.default_rng(semilla)
        # w[0] es el bias (umbral); se inicializa chico y aleatorio.
        self.w = self.rng.uniform(-escala_init, escala_init, size=n_entradas + 1)

    # ------------------------------------------------------------------ interno
    @staticmethod
    def _con_bias(X: np.ndarray) -> np.ndarray:
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _h(self, Xb: np.ndarray) -> np.ndarray:
        return Xb @ self.w

    # ------------------------------------------------------------------ público
    def predecir(self, X: np.ndarray) -> np.ndarray:
        return self.activacion.theta(self._h(self._con_bias(X)))

    def entrenar(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epocas: int = 200,
        epsilon: float = 0.0,
        mezclar: bool = True,
        registrar_diagnostico: bool = True,
        verbose: int = 0,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Historia:
        """Entrena y devuelve la historia por época (MSE sobre todo el conjunto).

        Si se pasa (X_val, y_val), registra además el MSE de validación por época
        (sin usarlo para actualizar pesos), para estudiar generalización/sobreajuste.
        """
        import time

        Xb = self._con_bias(X)
        p = Xb.shape[0]
        lote = p if self.tamano_lote is None else min(self.tamano_lote, p)
        hist = Historia()
        t0 = time.perf_counter()

        for epoca in range(epocas):
            orden = self.rng.permutation(p) if mezclar else np.arange(p)
            grad_acum = 0.0

            for inicio in range(0, p, lote):
                idx = orden[inicio:inicio + lote]
                xb = Xb[idx]
                h = xb @ self.w
                o = self.activacion.theta(h)
                delta = (y[idx] - o) * self.activacion.dtheta(h, o)
                # Promedio sobre el lote para que eta no dependa del tamaño del lote.
                dw = self.eta * (xb.T @ delta) / len(idx)
                self.w += dw
                grad_acum += float(np.linalg.norm(dw))

            # ---- evaluación de la época sobre TODO el conjunto
            h_all = Xb @ self.w
            o_all = self.activacion.theta(h_all)
            mse = error_cuadratico_medio(y, o_all)
            hist.mse.append(mse)
            hist.costo.append(float(0.5 * np.sum((y - o_all) ** 2)))
            hist.norma_gradiente.append(grad_acum)

            if registrar_diagnostico:
                d = self.activacion.dtheta(h_all, o_all)
                d_norm = d / (2.0 * self.beta) if self.activacion.nombre == "logistica" else d
                hist.derivada_media.append(float(np.mean(d)))
                hist.fraccion_saturada.append(float(np.mean(d_norm < self.umbral_saturacion)))

            if X_val is not None:
                hist.mse_validacion.append(error_cuadratico_medio(y_val, self.predecir(X_val)))

            if mse < hist.mejor_mse:
                hist.mejor_mse = mse
                hist.mejores_pesos = self.w.copy()

            hist.epocas_corridas = epoca + 1

            if verbose and (epoca % verbose == 0 or epoca == epocas - 1):
                print(f"    época {epoca + 1:>5}/{epocas}  MSE={mse:.6f}")

            if epsilon > 0.0 and mse < epsilon:
                break

        hist.tiempo_segundos = time.perf_counter() - t0
        return hist

    # ------------------------------------------------------- persistencia simple
    def guardar(self, ruta: str) -> None:
        np.savez(
            ruta,
            w=self.w,
            activacion=self.activacion.nombre,
            eta=self.eta,
            beta=self.beta,
        )

    @classmethod
    def cargar(cls, ruta: str) -> "PerceptronSimple":
        d = np.load(ruta, allow_pickle=False)
        modelo = cls(
            n_entradas=len(d["w"]) - 1,
            activacion=str(d["activacion"]),
            eta=float(d["eta"]),
            beta=float(d["beta"]),
        )
        modelo.w = d["w"]
        return modelo
