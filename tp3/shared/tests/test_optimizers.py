"""Chequeos del paso 4: python -m tps_sia.tp3.shared.tests.test_optimizers."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from tps_sia.tp3.shared.mlp import MLP
from tps_sia.tp3.shared.optimizers import Adam, Momentum, SGD, construir_optimizador


def assert_state_equal(test, actual, expected):
    test.assertEqual(set(actual), set(expected))
    for key in actual:
        if isinstance(actual[key], list):
            test.assertEqual(len(actual[key]), len(expected[key]))
            for a, b in zip(actual[key], expected[key]):
                np.testing.assert_array_equal(a, b)
        else:
            test.assertEqual(actual[key], expected[key])


def optimizers():
    return (SGD(0.1), Momentum(0.1, 0.5), Adam(0.1))


class OptimizerTests(unittest.TestCase):
    def test_sgd_and_momentum_manual_steps(self):
        for optimizer in (SGD(0.1), Momentum(0.1, 0.5)):
            parameters = [np.array([[1., -2.]]), np.array([0.5])]
            optimizer.paso(parameters, [np.array([[2., -4.]]), np.array([3.])])
            np.testing.assert_allclose(parameters[0], [[0.8, -1.6]])
            np.testing.assert_allclose(parameters[1], [0.2])
            optimizer.paso(parameters, [np.array([[-1., 2.]]), np.array([-1.])])
            if isinstance(optimizer, Momentum):
                np.testing.assert_allclose(parameters[0], [[0.8, -1.6]])
                np.testing.assert_allclose(parameters[1], [0.15])
                state = optimizer.export_state()
                np.testing.assert_array_equal(state["velocity"][0], [[0., 0.]])
                np.testing.assert_array_equal(state["velocity"][1], [0.5])
            else:
                np.testing.assert_allclose(parameters[0], [[0.9, -1.8]])
                np.testing.assert_allclose(parameters[1], [0.3])
            self.assertEqual(optimizer.updates, 2)

    def test_adam_manual_steps(self):
        # Beta y epsilon deliberadamente grandes permiten comprobar a mano
        # la corrección de sesgo y la posición de epsilon fuera de sqrt.
        optimizer = Adam(0.1, beta1=0.5, beta2=0.75, optimizer_epsilon=0.1)
        parameters = [np.array([1.]), np.array([-2.])]
        optimizer.paso(parameters, [np.array([2.]), np.array([-4.])])
        expected = [1 - 0.1 * 2 / 2.1, -2 + 0.1 * 4 / 4.1]
        np.testing.assert_allclose([p.item() for p in parameters], expected)
        optimizer.paso(parameters, [np.array([-2.]), np.array([0.])])
        # m=(-1/2,-1), v=(7/4,3), m_hat=(-2/3,-4/3), v_hat=(4,48/7).
        expected[0] += 0.1 * (2 / 3) / 2.1
        expected[1] += 0.1 * (4 / 3) / (np.sqrt(48 / 7) + 0.1)
        np.testing.assert_allclose([p.item() for p in parameters], expected, rtol=1e-14)
        state = optimizer.export_state()
        np.testing.assert_allclose([v.item() for v in state["first_moment"]], [-0.5, -1])
        np.testing.assert_allclose([v.item() for v in state["second_moment"]], [1.75, 3])
        self.assertEqual(state["updates"], 2)  # Un contador por lote, no por tensor.

    def test_defaults_and_factory(self):
        self.assertEqual(SGD().configuracion(), {"nombre": "sgd", "eta": 0.01})
        self.assertEqual(Adam().configuracion(), {
            "name": "adam", "learning_rate": 0.001, "beta1": 0.9,
            "beta2": 0.999, "optimizer_epsilon": 1e-8,
        })
        self.assertEqual(Momentum().momentum, 0.9)
        for optimizer in optimizers():
            config = optimizer.configuracion()
            restored = construir_optimizador(config)
            self.assertIs(type(restored), type(optimizer))
            self.assertEqual(restored.configuracion(), config)
        self.assertEqual(construir_optimizador({"name": "sgd", "learning_rate": 0.2}).eta, 0.2)
        for name, cls in (("sgd", SGD), ("momentum", Momentum), ("adam", Adam)):
            self.assertIsInstance(construir_optimizador({"name": name}), cls)
        for config in (None, {}, {"name": []}, {"name": "other"},
                       {"nombre": "sgd"}, {"name": "adam", "epsilon": 1e-8},
                       {"name": "momentum", "momentum": 1},
                       {"name": "sgd", "learning_rate": True}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                construir_optimizador(config)

    def test_invalid_hyperparameters(self):
        for bad in (0, -1, np.inf, np.nan, True, "0.1", 1j):
            for cls, keyword in ((SGD, "eta"), (Momentum, "learning_rate"),
                                 (Adam, "learning_rate"), (Adam, "optimizer_epsilon")):
                with self.subTest(cls=cls, keyword=keyword, bad=bad), self.assertRaises(ValueError):
                    cls(**{keyword: bad})
        for bad in (-0.1, 1, 2, np.inf, np.nan, True, "0.9"):
            for cls, keyword in ((Momentum, "momentum"), (Adam, "beta1"), (Adam, "beta2")):
                with self.subTest(cls=cls, keyword=keyword, bad=bad), self.assertRaises(ValueError):
                    cls(**{keyword: bad})
        # Cero es válido para los factores de decaimiento.
        parameters = [np.array([1., -1.])]
        Momentum(momentum=0).paso(parameters, [np.array([1., -1.])])
        Adam(beta1=0, beta2=0).paso(parameters, [np.array([0., 0.])])

    def test_invalid_gradients_are_atomic(self):
        for optimizer in optimizers():
            parameters = [np.array([[1., 2.]]), np.array([3.])]
            good = [np.ones((1, 2)), np.ones(1)]
            optimizer.paso(parameters, good)  # Probar también estado ya inicializado.
            before = [p.copy() for p in parameters]
            state = optimizer.export_state()
            invalid = [[], [good[0]], good + [np.ones(1)],
                       [good[0], np.ones((1, 1))], [good[0], np.array([np.nan])],
                       [good[0], np.array([np.inf])], [good[0], np.array([1j])],
                       [good[0], np.array(["x"], dtype=object)]]
            for gradients in invalid:
                with self.subTest(optimizer=type(optimizer), gradients=gradients), self.assertRaises(ValueError):
                    optimizer.paso(parameters, gradients)
                for p, original in zip(parameters, before):
                    np.testing.assert_array_equal(p, original)
                assert_state_equal(self, optimizer.export_state(), state)
            if isinstance(optimizer, SGD):
                optimizer.eta = np.nan
            else:
                optimizer.learning_rate = np.nan
            with self.assertRaises(ValueError):
                optimizer.paso(parameters, good)
            for p, original in zip(parameters, before):
                np.testing.assert_array_equal(p, original)
            self.assertEqual(optimizer.updates, state["updates"])

    def test_invalid_parameters_and_overflow_are_atomic(self):
        for optimizer in optimizers():
            for bad in (np.array([1]), np.array([np.nan]), np.array([np.inf]), np.array([1j])):
                parameters = [np.array([1.]), bad]
                before = [p.copy() for p in parameters]
                with self.assertRaises(ValueError):
                    optimizer.paso(parameters, [np.ones(1), np.ones(1)])
                for p, original in zip(parameters, before):
                    np.testing.assert_array_equal(p, original)
                self.assertEqual(optimizer.updates, 0)
            readonly = np.array([1.])
            readonly.flags.writeable = False
            with self.assertRaises(ValueError):
                optimizer.paso([np.array([1.]), readonly], [np.ones(1), np.ones(1)])
            with self.assertRaises(ValueError):
                optimizer.paso([], [])
        for optimizer in (SGD(1), Momentum(1), Adam(1)):
            parameters = [np.array([1.]), np.array([1e308])]
            before = [p.copy() for p in parameters]
            state = optimizer.export_state()
            with self.assertRaises(ValueError):
                optimizer.paso(parameters, [np.ones(1), np.array([-1e308])])
            for p, original in zip(parameters, before):
                np.testing.assert_array_equal(p, original)
            assert_state_equal(self, optimizer.export_state(), state)

    def test_float32_parameters_and_zero_gradients(self):
        for optimizer in optimizers():
            parameters = [np.array([1., -2.], dtype=np.float32)]
            optimizer.paso(parameters, [np.zeros(2)])
            np.testing.assert_array_equal(parameters[0], [1., -2.])
            optimizer.paso(parameters, [np.array([2., -3.])])
            self.assertEqual(parameters[0].dtype, np.dtype("float32"))
            self.assertTrue(np.all(np.isfinite(parameters[0])))

    def test_state_roundtrip_and_copy_isolation(self):
        for optimizer in optimizers():
            parameters = [np.array([[1., 2.]]), np.array([-1.])]
            gradients = [np.array([[0.3, -0.2]]), np.array([0.4])]
            restored = construir_optimizador(optimizer.configuracion())
            restored.restore_state(optimizer.export_state(), parameters)
            self.assertEqual(restored.updates, 0)
            for _ in range(3):
                optimizer.paso(parameters, gradients)
            state = optimizer.export_state()
            restored.restore_state(state, parameters)
            copy_parameters = [p.copy() for p in parameters]
            snapshot = deepcopy(state)
            for value in state.values():
                if isinstance(value, list):
                    for tensor in value:
                        tensor[:] = 123  # Ni export ni restore comparten memoria.
            assert_state_equal(self, optimizer.export_state(), snapshot)
            assert_state_equal(self, restored.export_state(), snapshot)
            for _ in range(4):
                optimizer.paso(parameters, gradients)
                restored.paso(copy_parameters, gradients)
            for actual, expected in zip(copy_parameters, parameters):
                np.testing.assert_array_equal(actual, expected)
            assert_state_equal(self, restored.export_state(), optimizer.export_state())

    def test_invalid_state_restore_is_atomic(self):
        for optimizer in optimizers():
            parameters = [np.ones((2, 2)), np.ones(2)]
            optimizer.paso(parameters, [np.ones_like(p) for p in parameters])
            before = [p.copy() for p in parameters]
            state = optimizer.export_state()
            variants = []
            for key, value in (("version", 2), ("updates", -1), ("updates", True),
                               ("updates", 1.5), ("config", {"name": "other"}),
                               ("config", {"name": "sgd", "learning_rate": 0.7})):
                broken = deepcopy(state)
                broken[key] = value
                variants.append(broken)
            broken = deepcopy(state)
            broken["unknown"] = 0
            variants.append(broken)
            for key, tensors in state.items():
                if isinstance(tensors, list):
                    for bad in ([], [np.ones((2, 2)), np.ones(3)],
                                [np.ones((2, 2)), np.array([np.nan, 1.])]):
                        broken = deepcopy(state)
                        broken[key] = bad
                        variants.append(broken)
                    broken = deepcopy(state)
                    broken["updates"] = 0
                    variants.append(broken)
            if isinstance(optimizer, Adam):
                broken = deepcopy(state)
                broken["second_moment"][-1][0] = -1
                variants.append(broken)
            for broken in variants:
                with self.subTest(optimizer=type(optimizer), broken=broken), self.assertRaises(ValueError):
                    optimizer.restore_state(broken, parameters)
                assert_state_equal(self, optimizer.export_state(), state)
                for parameter, original in zip(parameters, before):
                    np.testing.assert_array_equal(parameter, original)
            if not isinstance(optimizer, SGD):
                with self.assertRaises(ValueError):
                    optimizer.paso([np.ones(3)], [np.ones(3)])
                assert_state_equal(self, optimizer.export_state(), state)

    def test_all_mlp_weights_and_biases_updated(self):
        X = np.array([[0.2, -0.3], [0.8, 0.7], [-0.4, 0.6]])
        y = np.eye(2)[[0, 1, 1]]
        for optimizer in optimizers():
            model = MLP([2, 3, 2, 2], optimizador=optimizer, semilla=3)
            gradients = model.backprop(X, y)
            before = [p.copy() for p in model.parametros]
            optimizer.paso(model.parametros, gradients)
            for p, original, g in zip(model.parametros, before, gradients):
                expected = original - 0.1 * g
                if isinstance(optimizer, Adam):
                    expected = original - 0.1 * g / (np.abs(g) + 1e-8)
                np.testing.assert_allclose(p, expected, rtol=1e-13, atol=1e-15)
                self.assertFalse(np.array_equal(p, original))
            state = optimizer.export_state()
            for tensors in state.values():
                if isinstance(tensors, list):
                    self.assertEqual([t.shape for t in tensors], [p.shape for p in model.parametros])

    def test_mlp_checkpoint_resumes_exactly(self):
        X = np.array([[-1., 1.], [1., -1.], [-1., -1.], [1., 1.]])
        y = np.eye(2)[[1, 1, 0, 0]]
        with TemporaryDirectory() as directory:
            for optimizer in optimizers():
                model = MLP([2, 3, 2], tamano_lote=3, semilla=17, optimizador=optimizer)
                uninterrupted = MLP([2, 3, 2], tamano_lote=3, semilla=17,
                                    optimizador=construir_optimizador(optimizer.configuracion()))
                model.entrenar(X, y, epocas=3)
                path = Path(directory) / "model.npz"
                model.guardar(path)
                with np.load(path, allow_pickle=False) as data:
                    self.assertEqual(json.loads(str(data["configuracion"]))["version"], 3)
                restored = MLP.cargar(path)
                np.testing.assert_array_equal(restored.predecir(X), model.predecir(X))
                assert_state_equal(self, restored.optimizador.export_state(), optimizer.export_state())
                restored.entrenar(X, y, epocas=4)
                uninterrupted.entrenar(X, y, epocas=7)
                for a, b in zip(restored.parametros, uninterrupted.parametros):
                    np.testing.assert_array_equal(a, b)
                self.assertEqual(restored.optimizador.updates, 14)
                self.assertEqual(restored.historia.costo, uninterrupted.historia.costo)
                # También se puede guardar/cargar antes del primer paso.
                fresh = MLP([2, 2], optimizador=construir_optimizador(optimizer.configuracion()))
                fresh.guardar(path)
                assert_state_equal(self, MLP.cargar(path).optimizador.export_state(),
                                   fresh.optimizador.export_state())

    def test_legacy_sgd_and_corrupt_checkpoint(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model = MLP([2, 3, 2], optimizador=SGD(0.2))
            model.guardar(path)
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            config = json.loads(str(arrays["configuracion"]))
            config["version"] = 1
            config.pop("optimizer_state")
            arrays["configuracion"] = json.dumps(config)
            np.savez(path, **arrays)
            restored = MLP.cargar(path)
            for a, b in zip(restored.parametros, model.parametros):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(restored.optimizador.eta, 0.2)
            # Un checkpoint Adam con un momento inválido no se carga parcialmente.
            model = MLP([2, 3, 2], optimizador=Adam())
            model.entrenar(np.eye(2), np.eye(2), epocas=1)
            model.guardar(path)
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            arrays["optimizer_second_moment_3"] = np.full(2, -1.)
            np.savez(path, **arrays)
            with self.assertRaises(ValueError):
                MLP.cargar(path)

    def test_xor_with_momentum_and_adam(self):
        X = np.array([[-1., 1.], [1., -1.], [-1., -1.], [1., 1.]])
        y = np.array([1., 1., 0., 0.])
        for architecture in ([2, 2, 1], [2, 3, 2, 1]):
            for optimizer in (Momentum(0.1), Adam(0.03)):
                with self.subTest(architecture=architecture, optimizer=type(optimizer)):
                    model = MLP(architecture, salida="logistica", tamano_lote=2,
                                semilla=0, optimizador=optimizer)
                    history = model.entrenar(X, y, epocas=2000, epsilon=0.001)
                    np.testing.assert_array_equal(model.predecir_clases(X), y)
                    self.assertLess(history.costo[-1], 0.001)


if __name__ == "__main__":
    unittest.main()
