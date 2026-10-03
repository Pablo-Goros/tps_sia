"""Retry of atomic replacement under transient locks.

    python -m unittest -v tps_sia.tp3.shared.tests.test_atomic_files
"""
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tps_sia.tp3.shared import atomic_files
from tps_sia.tp3.shared.atomic_files import replace
from tps_sia.tp3.shared.experiments import write_json
from tps_sia.tp3.shared.mlp import MLP


def flaky_replace(failures, error=PermissionError):
    """os.replace that raises `error` on the first `failures` calls."""
    real = os.replace
    calls = []

    def fake(source, target):
        calls.append((source, target))
        if len(calls) <= failures:
            raise error(13, 'Access is denied')
        real(source, target)
    return fake, calls


class ReplaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.source, self.target = self.dir / 'new.tmp', self.dir / 'file.txt'
        self.source.write_text('new')
        self.target.write_text('old')

    def tearDown(self):
        self.tmp.cleanup()

    def test_retries_permission_error_then_succeeds(self):
        fake, calls = flaky_replace(3)
        sleeps = []
        with mock.patch.object(atomic_files.os, 'replace', fake):
            replace(self.source, self.target, sleep=sleeps.append)
        self.assertEqual(len(calls), 4)
        self.assertEqual(sleeps, list(atomic_files.RETRY_DELAYS[:3]))
        self.assertEqual(self.target.read_text(), 'new')
        self.assertFalse(self.source.exists())

    def test_persistent_lock_raises_after_all_attempts(self):
        fake, calls = flaky_replace(100)
        with mock.patch.object(atomic_files.os, 'replace', fake):
            with self.assertRaises(PermissionError):
                replace(self.source, self.target, sleep=lambda _: None)
        self.assertEqual(len(calls), len(atomic_files.RETRY_DELAYS) + 1)
        self.assertEqual(self.target.read_text(), 'old')

    def test_other_errors_are_not_retried(self):
        fake, calls = flaky_replace(1, FileNotFoundError)
        with mock.patch.object(atomic_files.os, 'replace', fake):
            with self.assertRaises(FileNotFoundError):
                replace(self.source, self.target, sleep=lambda _: None)
        self.assertEqual(len(calls), 1)

    @unittest.skipUnless(os.name == 'nt', 'Windows file-sharing semantics')
    def test_real_lock_released_during_retries(self):
        handle = self.target.open('rb')     # Blocks replacement of the target on Windows.
        with self.assertRaises(PermissionError):
            os.replace(self.source, self.target)
        threading.Timer(0.3, handle.close).start()
        replace(self.source, self.target)
        self.assertEqual(self.target.read_text(), 'new')


class CallerTests(unittest.TestCase):
    def test_model_and_json_saves_survive_transient_locks(self):
        model = MLP([2, 3, 2], semilla=0)
        with tempfile.TemporaryDirectory() as tmp:
            for name, save in (('model.npz', model.guardar),
                               ('data.json', lambda path: write_json(path, {'a': 1}))):
                path = Path(tmp) / name
                fake, calls = flaky_replace(2)
                with mock.patch.object(atomic_files.os, 'replace', fake), \
                        mock.patch.object(atomic_files.time, 'sleep', lambda _: None):
                    save(path)
                self.assertEqual(len(calls), 3, name)
                self.assertEqual(sorted(p.name for p in Path(tmp).iterdir() if p.suffix == '.tmp'), [])
            loaded = MLP.cargar(Path(tmp) / 'model.npz')
            for a, b in zip(model.pesos, loaded.pesos):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(json.loads((Path(tmp) / 'data.json').read_text(encoding='utf-8')), {'a': 1})


if __name__ == '__main__':
    unittest.main(verbosity=2)
