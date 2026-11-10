"""Public dataset behavior agreed in the TP5 specification."""
import unittest
import hashlib
from pathlib import Path
import tempfile
import subprocess
import sys

import numpy as np

from src.dataset import load_dataset, DEFAULT_DATASET


class DatasetTests(unittest.TestCase):
    def test_comments_do_not_contribute_pixels_or_patterns(self):
        text = Path('data/font.h').read_text(encoding='utf-8')
        text = text.replace('= {', '= { /* {0xffff, 999} */', 1)
        text = text.replace('0x04,', '0x04 /* 0xffff */ ,', 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'comments.h'
            path.write_text(text, encoding='utf-8')
            actual = load_dataset(path)
            np.testing.assert_array_equal(actual.images, load_dataset().images)

    def test_cli_produces_dataset_sheet_and_identity(self):
        import json
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-m', 'src.dataset', '--output', directory],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = Path(directory)
            with Image.open(output / 'dataset.png') as image:
                self.assertGreater(image.width, 1000)
                self.assertGreater(image.height, 500)
            manifest = json.loads((output / 'dataset.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['labels'], load_dataset().labels)
            self.assertEqual(manifest['sha256'], load_dataset().sha256)

    def test_malformed_initializers_are_rejected(self):
        original = Path('data/font.h').read_text(encoding='utf-8')
        cases = [
            (original.replace('{0x04, 0x04, 0x02, 0x00, 0x00, 0x00, 0x00},', ''), '32 patterns'),
            (original.replace('0x04, 0x04, 0x02,', '0x04, 0x02,', 1), '7 rows'),
            (original.replace('0x04,', '0x20,', 1), '0..31'),
            (original.replace('0x04,', '-1,', 1), '0..31'),
            (original[:-5], 'complete'),
            (original.replace('= {', '= { 123,', 1), 'Unexpected'),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.h'
            for text, message in cases:
                with self.subTest(message=message):
                    path.write_text(text, encoding='utf-8')
                    with self.assertRaisesRegex(ValueError, message):
                        load_dataset(path)

    def test_original_patterns_keep_orientation_and_identity(self):
        dataset = load_dataset()
        self.assertEqual(dataset.images.shape, (32, 7, 5))
        self.assertEqual(dataset.vectors.shape, (32, 35))
        self.assertEqual(dataset.vectors.dtype, np.dtype(float))
        self.assertTrue(np.isin(dataset.images, [0, 1]).all())
        np.testing.assert_array_equal(dataset.images[0], [
            [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]])
        np.testing.assert_array_equal(dataset.images[2, 0], [1, 0, 0, 0, 0])
        np.testing.assert_array_equal(dataset.images[-1], np.ones((7, 5)))
        np.testing.assert_array_equal(dataset.vectors, dataset.images.reshape(32, 35))
        self.assertEqual(dataset.labels[0], '0x60 `')
        self.assertEqual(dataset.labels[-1], '0x7f DEL')
        raw = DEFAULT_DATASET.read_bytes()
        self.assertEqual(dataset.sha256, hashlib.sha256(raw).hexdigest())
        # Git checkouts may use CRLF; validate the known content independently.
        self.assertEqual(hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest(),
                         '095b431bdfa2d228eeaa337fb45cd963088cc13380c189e01436e446767502f8')


if __name__ == '__main__':
    unittest.main()
