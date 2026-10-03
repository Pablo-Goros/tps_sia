"""Checks of the ej3 data step with synthetic data only.

No real CSV (more_digits, digits or digits_test) is read. The end-to-end test
writes small synthetic CSVs to a temporary directory and points the command
module at them.

    python -m unittest -v tps_sia.tp3.ej3.tests.test_dataset
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tps_sia.tp3.shared.digit_dataset import particionar
from tps_sia.tp3.ej3.src import data_exploration
from tps_sia.tp3.ej3.src.dataset import (class_counts, duplicate_summary, grouped_stratified_split,
                                         hash_overlap_counts, image_hashes, overlap_summary,
                                         split_summary)


def synthetic(per_class=20, classes=(0, 1, 2, 3, 4, 5, 6, 7, 9), seed=0):
    """Distinct random images per class (8 absent by default)."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.array(classes), per_class)
    X = rng.integers(0, 11, size=(len(labels), 784)).astype(np.float32) / 10
    return X, labels


def with_copies(X, labels, copies):
    """Append copies: {source row: number of extra copies}."""
    rows = [i for i, n in copies.items() for _ in range(n)]
    return np.vstack([X, X[rows]]), np.concatenate([labels, labels[rows]])


class SplitTests(unittest.TestCase):
    def test_duplicate_groups_never_cross(self):
        X, labels = synthetic()
        X, labels = with_copies(X, labels, {0: 3, 5: 1, 21: 4, 40: 2, 100: 5, 170: 1})
        hashes = image_hashes(X)
        for seed in range(20):
            split = grouped_stratified_split(labels, hashes, seed)
            train = {hashes[i] for i in split['train']}
            validation = {hashes[i] for i in split['validation']}
            self.assertFalse(train & validation, f'seed {seed}: an image crosses train/validation')
            self.assertEqual(len(split['train']) + len(split['validation']), len(labels))

    def test_approximate_stratification(self):
        X, labels = synthetic(per_class=50)
        X, labels = with_copies(X, labels, {0: 2, 60: 3, 120: 1})
        split = grouped_stratified_split(labels, image_hashes(X), 42, 0.2)
        for digit in np.unique(labels):
            n = np.sum(labels == digit)
            in_validation = np.sum(labels[split['validation']] == digit)
            self.assertGreater(in_validation, 0)
            self.assertLess(in_validation, n)
            self.assertLessEqual(abs(in_validation / n - 0.2), 0.08, f'class {digit}')

    def test_deterministic_with_same_seed(self):
        X, labels = synthetic()
        X, labels = with_copies(X, labels, {3: 2, 50: 1})
        hashes = image_hashes(X)
        a = grouped_stratified_split(labels, hashes, 42)
        b = grouped_stratified_split(labels, hashes, 42)
        c = grouped_stratified_split(labels, hashes, 7)
        np.testing.assert_array_equal(a['train'], b['train'])
        np.testing.assert_array_equal(a['validation'], b['validation'])
        self.assertFalse(np.array_equal(np.sort(a['validation']), np.sort(c['validation'])))

    def test_without_duplicates_matches_shared_partition(self):
        X, labels = synthetic(per_class=30, classes=tuple(range(10)))
        y = np.eye(10, dtype=np.float32)[labels]
        split = grouped_stratified_split(labels, image_hashes(X), 42, 0.2)
        *_, train, validation = particionar(X, y, 42, return_indices=True, validation_fraction=0.2)
        np.testing.assert_array_equal(split['train'], train)
        np.testing.assert_array_equal(split['validation'], validation)

    def test_label_conflicts_detected_and_excluded(self):
        X, labels = synthetic()
        X, labels = with_copies(X, labels, {0: 1, 30: 2, 45: 1})
        n = len(labels)
        labels = labels.copy()
        labels[n - 4] = 7          # copy of row 0 (class 0) relabelled as 7
        labels[n - 1] = 3          # copy of row 45 (class 2) relabelled as 3
        hashes = image_hashes(X)
        summary = duplicate_summary(hashes, labels)
        self.assertEqual(summary['duplicate_groups'], 3)
        self.assertEqual(summary['label_conflicts']['groups'], 2)
        self.assertEqual([d['labels'] for d in summary['label_conflicts']['details']], [[0, 7], [2, 3]])
        split = grouped_stratified_split(labels, hashes, 42)
        excluded = sorted(i for e in split['excluded'] for i in e['rows'])
        self.assertEqual(excluded, sorted([0, n - 4, 45, n - 1]))
        used = set(split['train'].tolist()) | set(split['validation'].tolist())
        self.assertFalse(used & set(excluded))
        self.assertEqual(len(used) + len(excluded), n)

    def test_missing_class_detected(self):
        X, labels = synthetic()
        counts = class_counts(labels)
        self.assertEqual(counts['missing_classes'], [8])
        self.assertEqual(counts['counts']['8'], 0)
        summary = split_summary(labels, grouped_stratified_split(labels, image_hashes(X), 42))
        self.assertEqual(summary['train']['missing_classes'], [8])
        self.assertEqual(summary['validation']['missing_classes'], [8])

    def test_class_with_a_single_image_is_rejected(self):
        X, labels = synthetic(per_class=10, classes=(0, 1))
        X, labels = np.vstack([X, X[:1] + 0.05]), np.concatenate([labels, [2]])
        X, labels = with_copies(X, labels, {len(labels) - 1: 3})   # class 2: one image, four copies
        with self.assertRaisesRegex(ValueError, 'Class 2'):
            grouped_stratified_split(labels, image_hashes(X), 42)


class OverlapTests(unittest.TestCase):
    def test_overlap_counts(self):
        X, labels = synthetic(per_class=5)
        reference_X = np.vstack([X[:10], X[20:25]])
        reference_labels = np.concatenate([labels[:10], labels[20:25]])
        reference_labels[0] = 9                      # same image, different label
        result = overlap_summary(image_hashes(X), labels, image_hashes(reference_X), reference_labels)
        self.assertEqual(result['rows_with_identical_image'], 15)
        self.assertEqual(result['same_label'], 14)
        self.assertEqual(result['different_label'], 1)
        self.assertEqual(result['new_rows'], len(labels) - 15)
        self.assertTrue(result['reference_is_subset'])

    def test_negative_zero_is_the_same_image(self):
        a = np.zeros((1, 784), dtype=np.float32)
        self.assertEqual(image_hashes(a), image_hashes(-a))

    def test_integrity_counts_only(self):
        X, _ = synthetic(per_class=3)
        result = hash_overlap_counts(image_hashes(X[:6]), {'a': image_hashes(X[:4]), 'b': image_hashes(X[2:])})
        self.assertEqual(result, {'rows_also_in_a': 4, 'rows_also_in_b': 4, 'rows_also_in_all': 2, 'rows': 6})


def write_csv(path, X, labels):
    with path.open('w', encoding='utf-8', newline='') as target:
        target.write('label,image\n')
        for label, row in zip(labels, X):
            target.write(f'{int(label)},"{[float(v) for v in row]}"\n')


class CommandTests(unittest.TestCase):
    def test_outputs_are_portable_and_test_stays_out_of_the_split(self):
        X, labels = synthetic(per_class=12, seed=1)
        X, labels = with_copies(X, labels, {0: 1, 30: 2})
        X_digits, labels_digits = X[:40], labels[:40]
        X_test = np.vstack([X[:3], synthetic(per_class=1, seed=2)[0][:5]])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'tp3'
            data, output = root / 'data', root / 'ej3' / 'results'
            data.mkdir(parents=True)
            write_csv(data / 'more_digits.csv', X, labels)
            write_csv(data / 'digits.csv', X_digits, labels_digits)
            write_csv(data / 'digits_test.csv', X_test, np.zeros(len(X_test), dtype=int))
            with mock.patch.multiple(data_exploration, TP3=root, MORE_DIGITS=data / 'more_digits.csv',
                                     DIGITS=data / 'digits.csv', DIGITS_TEST=data / 'digits_test.csv'):
                data_exploration.main(['--output-dir', str(output), '--cache-dir', str(root / 'ej3' / 'cache')])
            manifest = json.loads((output / 'split_manifest.json').read_text(encoding='utf-8'))
            report = json.loads((output / 'data_report.json').read_text(encoding='utf-8'))
            with np.load(output / 'split_indices.npz') as saved:
                train, validation = saved['train'], saved['validation']
            text = '\n'.join(p.read_text(encoding='utf-8') for p in output.glob('*.json'))
            text += (output / 'data_report.md').read_text(encoding='utf-8')

        self.assertNotIn(str(Path(tmp)), text)
        self.assertNotIn(Path(tmp).as_posix(), text)
        self.assertEqual(manifest['dataset']['path'], 'data/more_digits.csv')
        for value in (manifest['dataset']['path'], report['digits']['path'],
                      report['integrity_check_test_overlap']['path']):
            self.assertFalse(Path(value).is_absolute())
        self.assertEqual(manifest['train_size'] + manifest['validation_size'], len(labels))
        self.assertLess(max(train.max(), validation.max()), len(labels))
        self.assertEqual(manifest['train']['missing_classes'], [8])
        self.assertEqual(report['duplicates']['duplicate_groups'], 2)
        integrity = report['integrity_check_test_overlap']
        self.assertEqual((integrity['rows'], integrity['rows_also_in_more_digits'],
                          integrity['rows_also_in_digits']), (8, 3, 3))


if __name__ == '__main__':
    unittest.main(verbosity=2)
