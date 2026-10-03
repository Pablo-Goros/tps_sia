"""Data checks and grouped development split for ej3 (pure functions).

Images are identified by the SHA-256 of their float32 pixel bytes, so two rows
are "identical" only when every pixel value is equal. The grouped split keeps
all copies of an image on the same side. With no duplicates it consumes the
random generator exactly like shared.digit_dataset.particionar, so it returns
the same indices as the shared runner for the same seed and fraction.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np

N_CLASSES = 10


def image_hashes(X: np.ndarray) -> list[str]:
    """SHA-256 per row of float32 pixels; -0.0 is normalized to 0.0."""
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float32) + np.float32(0.0))
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in X]


def labels_of(y: np.ndarray) -> np.ndarray:
    """Integer labels from one-hot targets or from a label vector."""
    y = np.asarray(y)
    return y.argmax(axis=1) if y.ndim == 2 else y.astype(np.int64)


def class_counts(labels: np.ndarray, n_classes: int = N_CLASSES) -> dict:
    counts = np.bincount(labels, minlength=n_classes)
    total = int(counts.sum())
    return {
        'counts': {str(d): int(c) for d, c in enumerate(counts)},
        'proportions': {str(d): (float(c) / total if total else 0.0) for d, c in enumerate(counts)},
        'missing_classes': [d for d, c in enumerate(counts) if c == 0],
    }


def image_groups(hashes: list[str]) -> list[list[int]]:
    """Row indices per distinct image, ordered by first occurrence."""
    groups: dict[str, list[int]] = {}
    for index, key in enumerate(hashes):
        groups.setdefault(key, []).append(index)
    return list(groups.values())


def duplicate_summary(hashes: list[str], labels: np.ndarray) -> dict:
    """Exact duplicates and label conflicts (same image, different labels)."""
    groups = [g for g in image_groups(hashes) if len(g) > 1]
    conflicts = [g for g in groups if len({int(labels[i]) for i in g}) > 1]
    consistent = [g for g in groups if g not in conflicts]
    return {
        'unique_images': len(set(hashes)),
        'duplicate_groups': len(groups),
        'rows_in_duplicate_groups': sum(len(g) for g in groups),
        'extra_copies': sum(len(g) - 1 for g in groups),
        'largest_group': max((len(g) for g in groups), default=1),
        'duplicate_groups_by_class': {str(k): v for k, v in sorted(
            Counter(int(labels[g[0]]) for g in consistent).items())},
        'label_conflicts': {
            'groups': len(conflicts),
            'rows': sum(len(g) for g in conflicts),
            'details': [{'rows': [int(i) for i in g],
                         'labels': sorted({int(labels[i]) for i in g})} for g in conflicts],
        },
    }


def overlap_summary(hashes: list[str], labels: np.ndarray,
                    reference_hashes: list[str], reference_labels: np.ndarray,
                    n_classes: int = N_CLASSES) -> dict:
    """Rows of a dataset whose image also appears in a reference dataset."""
    reference: dict[str, set[int]] = {}
    for key, label in zip(reference_hashes, reference_labels):
        reference.setdefault(key, set()).add(int(label))
    same = np.zeros(n_classes, dtype=np.int64)
    different = np.zeros(n_classes, dtype=np.int64)
    new = np.zeros(n_classes, dtype=np.int64)
    for key, label in zip(hashes, labels):
        if key not in reference:
            new[label] += 1
        elif reference[key] == {int(label)}:
            same[label] += 1
        else:
            different[label] += 1
    own = set(hashes)
    reference_rows_found = sum(1 for key in reference_hashes if key in own)
    return {
        'rows_with_identical_image': int(same.sum() + different.sum()),
        'same_label': int(same.sum()),
        'different_label': int(different.sum()),
        'new_rows': int(new.sum()),
        'by_class': {str(d): {'same_label': int(same[d]), 'different_label': int(different[d]),
                              'new': int(new[d])} for d in range(n_classes)},
        'reference_rows': len(reference_hashes),
        'reference_rows_found': reference_rows_found,
        'reference_is_subset': reference_rows_found == len(reference_hashes),
    }


def hash_overlap_counts(hashes: list[str], others: dict[str, list[str]]) -> dict:
    """Counts only: rows of `hashes` whose image appears in each other dataset."""
    sets = {name: set(values) for name, values in others.items()}
    result = {f'rows_also_in_{name}': sum(1 for key in hashes if key in found)
              for name, found in sets.items()}
    result['rows_also_in_all'] = sum(1 for key in hashes if all(key in s for s in sets.values()))
    result['rows'] = len(hashes)
    return result


def grouped_stratified_split(labels: np.ndarray, hashes: list[str], seed: int = 42,
                             validation_fraction: float = 0.2) -> dict:
    """Stratified train/validation indices with identical images kept together.

    Groups whose copies carry different labels are excluded from both sides,
    since they have no single target. Per class, groups are shuffled and moved
    to validation until it holds round(n * fraction) samples, keeping at least
    one sample on each side; large groups can make the proportion approximate.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) != len(hashes) or len(labels) == 0:
        raise ValueError('labels and hashes must be non-empty and of equal length.')
    if not np.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between 0 and 1.')
    rng = np.random.default_rng(seed)
    groups, excluded = [], []
    for group in image_groups(hashes):
        group_labels = {int(labels[i]) for i in group}
        if len(group_labels) > 1:
            excluded.append({'rows': group, 'labels': sorted(group_labels),
                             'reason': 'identical image with different labels'})
        else:
            groups.append(group)
    group_label = np.array([labels[g[0]] for g in groups], dtype=np.int64)
    train, validation = [], []
    for digit in np.unique(group_label):
        class_groups = rng.permutation(np.flatnonzero(group_label == digit))
        if len(class_groups) < 2:
            raise ValueError(f'Class {digit} needs at least two distinct images to stratify.')
        n = sum(len(groups[g]) for g in class_groups)
        target = min(n - 1, max(1, int(n * validation_fraction + 0.5)))
        taken = 0
        for position, g in enumerate(class_groups):
            remaining = len(class_groups) - position - 1
            if taken >= target or remaining == 0:
                train.extend(i for g2 in class_groups[position:] for i in groups[g2])
                break
            validation.extend(groups[g])
            taken += len(groups[g])
    train_indices = rng.permutation(np.array(train, dtype=np.int64))
    validation_indices = rng.permutation(np.array(validation, dtype=np.int64))
    return {'train': train_indices, 'validation': validation_indices, 'excluded': excluded}


def split_summary(labels: np.ndarray, split: dict, n_classes: int = N_CLASSES) -> dict:
    train, validation = split['train'], split['validation']
    return {
        'train_size': int(len(train)),
        'validation_size': int(len(validation)),
        'excluded_rows': sum(len(e['rows']) for e in split['excluded']),
        'train': class_counts(labels[train], n_classes),
        'validation': class_counts(labels[validation], n_classes),
    }
