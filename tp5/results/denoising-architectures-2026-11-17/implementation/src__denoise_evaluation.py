"""Numerical recovery evidence; deliberately produces no figures."""
import csv
import numpy as np
from .metrics import reconstruction_metrics


def recovery_metrics(targets, inputs, predictions):
    result = reconstruction_metrics(targets, predictions)
    before = reconstruction_metrics(targets, inputs)
    differences = np.asarray(before['pixel_errors']) - result['pixel_errors']
    result.update(input_pixel_errors=before['pixel_errors'],
                  input_mean_pixel_errors=before['mean_pixel_errors'],
                  mean_error_reduction=float(differences.mean()),
                  improved_fraction=float(np.mean(differences > 0)),
                  worsened_fraction=float(np.mean(differences < 0)),
                  unchanged_fraction=float(np.mean(differences == 0)))
    return result


def grouped_metrics(collection, predictions, by_character=False):
    rows = []
    for p in np.unique(collection['levels']):
        characters = np.unique(collection['character_indices']) if by_character else [None]
        for character in characters:
            mask = collection['levels'] == p
            if character is not None:
                mask &= collection['character_indices'] == character
            values = recovery_metrics(collection['targets'][mask], collection['inputs'][mask], predictions[mask])
            values.pop('pixel_errors'); values.pop('input_pixel_errors')
            rows.append({'level': float(p), 'character_index': None if character is None else int(character),
                         'count': int(mask.sum()), **values})
    return rows


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
