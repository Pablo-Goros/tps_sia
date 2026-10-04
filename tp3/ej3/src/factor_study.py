"""Controlled data and configuration comparisons on the same ej3 validation."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tps_sia.tp3.shared.digit_dataset import cargar
from tps_sia.tp3.shared.experiments import (
    recorded_config, sha256_file, validate_config, write_json,
)
from tps_sia.tp3.shared.staged_search import execute_jobs, group_summary
from .dataset import image_hashes, labels_of
from .protocol import CONFIG, OUTPUT, TP3, Development, bind_output, read_json

PARTS = ('dataset', 'coverage', 'size', 'techniques')
FACTOR_OUTPUT = TP3 / 'ej3/results/factors'


def nested_subsets(indices, labels, fractions, seed):
    """Classwise shuffled prefixes preserve coverage and nest every subset."""
    indices = np.asarray(indices, dtype=np.int64)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(indices[labels[indices] == digit])
              for digit in np.unique(labels[indices])]
    return {fraction: np.concatenate([group[:max(1, int(len(group) * fraction))]
                                     for group in groups]) for fraction in fractions}


def eligible_indices(X, validation):
    reserved = set(image_hashes(validation))
    return np.asarray([i for i, key in enumerate(image_hashes(X)) if key not in reserved],
                      dtype=np.int64)


class FactorStudy:
    def __init__(self, config=CONFIG, output=FACTOR_OUTPUT, selection=OUTPUT / 'selection.json',
                 reference_selection=None, workers=1, pause_after=None, verbose=0, root=TP3):
        self.development = Development(config, reference_selection, root)
        d = self.development
        self.protocol = d.protocol
        self.spec = self.protocol['factor_study']
        selection_path = Path(selection).resolve()
        selected = read_json(selection_path)
        if selected.get('identity') != d.identity:
            raise ValueError('Ej3 selection belongs to a different protocol or references.')
        self.selected = d.config(selected['config'])
        self.output, self.workers = Path(output).resolve(), workers
        self.pause_after, self.verbose = pause_after, verbose
        digits = (d.root / self.spec['digits_dataset']).resolve()
        cache = (d.root / self.spec['digits_cache']).resolve()
        validate_config({'dataset': str(digits), 'cache': str(cache)})
        digest = sha256_file(digits)
        X, y = cargar(digits, cache)
        if sha256_file(digits) != digest:
            raise ValueError('digits dataset changed during loading.')
        eligible = eligible_indices(X, d.X[d.validation])
        if not len(eligible):
            raise ValueError('No digits samples remain outside the common validation.')
        digits_config = d.config(d.reference, dataset=str(digits), cache=str(cache), data={
            **d.data, 'train_indices': eligible.tolist(), 'dataset_sha256': digest})
        labels = labels_of(d.y)
        no_eight = d.train[labels[d.train] != 8]
        if not np.any(labels[d.train] == 8):
            raise ValueError('Coverage study requires digit 8 in the ej3 training partition.')
        reference = d.config(d.reference)
        self.variants = {
            'dataset': [('digits-eligible', digits_config), ('more-digits', reference)],
            'coverage': [('without-8', d.config(reference, data={**d.data, 'train_indices': no_eight.tolist()})),
                         ('with-8', reference)],
            'size': [(f'train-{fraction:g}', d.config(reference, data={**d.data, 'train_indices': subset.tolist()}))
                     for fraction, subset in nested_subsets(d.train, labels,
                         self.spec['subset_fractions'], self.spec['subset_seed']).items()],
            'techniques': [('ej2-config', reference), ('ej3-config', self.selected)],
        }
        self.identity = {**d.identity, 'study': 'required-factors',
                         'selection_sha256': sha256_file(selection_path), 'digits_sha256': digest}
        bind_output(self.output, self.identity)
        self.data_summary = {'validation_source': self.protocol['dataset'],
                             'validation_indices': d.validation.tolist(),
                             'validation_class_counts': d.y[d.validation].sum(axis=0).astype(int).tolist(),
                             'digits_original_samples': len(X), 'digits_eligible_samples': len(eligible),
                             'digits_excluded_indices': np.setdiff1d(np.arange(len(X)), eligible).tolist(),
                             'limitations': [
                                 'Dataset comparison changes images, size and class composition together.',
                                 'Removing digit 8 also reduces training size; this is a coverage ablation.',
                                 'Size subsets preserve classes approximately, not identical class proportions.',
                                 'Comparisons use development validation and do not prove final generalization.']}

    def jobs(self, part):
        from tps_sia.tp3.shared.staged_search import materialize
        return [(validate_config(materialize(config, self.development.root, seed)), label)
                for label, config in self.variants[part] for seed in self.spec['seeds']]

    def run_part(self, part, dry_run=False):
        path = self.output / f'{part}.json'
        if path.exists():
            value = read_json(path)
            if value['identity'] != self.identity:
                raise ValueError('Stored factor study belongs to another protocol.')
            return value
        jobs = self.jobs(part)
        if dry_run:
            return {'part': part, 'jobs': [{'label': label,
                     'config': recorded_config(config, self.development.root)} for config, label in jobs]}
        rows = execute_jobs(jobs, self.output, self.identity, self.development.root,
                            self.workers, self.pause_after, self.verbose)
        # Keep labels when two comparisons happen to use exactly the same config.
        summary = []
        for label, config in self.variants[part]:
            summary.extend(group_summary([r for r in rows if r['label'] == label], self.spec['seeds']))
        counts = []
        for label, config in self.variants[part]:
            source_X, source_y = cargar(config['dataset'], config['cache'])
            indices = config['data']['train_indices']
            counts.append({'label': label, 'train_samples': len(indices),
                           'train_class_counts': source_y[indices].sum(axis=0).astype(int).tolist()})
        value = {'part': part, 'identity': self.identity, 'runs': rows, 'groups': summary,
                 'data': self.data_summary, 'variants': counts,
                 'configurations': [{'label': label, 'config': recorded_config(config, self.development.root)}
                                    for label, config in self.variants[part]]}
        write_json(path, value)
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=(*PARTS, 'all'), required=True)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output-dir', type=Path, default=FACTOR_OUTPUT)
    parser.add_argument('--selection', type=Path, default=OUTPUT / 'selection.json')
    parser.add_argument('--reference-selection', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--pause-after', type=int)
    parser.add_argument('--verbose', type=int, default=10)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    study = FactorStudy(args.config, args.output_dir, args.selection, args.reference_selection,
                        args.workers, args.pause_after, args.verbose)
    parts = PARTS if args.part == 'all' else (args.part,)
    for part in parts:
        value = study.run_part(part, args.dry_run)
        if args.dry_run:
            import json
            print(json.dumps(value, indent=2, ensure_ascii=False))
        else:
            print(f'Saved {part} in {args.output_dir}')


if __name__ == '__main__':
    main()
