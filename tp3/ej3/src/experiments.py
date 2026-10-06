"""Run one ej3 development stage or an explicit experiment configuration."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from tps_sia.tp3.shared.experiments import (
    config_identity, fingerprint, recorded_config, write_json,
)
from tps_sia.tp3.shared.staged_search import execute_jobs, group_summary, rank
from .protocol import CONFIG, OUTPUT, TP3, Development, bind_output, read_json, joint_architectures

STAGES = ('controls', 'rates', 'architectures', 'batches', 'confirmation')


class Search:
    def __init__(self, config=CONFIG, output=OUTPUT, workers=1,
                 reference_selection=None, pause_after=None, verbose=0, root=TP3):
        self.development = Development(config, reference_selection, root)
        self.protocol = self.development.protocol
        self.output, self.workers = Path(output).resolve(), workers
        self.pause_after, self.verbose = pause_after, verbose
        self.identity = self.development.identity
        bind_output(self.output, self.identity)

    def stage_path(self, stage):
        return self.output / f'stage_{stage}.json'

    def load_stage(self, stage):
        path = self.stage_path(stage)
        if not path.exists():
            raise ValueError(f'Run --stage {stage} first.')
        value = read_json(path)
        if value['identity'] != self.identity:
            raise ValueError(f'{stage} belongs to another protocol.')
        return value

    def run_labeled(self, jobs):
        return execute_jobs(jobs, self.output, self.identity, self.development.root,
                            self.workers, self.pause_after, self.verbose)

    def planned_jobs(self, stage):
        d, p = self.development, self.protocol
        if stage == 'controls':
            return [(d.baseline, 'baseline'), (d.reference, 'ej2-reference')]
        if stage == 'joint':
            if 'joint' not in p['stages']:
                raise ValueError('Joint search requires a new protocol with stages.joint.')
            spec = p['stages']['joint']
            return [(d.config(d.reference, architecture=architecture,
                              strategy='mini_batch', batch_size=size,
                              optimizer={**d.reference['optimizer'], 'learning_rate': rate},
                              **({'l2': strength} if strength else {}),
                              **({'augmentation': {'name': 'translation', 'max_shift': shift}}
                                 if shift else {})),
                     f'joint-rate-{rate:g}-batch-{size}' +
                     (f'-l2-{strength:g}' if 'l2_values' in spec else '') +
                     (f'-shift-{shift}' if 'translation_shifts' in spec else '') +
                     ('-architecture-' + '-'.join(map(str, architecture)) if 'architectures' in spec else ''))
                    for rate in spec['learning_rates'] for size in spec['batch_sizes']
                    for strength in spec.get('l2_values', [0])
                    for shift in spec.get('translation_shifts', [0])
                    for architecture in joint_architectures(spec)]
        if stage == 'confirmation':
            pool = []
            previous_stages = ('controls', 'joint') if 'joint' in p['stages'] else STAGES[:-1]
            for previous in previous_stages:
                pool.extend(self.load_stage(previous)['runs'])
            candidates = {}
            for row in rank(pool):
                candidates.setdefault(row['config_id'], row)
            finalists = list(candidates.values())[:p['stages'][stage]['finalists']]
            for row in self.load_stage('controls')['runs']:
                if row['config_id'] not in {r['config_id'] for r in finalists}:
                    finalists.append(row)
            if {'l2_values', 'translation_shifts'} & set(p['stages'].get('joint', {})):
                for row in self.load_stage('joint')['runs']:
                    if (not row['config'].get('l2', 0)
                            and ('l2_values' in p['stages']['joint']
                                 or not row['config'].get('augmentation'))
                            and row['config_id'] not in {r['config_id'] for r in finalists}):
                        finalists.append(row)
            return [(d.config(row['config'], seed), row['label'])
                    for row in finalists for seed in p['confirmation_seeds']]
        previous = STAGES[STAGES.index(stage) - 1]
        base = self.load_stage(previous)['winner']['config']
        spec = p['stages'][stage]
        if stage == 'rates':
            rates = sorted(set(spec['learning_rates'] + [base['optimizer']['learning_rate']]))
            return [(d.config(base, optimizer={**base['optimizer'], 'learning_rate': rate}),
                     f'rate-{rate:g}') for rate in rates]
        if stage == 'architectures':
            variants = copy.deepcopy(spec['architectures'])
            if base['architecture'] not in variants:
                variants.append(base['architecture'])
            return [(d.config(base, architecture=value), 'architecture-' + '-'.join(map(str, value)))
                    for value in variants]
        sizes = list(spec['batch_sizes'])
        if base['strategy'] == 'mini_batch':
            sizes.append(base['batch_size'])
        jobs = [(d.config(base, strategy='mini_batch', batch_size=size), f'batch-{size}')
                for size in sorted(set(sizes))]
        if base['strategy'] != 'mini_batch':
            jobs.append((d.config(base), 'incumbent-training-strategy'))
        return jobs

    def run_stage(self, stage, dry_run=False):
        if stage not in (*STAGES, 'joint'):
            raise ValueError(f'Unknown stage: {stage}')
        if self.stage_path(stage).exists():
            value = self.load_stage(stage)
            if stage == 'confirmation' and not (self.output / 'selection.json').exists():
                self.save_selection(value)
            return value
        jobs = self.planned_jobs(stage)
        if dry_run:
            return {'stage': stage, 'jobs': [
                {'label': label, 'config_id': config_identity(c, path_root=self.development.root),
                 'seed': c['model_seed'], 'config': recorded_config(c, self.development.root)}
                for c, label in jobs]}
        rows = self.run_labeled(jobs)
        if stage == 'confirmation':
            # Duplicate labels for an identical configuration never count as more seeds.
            unique = {(r['config_id'], r['seed']): r for r in rows}
            summary = group_summary(list(unique.values()), self.protocol['confirmation_seeds'])
            completed = [r for r in summary if r['status'] == 'completed']
            if not completed:
                raise ValueError('No configuration completed every confirmation seed.')
            value = {'stage': stage, 'identity': self.identity, 'runs': list(unique.values()),
                     'ranking': summary, 'winner': completed[0]}
            self.save_selection(value)
        else:
            ranked = rank(rows)
            if not ranked:
                raise ValueError('No completed candidate; inspect saved failed run reports.')
            value = {'stage': stage, 'identity': self.identity, 'runs': rows,
                     'ranking': ranked, 'winner': ranked[0]}
        write_json(self.stage_path(stage), value)
        return value

    def save_selection(self, stage):
        winner = stage['winner']
        seed = self.protocol['primary_seed']
        row = next(r for r in stage['runs'] if r['config_id'] == winner['config_id'] and r['seed'] == seed)
        report = read_json(self.output / 'runs' / row['run_id'] / 'results.json')
        write_json(self.output / 'selection.json', {
            'schema_version': 1, 'identity': self.identity, 'scope': 'development validation only',
            'selection_rule': self.protocol['ranking'], 'config': winner['config'],
            'config_id': winner['config_id'], 'seeds': self.protocol['confirmation_seeds'],
            'validation_model': f'runs/{row["run_id"]}/{report["artifacts"]["best_model"]}',
            'model_sha256': report['model_sha256'], 'chosen_epoch': row['chosen_epoch'],
            'dataset': report['dataset'], 'ranking': stage['ranking'],
            'validation_target_met': winner['accuracy_mean'] >= 0.98,
            'limitations': ['The target flag refers to mean validation accuracy, not final generalization.',
                            'Repeated selection on a common validation set can be optimistic.']})

    def run_single(self, path, seed=None, dry_run=False):
        """Full or partial configuration overrides the reference and shared budget."""
        overrides = read_json(path)
        if not isinstance(overrides, dict):
            raise ValueError('Experiment must be a JSON configuration object.')
        if set(overrides) & {'dataset', 'cache', 'data', 'model_seed'}:
            raise ValueError('Custom jobs use the fixed ej3 partition; choose the model seed with --seed.')
        base = copy.deepcopy(self.development.reference)
        for key, value in overrides.items():
            if key in ('optimizer', 'stopping', 'preprocessing', 'activation_parameters', 'weight_logging'):
                if not isinstance(value, dict):
                    raise ValueError(f'{key} must be an object.')
                base[key].update(value)
            else:
                base[key] = value
        # Explicit data selection belongs to the factor study, not custom search jobs.
        from tps_sia.tp3.shared.experiments import validate_config
        base.update(dataset=str(self.development.dataset), cache=str(self.development.cache),
                    data=copy.deepcopy(self.development.data))
        base['model_seed'] = self.protocol['primary_seed'] if seed is None else seed
        config = validate_config(base)
        if dry_run:
            return {'config': recorded_config(config, self.development.root)}
        label = 'custom-' + fingerprint(overrides)[:12]
        rows = self.run_labeled([(config, label)])
        write_json(self.output / f'{label}-seed-{config["model_seed"]}.json',
                   {'identity': self.identity, 'runs': rows})
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--stage', choices=(*STAGES, 'joint'))
    mode.add_argument('--experiment', type=Path, help='JSON overrides for one independent run')
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--reference-selection', type=Path)
    parser.add_argument('--seed', type=int, help='Only for --experiment')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--pause-after', type=int, help='Pause after this many epochs in this invocation')
    parser.add_argument('--verbose', type=int, default=10)
    parser.add_argument('--dry-run', action='store_true', help='Prepare jobs without training')
    args = parser.parse_args()
    if args.seed is not None and args.stage:
        parser.error('--seed is only valid with --experiment; stage seeds belong to the protocol')
    search = Search(args.config, args.output_dir, args.workers, args.reference_selection,
                    args.pause_after, args.verbose)
    value = (search.run_stage(args.stage, args.dry_run) if args.stage
             else search.run_single(args.experiment, args.seed, args.dry_run))
    if args.dry_run:
        import json
        print(json.dumps(value, indent=2, ensure_ascii=False))
    else:
        print(f'Saved development results in {args.output_dir}')


if __name__ == '__main__':
    main()
