"""One-factor-at-a-time search for ej2 (protocol v2): one command per stage.

Each invocation runs only the requested stage. It reads the decisions of the
previous stages from the output directory, runs (or reuses) the needed runs and
applies the rules pre-registered in configs/search_v2.json. Runs are stored as
runs/<config_id>-seed-<seed>/ with dataset and cache paths relative to tp3/, so
the same configuration has the same id on every machine. digits_test.csv is
rejected by the shared runner before any data is loaded.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from tps_sia.tp3.shared.experiments import (config_identity, recorded_config, run, sha256_file,
                                            validate_config, write_json)

TP3 = Path(__file__).resolve().parents[2]
CONFIG = TP3 / 'ej2' / 'configs' / 'search_v2.json'
OUTPUT = TP3 / 'ej2' / 'results' / 'v2'
REQUIRED = {'schema_version', 'protocol', 'base', 'primary_seed', 'ranking', 'near_tie',
            'improvement_rule', 'stages'}


# Reexports preserve the existing ej2 command and consumers.
from tps_sia.tp3.shared.staged_search import (
    _job, _same, best_neighbor, border_extension, choose_activation, compare_groups,
    execute_jobs, group_summary, load_protocol, materialize, near_tie_pair,
    pooled_sigma, protocol_sha256, rank, rank_groups, rank_key, resolve_near_tie,
    sequence_neighbor, summarize,
)


class StagedSearch:
    def __init__(self, config_path=CONFIG, output=OUTPUT, workers=1, root=TP3):
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError('workers must be a positive integer.')
        self.config_path = Path(config_path).resolve()
        self.protocol = load_protocol(self.config_path)
        self.sha = protocol_sha256(self.config_path)
        self.root, self.output, self.workers = Path(root).resolve(), Path(output).resolve(), workers
        self.metadata = {'protocol': self.protocol['protocol'], 'search_sha256': self.sha}
        self.primary = self.protocol['primary_seed']
        base = validate_config(materialize(self.protocol['base'], self.root, self.primary))
        if Path(base['dataset']).name != 'digits.csv':
            raise ValueError('The ej2 search uses only digits.csv.')

    # -- storage
    def stage_path(self, stage):
        return self.output / f'stage_{stage}.json'

    def load_stage(self, stage):
        path = self.stage_path(stage)
        if not path.exists():
            raise ValueError(f'Stage {stage} has not been run; run it first.')
        data = json.loads(path.read_text(encoding='utf-8'))
        if data['search_sha256'] != self.sha:
            raise ValueError(f'Stage {stage} used a different protocol (SHA-256 mismatch).')
        return data

    def save_stage(self, stage, data):
        self.output.mkdir(parents=True, exist_ok=True)
        value = {'schema_version': 1, 'stage': stage, 'factor': self.protocol['stages'][str(stage)]['factor'],
                 'protocol': self.protocol['protocol'], 'search_sha256': self.sha, **data}
        write_json(self.stage_path(stage), value)
        return value

    # -- runs
    def config(self, base, seed=None, **changes):
        config = copy.deepcopy(base)
        config.update(copy.deepcopy(changes))
        return materialize(config, self.root, self.primary if seed is None else seed)

    def run_labeled(self, jobs):
        """Run (config, label) pairs; identical configuration/seed pairs run once."""
        return execute_jobs(jobs, self.output, self.metadata, self.root, self.workers)

    def decide(self, stage, rows, incumbent_id):
        """Seed-42 ranking plus the pre-registered near-tie rule for this stage."""
        ranked = rank(rows)
        if not ranked:
            raise ValueError(f'No completed runs in stage {stage}.')
        near = self.protocol['near_tie']
        pair = near_tie_pair(ranked, near['threshold_accuracy']) if stage in near['stages'] else None
        result = {'incumbent_config_id': incumbent_id, 'near_tie': None}
        order = [r['config_id'] for r in ranked]
        if pair is not None:
            extra = self.run_labeled([(self.config(r['config'], seed), r['label'])
                                      for r in pair for seed in near['extra_seeds']])
            summary = group_summary(pair + extra, [self.primary, *near['extra_seeds']])
            winner_id, evidence = resolve_near_tie(summary, incumbent_id,
                                                   self.protocol['improvement_rule']['sigma_multiplier'])
            loser_id = next(r['config_id'] for r in pair if r['config_id'] != winner_id)
            order = [winner_id, loser_id] + order[2:]
            result['near_tie'] = {'triggered_by': [r['run_id'] for r in pair],
                                  'difference': pair[0]['accuracy'] - pair[1]['accuracy'],
                                  'extra_runs': extra, 'groups': summary, **evidence}
        winner = next(r for r in ranked if r['config_id'] == order[0])
        result.update(order=order, winner=winner)
        return result

    # -- stages
    def stage_0(self):
        spec = self.protocol['stages']['0']
        jobs = [(self.config(self.protocol['base'], activation=activation,
                             optimizer={**spec['optimizer'], 'learning_rate': rate}),
                 f'{activation}-sgd-{rate:g}')
                for activation in spec['activations'] for rate in spec['learning_rates']]
        rows = self.run_labeled(jobs)
        winner, decision = choose_activation(rows, spec['tie_threshold_accuracy'], spec['tie_preference'])
        return self.save_stage(0, {'rule': spec['rule'], 'runs': rows, 'decision': decision,
                                   'winner': winner, 'control': winner})

    def stage_1(self):
        base = self.load_stage(0)['winner']['config']
        spec = self.protocol['stages']['1']
        border = spec['border_rule']
        mantissas = tuple(border['sequence_mantissas'])
        grids = {name: sorted(o['learning_rates']) for name, o in spec['optimizers'].items()}
        rows = {name: [] for name in grids}
        pending = {name: list(grid) for name, grid in grids.items()}
        sides = {(name, side): 0 for name in grids for side in ('up', 'down')}
        extensions, unresolved, round_number = [], {}, 0
        while pending:
            round_number += 1
            jobs = []
            for name, rates in pending.items():
                settings = spec['optimizers'][name]['settings']
                jobs += [(self.config(base, optimizer={'name': name, 'learning_rate': rate, **settings}),
                          f'{name}-{rate:g}') for rate in rates]
            for row in self.run_labeled(jobs):
                rows[row['optimizer']].append(row)
            pending = {}
            for name in grids:
                extension = border_extension(rows[name], grids[name], mantissas)
                if extension is None:
                    if not rank(rows[name]):
                        unresolved[name] = 'no completed runs'
                    continue
                side, rate = extension
                best = rank(rows[name])[0]
                if sides[(name, side)] >= border['max_extensions_per_side']:
                    unresolved[name] = f'best rate {best["learning_rate"]:g} still on the {side} border'
                    continue
                sides[(name, side)] += 1
                grids[name] = sorted(grids[name] + [rate])
                pending[name] = [rate]
                extensions.append({'round': round_number, 'optimizer': name, 'direction': side,
                                   'best_learning_rate': best['learning_rate'],
                                   'best_accuracy': best['accuracy'], 'added_learning_rate': rate})
        best = {name: rank(r)[0] for name, r in rows.items() if rank(r)}
        return self.save_stage(1, {'rule': border['rule'], 'grids': grids, 'runs': rows,
                                   'extensions': extensions, 'unresolved_borders': unresolved,
                                   'best': best})

    def stage_2(self):
        stage_1 = self.load_stage(1)
        spec = self.protocol['stages']['2']
        candidates = list(stage_1['best'].values())  # Reused runs; nothing is retrained.
        incumbent = next((r['config_id'] for r in candidates if r['optimizer'] == spec['incumbent']), None)
        decision = self.decide(2, candidates, incumbent)
        by_id = {r['config_id']: r for r in candidates}
        top = [by_id[config_id] for config_id in decision['order'][:spec['keep_top']]]
        return self.save_stage(2, {'rule': spec['rule'], 'runs': candidates, 'decision': decision,
                                   'winner': decision['winner'], 'top': top})

    def _factor_stage(self, stage, base, variants, incumbent_value, key, extra=None):
        rows = self.run_labeled([(self.config(base, **{key: value}), label) for value, label in variants])
        incumbent = next((r['config_id'] for r in rows if r[key] == incumbent_value), None)
        decision = self.decide(stage, rows, incumbent)
        return self.save_stage(stage, {**(extra or {}), 'runs': rows, 'decision': decision,
                                       'winner': decision['winner']})

    def stage_3(self):
        base = self.load_stage(2)['winner']['config']
        spec = self.protocol['stages']['3']
        variants = [([784, width, 10], f'width-{width}') for width in spec['widths']]
        return self._factor_stage(3, base, variants, [784, spec['incumbent'], 10], 'architecture')

    def stage_4(self):
        winner = self.load_stage(3)['winner']
        spec = self.protocol['stages']['4']
        width = winner['architecture'][1]
        variants = []
        for divisors in spec['depth_divisors']:
            if any(width % d for d in divisors):
                raise ValueError(f'Width {width} is not divisible by {divisors}.')
            hidden = [width // d for d in divisors]
            variants.append(([784, *hidden, 10], 'hidden-' + '-'.join(map(str, hidden))))
        incumbent = [784, *[width // d for d in spec['incumbent']], 10]
        return self._factor_stage(4, winner['config'], variants, incumbent, 'architecture')

    def stage_5(self):
        base = self.load_stage(4)['winner']['config']
        spec = self.protocol['stages']['5']
        variants = [(size, f'batch-{size}') for size in spec['batch_sizes']]
        return self._factor_stage(5, base, variants, spec['incumbent'], 'batch_size',
                                  {'learning_rate_policy': spec['learning_rate_policy']})

    def stage_6(self):
        stages = {i: self.load_stage(i) for i in range(1, 6)}
        spec = self.protocol['stages']['6']
        base = stages[5]['winner']['config']
        first = stages[4]['winner']['architecture']
        pool = rank(stages[3]['runs'] + stages[4]['runs'])
        architectures = [first] + [a for a in dict.fromkeys(tuple(r['architecture']) for r in pool)
                                   if list(a) != first][:spec['architectures'] - 1]
        architectures = [list(a) for a in architectures]
        rates = {}
        for row in stages[2]['top'][:spec['optimizers']]:
            name = row['optimizer']
            neighbor = best_neighbor(stages[1]['runs'][name], stages[1]['grids'][name], row['learning_rate'])
            rates[name] = {'optimizer': row['config']['optimizer'], 'learning_rates': [row['learning_rate'], neighbor]}
        jobs = []
        for name, item in rates.items():
            for rate in item['learning_rates']:
                for architecture in architectures:
                    label = f'{name}-{rate:g}-hidden-' + '-'.join(map(str, architecture[1:-1]))
                    for seed in spec['seeds']:
                        jobs.append((self.config(base, seed, architecture=architecture,
                                                 optimizer={**item['optimizer'], 'learning_rate': rate}), label))
        rows = self.run_labeled(jobs)
        summary = group_summary(rows, spec['seeds'])
        completed = [s for s in summary if s['status'] == 'completed']
        evidence = None
        if len(completed) >= 2:
            evidence = compare_groups(completed[0], completed[1],
                                      self.protocol['improvement_rule']['sigma_multiplier'])
        return self.save_stage(6, {'rule': spec['rule'], 'architectures': architectures,
                                   'learning_rates': rates, 'runs': rows, 'ranking': summary,
                                   'top_vs_runner_up': evidence})

    def stage_7(self):
        stage_0, stage_6 = self.load_stage(0), self.load_stage(6)
        spec = self.protocol['stages']['7']
        finalists = [s for s in stage_6['ranking'] if s['status'] == 'completed'][:spec['finalists']]
        control = stage_0['control']
        candidates = [(s['config'], s['label']) for s in finalists]
        if control['config_id'] not in [s['config_id'] for s in finalists]:
            candidates.append((control['config'], 'control-' + control['label']))
        rows = self.run_labeled([(self.config(config, seed), label)
                                 for config, label in candidates for seed in spec['seeds']])
        summary = group_summary(rows, spec['seeds'])
        completed = [s for s in summary if s['status'] == 'completed']
        if not completed:
            raise ValueError('No finalist completed every confirmation seed.')
        winner = completed[0]
        multiplier = self.protocol['improvement_rule']['sigma_multiplier']
        evidence = {}
        if len(completed) >= 2:
            evidence['winner_vs_runner_up'] = compare_groups(winner, completed[1], multiplier)
        control_group = next((s for s in completed if s['config_id'] == control['config_id']), None)
        if control_group is not None and control_group is not winner:
            evidence['winner_vs_control'] = compare_groups(winner, control_group, multiplier)
        stage = self.save_stage(7, {'ranking_rule': spec['ranking'], 'control_config_id': control['config_id'],
                                    'runs': rows, 'ranking': summary, 'evidence': evidence})
        candidate = next(r for r in rows if r['config_id'] == winner['config_id'] and r['seed'] == self.primary)
        directory = self.output / 'runs' / candidate['run_id']
        report = json.loads((directory / 'results.json').read_text(encoding='utf-8'))
        selection = {
            'schema_version': 2, 'protocol': self.protocol['protocol'], 'search_sha256': self.sha,
            'paths_relative_to': 'tp3', 'selection_rule': spec['ranking'],
            'config': winner['config'], 'config_id': winner['config_id'], 'seeds': spec['seeds'],
            'delivery_seed': self.primary, 'dataset': report['dataset'],
            'candidate_model': f'runs/{candidate["run_id"]}/{report["artifacts"]["best_model"]}',
            'model_sha256': report['model_sha256'], 'candidate_epoch': report['chosen_epoch'],
            'retraining_epochs': int(np.median(winner['chosen_epochs'])),
            'retraining_epoch_rule': 'median best validation-loss epoch across the confirmation seeds',
            'delivery_policy': self.protocol['delivery'],
            'evidence': {'stages': [f'stage_{i}.json' for i in range(8)], 'ranking': summary,
                         'comparisons': evidence,
                         'sigma_definition': self.protocol['improvement_rule']['sigma']},
            'limitations': ['Validation has no digit 8; its recall is not evaluable.',
                            'Digit 5 has 54 validation samples; each error moves its recall by ~1.85 points.',
                            'Repeated validation selection can be optimistic; final test remains reserved.']}
        write_json(self.output / 'selection.json', selection)
        return stage

    def run_stage(self, stage):
        if stage not in range(8):
            raise ValueError('Stage must be between 0 and 7.')
        if self.stage_path(stage).exists():
            existing = self.load_stage(stage)
            print(f'Stage {stage} already decided: {self.stage_path(stage)}', flush=True)
            return existing
        return getattr(self, f'stage_{stage}')()


def _winner_text(stage):
    if 'winner' in stage:
        w = stage['winner']
        return f'{w["label"]} ({w["config_id"]}, accuracy {w["accuracy"]:.4f})'
    if 'best' in stage:  # Stage 1: best learning rate per optimizer.
        return ', '.join(f'{name} {row["learning_rate"]:g} (accuracy {row["accuracy"]:.4f})'
                         for name, row in stage['best'].items())
    ranking =[s for s in stage['ranking'] if s['status'] == 'completed']
    if ranking:
        return f'{ranking[0]["label"]} ({ranking[0]["config_id"]}, mean accuracy {ranking[0]["accuracy_mean"]:.4f})'
    return 'none'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=int, required=True, choices=range(8))
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--workers', type=int, default=1, help='Corridas simultáneas (procesos)')
    args = parser.parse_args()
    search = StagedSearch(args.config, args.output_dir, args.workers)
    stage = search.run_stage(args.stage)
    print(f'Stage {args.stage} ({stage["factor"]}): {_winner_text(stage)}', flush=True)


if __name__ == '__main__':
    main()
