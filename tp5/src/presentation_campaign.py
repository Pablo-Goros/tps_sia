"""Execute the bounded decision tree in experiment_plan.md, without publishing."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import matplotlib
import numpy as np

from .autoencoder import Autoencoder
from .compare import run_comparison
from .dataset import load_dataset
from .metrics import reconstruction_metrics
from .train import write_json


APPROVAL = ('User authorized experiment_plan.md on 2026-11-10, then explicitly '
            'authorized reconstructing three missing references: at most 43 new '
            'runs and 321000 epochs. No retries or additional sweeps.')
BASE = dict(name='reference', architecture=[35,40,20,10,2,10,20,40,35],
            optimizer={'name':'rmsprop', 'learning_rate':.002}, batch_size=32,
            max_epochs=3000, evaluate_every=100, stop_on_tolerance=True,
            loss='half_squared_error', latent_activation='tanh', lr_scheduler=None)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def variant(base, name, **changes):
    return {**copy.deepcopy(base), 'name':name, **changes}


def rank(group, historical=False):
    return (() if historical else (-group['success_fraction'],)) + tuple(
        group[key+'_mean'] for key in ('max_pixel_errors','mean_pixel_errors','mse_per_pixel'))


class Campaign:
    def __init__(self, output):
        self.root = Path(output).resolve()
        if self.root.exists() and any(self.root.iterdir()):
            raise ValueError('Campaign output must be empty; no implicit retries.')
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        self.summaries = []
        self.new_runs = self.epochs = 0
        self.dataset = load_dataset()
        self.started = time.perf_counter()

    def verify(self, row):
        source = Path(row['source_run'])
        model = Autoencoder.load(source/'model.npz')
        actual = reconstruction_metrics(self.dataset.vectors, model.reconstruct(self.dataset.vectors))
        saved = json.loads((source/'metrics.json').read_text(encoding='utf-8'))
        for key, value in actual.items():
            if isinstance(value, float):
                if not np.isclose(value, saved[key], rtol=1e-12, atol=1e-14):
                    raise ValueError(f'Reloaded metric differs: {source} {key}')
            elif value != saved[key]:
                raise ValueError(f'Reloaded metric differs: {source} {key}')
        with np.load(source/'predictions.npz') as archive:
            np.testing.assert_array_equal(model.reconstruct(self.dataset.vectors), archive['predictions'])
            np.testing.assert_array_equal(model.encode(self.dataset.vectors), archive['codes'])
        np.testing.assert_array_equal(model.decode(model.encode(self.dataset.vectors)), model.reconstruct(self.dataset.vectors))
        manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
        if digest(source/'model.npz') != manifest['model']['sha256']:
            raise ValueError('Model hash mismatch')
        if manifest['dataset']['sha256'] != self.dataset.sha256:
            raise ValueError('Dataset hash mismatch')
        config = manifest['config']
        expected = config['optimizer']['learning_rate'] * np.ones(model.history['epocas_corridas'])
        if config.get('lr_scheduler'):
            horizon = config['lr_scheduler']['total_epochs']
            expected *= (1+np.cos(np.pi*np.arange(len(expected))/horizon))/2
        np.testing.assert_allclose(model.history['learning_rate'], expected, rtol=1e-13, atol=1e-18)
        return {'run_id': row['run_id'], 'sha256': digest(source/'model.npz'),
                'metrics_reproduced': True, 'predictions_codes_and_rates_verified': True}

    def stage(self, name, variants, seeds=(42,43,44)):
        protocol = dict(stage=name, approval=APPROVAL, seeds=list(seeds),
                        selection='success_max_mean_mse', variants=variants, reuse_runs={})
        for item in variants:
            for seed in seeds:
                run_id = f"{item['name']}-seed-{seed}"
                if run_id in self.cache:
                    protocol['reuse_runs'][run_id] = self.cache[run_id]
        added = [(v,s) for v in variants for s in seeds
                 if f"{v['name']}-seed-{s}" not in protocol['reuse_runs']]
        if self.new_runs+len(added) > 43 or self.epochs+sum(v['max_epochs'] for v,s in added) > 321000:
            raise ValueError('Campaign budget would be exceeded')
        path = self.root/f'{name}.json'
        write_json(path, protocol)
        print(f'STAGE {name}: {len(added)} new runs', flush=True)
        summary = run_comparison(path, self.root/name)
        checks = []
        for row in summary['runs']:
            if row['status'] != 'completed':
                raise RuntimeError(f'Stop campaign after incomplete run: {row}')
            checks.append(self.verify(row))
            if not row['reused']:
                self.new_runs += 1
                self.epochs += row['epochs_completed']
            self.cache[row['run_id']] = row['source_run']
        write_json(self.root/name/'reload-verification.json', checks)
        historical = min(summary['aggregates'], key=lambda g: rank(g, historical=True))
        decision = {'stage': name, 'winner': summary['selected_variant'],
                    'historical_ranking_winner': historical['variant'],
                    'aggregates': summary['aggregates']}
        self.summaries.append(decision)
        write_json(self.root/'decisions.json', self.summaries)
        write_json(self.root/'budget.json', {'new_runs':self.new_runs, 'epochs':self.epochs,
                   'max_runs':43, 'max_epochs':321000, 'elapsed_seconds':time.perf_counter()-self.started})
        winner = next(v for v in variants if v['name'] == summary['selected_variant'])
        print(f'DECISION {name}: {winner["name"]}; aggregates='+json.dumps(summary['aggregates']), flush=True)
        return winner, summary

    def run(self):
        prior = {str(p.resolve()):digest(p) for p in Path('results').rglob('*')
                 if p.is_file() and not p.resolve().is_relative_to(self.root)}
        write_json(self.root/'prior-hashes.json', prior)
        write_json(self.root/'environment.json', dict(python=platform.python_version(),
            numpy=np.__version__, matplotlib=matplotlib.__version__,
            OPENBLAS_NUM_THREADS=os.getenv('OPENBLAS_NUM_THREADS'), OMP_NUM_THREADS=os.getenv('OMP_NUM_THREADS'),
            reference_note='Historic broad/final artifacts and Python 3.13 unavailable. Three controls reconstructed with explicit authorization.',
            sigmoid_note='Local logistic output uses sigmoid(2*beta*h), beta=1. BCE derivative includes 2*beta.',
            cosine_note='Update t uses lr0*(1+cos(pi*t/T))/2, t=0..T-1. Boundary at T is zero; no extra zero-rate update.'))
        # Check the available old pilot before any campaign training.
        pilot = Path('results/pilot')
        old_model = Autoencoder.load(pilot/'model.npz')
        saved = json.loads((pilot/'metrics.json').read_text(encoding='utf-8'))
        actual = reconstruction_metrics(self.dataset.vectors, old_model.reconstruct(self.dataset.vectors))
        if actual['pixel_errors'] != saved['pixel_errors'] or not np.isclose(actual['mse_per_pixel'], saved['mse_per_pixel'], rtol=1e-12):
            raise ValueError('Available old pilot does not reproduce')
        initial_checks = []
        for seed in (42,43,44):
            initial = Autoencoder(architecture=BASE['architecture'], seed=seed)
            for loss in ('half_squared_error','bce'):
                for latent in ('tanh','linear'):
                    other = Autoencoder(architecture=BASE['architecture'], seed=seed, loss=loss, latent_activation=latent)
                    for a,b in zip(initial._mlp.parametros,other._mlp.parametros):
                        np.testing.assert_array_equal(a,b)
            initial_checks.append({'seed':seed, 'all_initial_weights_equal':True})
        write_json(self.root/'preflight.json', {'old_pilot_reproduced':True, 'initial_weights':initial_checks,
                   'dataset_sha256':self.dataset.sha256, 'approval':APPROVAL})
        variants = [BASE, variant(BASE,'bce-tanh',loss='bce'),
                    variant(BASE,'squared-linear',latent_activation='linear'),
                    variant(BASE,'bce-linear',loss='bce',latent_activation='linear')]
        best, _ = self.stage('01-loss-latent',variants)
        adam = variant(best,best['name']+'-adam',optimizer={'name':'adam','learning_rate':.002})
        best, _ = self.stage('02-optimizer',[best,adam])
        settings = [best]
        for lr, cosine in ((.002,True),(.003,False),(.003,True)):
            opt = {**best['optimizer'],'learning_rate':lr}
            scheduler = {'name':'cosine','total_epochs':3000,'min_lr':0.} if cosine else None
            settings.append(variant(best,best['name']+f'-lr-{int(lr*10000)}-'+('cosine' if cosine else 'constant'), optimizer=opt,lr_scheduler=scheduler))
        best, _ = self.stage('03-rate-schedule',settings)
        narrow = variant(best,best['name']+'-narrow',architecture=[35,25,15,8,2,8,15,25,35])
        best, _ = self.stage('04-architecture',[best,narrow])
        long = [best]
        for cosine in (False,True):
            long.append(variant(best,best['name']+'-long-'+('cosine' if cosine else 'constant'),
                         max_epochs=15000,lr_scheduler={'name':'cosine','total_epochs':15000,'min_lr':0.} if cosine else None))
        best, _ = self.stage('05-training-budget',long)
        # Freeze before evaluating any new seed. No later tuning.
        write_json(self.root/'frozen-finalist.json',best)
        control = variant(BASE,'confirmation-reference',max_epochs=best['max_epochs'])
        finalist = variant(best,'confirmation-finalist')
        winner, confirmation = self.stage('06-confirmation',[control,finalist],seeds=(45,46,47,48,49))
        promoted = winner['name'] == 'confirmation-finalist' and any(
            r['all_within_tolerance'] for r in confirmation['runs'] if r['variant']=='confirmation-finalist')
        write_json(self.root/'outcome.json', {'confirmed':promoted, 'finalist':best,
                   'confirmation_selected':confirmation['selected'],
                   'new_runs':self.new_runs,'epochs':self.epochs,
                   'elapsed_seconds':time.perf_counter()-self.started})
        changed = [p for p,h in prior.items() if not Path(p).is_file() or digest(Path(p)) != h]
        write_json(self.root/'preservation.json', {'verified_files':len(prior),'changed':changed})
        if changed:
            raise ValueError('Prior artifacts changed')
        print('CAMPAIGN COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    Campaign(args.output).run()


if __name__ == '__main__':
    main()
