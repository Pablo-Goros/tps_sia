"""Continue the approved campaign, then compare optimizers; never expand it silently."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys


class _Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retry-interrupted', action='store_true',
                        help='Explicitly authorize a full retry of each incomplete attempt; increases total budget')
    args = parser.parse_args()
    # Set before importing NumPy through comparison, unless caller explicitly chose another value.
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    from .compare import run_comparison
    from .dataset import DEFAULT_DATASET
    from .train import write_json

    root = Path(__file__).resolve().parents[1]
    results = root/'results'
    results.mkdir(exist_ok=True)
    with (results/'overnight.log').open('a', encoding='utf-8') as log:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log)), contextlib.redirect_stderr(_Tee(sys.stderr, log)):
            print('Overnight continuation: completed runs are validated and skipped.', flush=True)
            if args.retry_interrupted:
                print('Retry authorized by command flag: interrupted attempts are archived and charged their full original epoch cap. Current campaign cap: up to 27000 epochs including one 3000-epoch interruption, versus original 24000.', flush=True)
            try:
                architecture_protocol = root/'configs/comparison-architectures.json'
                first = run_comparison(architecture_protocol, results/'comparison-architectures',
                                       DEFAULT_DATASET, resume=True, retry_interrupted=args.retry_interrupted)
                if any(group['completed'] != group['runs'] for group in first['aggregates']):
                    raise ValueError('Architecture stage is incomplete. Review failures before optimizer selection; no partial architecture comparison used.')
                protocol = json.loads(architecture_protocol.read_text(encoding='utf-8'))
                winner = next(v for v in protocol['variants'] if v['name'] == first['selected_variant'])
                adam = {**winner, 'name': 'adam'}
                momentum = {**winner, 'name': 'momentum', 'optimizer': {'name': 'momentum', 'learning_rate': .05, 'momentum': .9}}
                second_protocol = {**protocol, 'stage': 'optimizers', 'variants': [adam, momentum],
                                   'reuse_runs': {f'adam-seed-{seed}': str(results/'comparison-architectures'/f'{winner["name"]}-seed-{seed}') for seed in protocol['seeds']}}
                optimizer_config = root/'configs/comparison-optimizers.json'
                write_json(optimizer_config, second_protocol)
                second = run_comparison(optimizer_config, results/'comparison-optimizers',
                                        DEFAULT_DATASET, resume=True, retry_interrupted=args.retry_interrupted)
                if any(group['completed'] != group['runs'] for group in second['aggregates']):
                    raise ValueError('Optimizer stage incomplete; review retained evidence before selection.')
                write_json(results/'overnight-status.json', {'status': 'completed', 'selected': second['selected'],
                           'next_action': 'Review full metrics and generate exploration; visual letter validation remains pending.'})
                print(json.dumps(second['selected'], indent=2), flush=True)
                print('Campaign complete. Review evidence; no extra training or visual letter claim is automatic.', flush=True)
            except BaseException as error:
                write_json(results/'overnight-status.json', {'status': 'stopped', 'reason': f'{type(error).__name__}: {error}'})
                print(f'Stopped: {type(error).__name__}: {error}', flush=True)
                raise


if __name__ == '__main__':
    main()
