"""Ejecuta una configuración explícita de desarrollo; los barridos son opt-in."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from tps_sia.tp3.shared.experiments import run, sha256_file, validate_config
from .datos_digitos import EJERCICIO

CONFIG = EJERCICIO / 'configs' / 'search.json'


def load_search(path: str | Path) -> dict:
    path = Path(path).resolve()
    search = json.loads(path.read_text(encoding='utf-8'))
    if set(search) != {'schema_version', 'selection_rule', 'seeds', 'base', 'candidates'} or search['schema_version'] != 1:
        raise ValueError('Invalid search schema.')
    if not isinstance(search['selection_rule'], str) or not search['selection_rule'].strip():
        raise ValueError('A predefined selection rule is required.')
    if (not isinstance(search['seeds'], list) or not search['seeds']
            or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in search['seeds'])
            or len(set(search['seeds'])) != len(search['seeds'])):
        raise ValueError('Expected distinct model seeds.')
    if not isinstance(search['candidates'], list):
        raise ValueError('Candidates must be a list.')
    names = set()
    for candidate in search['candidates']:
        if (not isinstance(candidate, dict) or set(candidate) != {'name', 'overrides'}
                or not isinstance(candidate['name'], str) or not candidate['name']
                or candidate['name'] in names or not isinstance(candidate['overrides'], dict)):
            raise ValueError('Expected distinct named candidates with overrides.')
        names.add(candidate['name'])
        for seed in search['seeds']:
            config = candidate_config(search, candidate['name'], seed, path.parent)
            validate_config(config)
    if not names:
        raise ValueError('No candidates configured.')
    return search


def candidate_config(search: dict, name: str, seed: int, base_dir: Path) -> dict:
    if seed not in search['seeds']:
        raise ValueError('Seed must be predefined in the search configuration.')
    matches = [c for c in search['candidates'] if c['name'] == name]
    if len(matches) != 1:
        raise ValueError(f'Unknown candidate: {name}')
    config = copy.deepcopy(search['base'])
    for key, value in matches[0]['overrides'].items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    config['model_seed'] = seed
    for key in ('dataset', 'cache'):
        config[key] = str((base_dir / config[key]).resolve())
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--candidate', default='baseline')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--initial-model', type=Path)
    parser.add_argument('--pause-after', type=int)
    parser.add_argument('--all', action='store_true', help='Ejecutar explícitamente todos los candidatos/semillas')
    args = parser.parse_args()
    search = load_search(args.config)
    if args.all and (args.resume or args.initial_model or args.pause_after):
        parser.error('--all no se combina con reanudación, pesos iniciales ni pausa')
    jobs = [(c['name'], seed) for c in search['candidates'] for seed in search['seeds']] if args.all else [(args.candidate, args.seed)]
    for name, seed in jobs:
        config = candidate_config(search, name, seed, args.config.resolve().parent)
        report = run(config, args.output_dir, resume=args.resume, initial_model=args.initial_model,
                     pause_after=args.pause_after, verbose=1,
                     metadata={'candidate': name, 'search_path': str(args.config.resolve()),
                               'search_sha256': sha256_file(args.config), 'seeds': search['seeds'],
                               'selection_rule': search['selection_rule'], 'search': search})
        print(f"{name}: {report['status']} · {report['run_directory']}", flush=True)


if __name__ == '__main__':
    main()
