"""Reload-only exploration of the two-dimensional character codes."""
import argparse
import json
from pathlib import Path
import numpy as np

from .autoencoder import Autoencoder
from .dataset import DEFAULT_DATASET, load_dataset
from .plots import plot_latent, plot_candidates
from .train import model_identity, write_json


def candidate_distances(predictions, originals):
    """Minimum binary Hamming distance to every training pattern."""
    predictions = np.asarray(predictions, dtype=float)
    originals = np.asarray(originals)
    if (predictions.ndim != 2 or predictions.shape[1] != 35
            or originals.shape != (32, 35) or not np.isfinite(predictions).all()
            or not np.isin(originals, [0, 1]).all()):
        raise ValueError('Expected finite (N,35) predictions and 32 binary originals')
    return np.count_nonzero((predictions[:, None, :] >= 0.5) != originals[None, :, :], axis=2).min(axis=1).tolist()


def visualize_latent(model_path, output, dataset_path=DEFAULT_DATASET):
    """Save the 32 labeled codes and their figure using a persisted model."""
    dataset = load_dataset(dataset_path)
    model = Autoencoder.load(model_path)
    coordinates = model.encode(dataset.vectors)
    result = {
        'model': model_identity(model_path),
        'dataset': {'path': dataset.path, 'sha256': dataset.sha256},
        'points': [{'code': index + 96, 'label': label, 'coordinates': point.tolist()}
                   for index, (label, point) in enumerate(zip(dataset.labels, coordinates))],
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'latent.json', result)
    plot_latent(result, output / 'latent.png')
    return result


def explore_candidates(model_path, output, config, dataset_path=DEFAULT_DATASET):
    """Decode configured interpolations and a grid over the occupied code range."""
    output = Path(output)
    if (output / 'candidates.json').exists():
        raise ValueError('candidates.json already exists; use a new output directory to preserve visual reviews')
    if set(config) != {'pairs', 'steps', 'grid_size'}:
        raise ValueError('Exploration config requires pairs, steps and grid_size')
    for key in ('steps', 'grid_size'):
        if type(config[key]) is not int or not 2 <= config[key] <= 50:
            raise ValueError(f'{key} must be an integer between 2 and 50')
    pairs = config['pairs']
    if (not isinstance(pairs, list) or not 1 <= len(pairs) <= 32
            or any(not isinstance(pair, list) or len(pair) != 2
                   or any(type(code) is not int or not 96 <= code <= 127 for code in pair)
                   for pair in pairs)):
        raise ValueError('pairs must contain 1–32 pairs of original codes 96–127')
    dataset = load_dataset(dataset_path)
    model = Autoencoder.load(model_path)
    codes = model.encode(dataset.vectors)
    low, high = codes.min(axis=0), codes.max(axis=0)
    coordinates, kinds = [], []
    for pair_index, (start, end) in enumerate(pairs):
        for fraction in np.linspace(0, 1, config['steps']):
            coordinates.append((1-fraction)*codes[start-96] + fraction*codes[end-96])
            kinds.append({'kind': 'interpolation', 'pair_index': pair_index, 'fraction': float(fraction)})
    split = len(coordinates)
    for y in np.linspace(low[1], high[1], config['grid_size']):
        for x in np.linspace(low[0], high[0], config['grid_size']):
            coordinates.append([x, y])
            kinds.append({'kind': 'grid'})
    coordinates = np.asarray(coordinates)
    if model.configuration['latent_activation'] == 'tanh' and np.any(np.abs(coordinates) > 1 + 1e-12):
        raise ValueError('Exploration requires tanh codes within [-1,1]')
    predictions = model.decode(coordinates)
    distances = candidate_distances(predictions, dataset.vectors)
    candidates = [dict(id=index, coordinates=point.tolist(), continuous=prediction.tolist(),
                       binary=(prediction >= 0.5).astype(int).tolist(), min_hamming=distance,
                       classification='new_candidate' if distance else 'training_copy', **kind)
                  for index, (point, prediction, distance, kind) in enumerate(zip(coordinates, predictions, distances, kinds))]
    result = {'model': model_identity(model_path),
              'dataset': {'path': dataset.path, 'sha256': dataset.sha256},
              'config': config, 'range': {'min': low.tolist(), 'max': high.tolist()},
              'candidates': candidates, 'review': {'status': 'pending', 'selected_id': None}}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'candidates.json', result)
    plot_candidates(candidates[:split], config['steps'], output / 'interpolations.png', result)
    plot_candidates(candidates[split:], config['grid_size'], output / 'grid.png', result)
    return result


def record_review(candidates_path, model_path, selected_id, note, dataset_path=DEFAULT_DATASET):
    """Record the student's explicit visual decision for this exact model."""
    if not isinstance(note, str) or not note.strip():
        raise ValueError('A visual review note is required')
    result = json.loads(Path(candidates_path).read_text(encoding='utf-8'))
    identity = model_identity(model_path)
    if identity['sha256'] != result['model']['sha256']:
        raise ValueError('Review model does not match candidate model')
    dataset = load_dataset(dataset_path)
    if dataset.sha256 != result['dataset']['sha256']:
        raise ValueError('Review dataset does not match candidate dataset')
    review = {'status': 'no_suitable_letter', 'selected_id': None,
              'model_sha256': identity['sha256'], 'note': note.strip()}
    if selected_id is not None:
        if type(selected_id) is not int:
            raise ValueError('Candidate id must be an integer')
        candidate = next((p for p in result['candidates'] if p['id'] == selected_id), None)
        if candidate is None:
            raise ValueError('Unknown candidate id')
        prediction = Autoencoder.load(model_path).decode(np.array([candidate['coordinates']]))
        if candidate_distances(prediction, dataset.vectors)[0] == 0:
            raise ValueError('A training copy cannot be selected as a new letter')
        if not np.allclose(prediction[0], candidate['continuous'], rtol=0, atol=1e-12):
            raise ValueError('Candidate does not regenerate from the saved model')
        review.update(status='letter_selected', selected_id=selected_id,
                      coordinates=candidate['coordinates'], binary=(prediction[0] >= 0.5).astype(int).tolist())
    result['review'] = review
    write_json(candidates_path, result)
    return review


def main():
    parser = argparse.ArgumentParser(description='Visualize saved autoencoder codes without training')
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--output', type=Path, default=Path('results/latent'))
    parser.add_argument('--config', type=Path, help='Generate candidates using this JSON configuration')
    parser.add_argument('--review', type=Path, help='Record an explicit visual decision in candidates.json')
    parser.add_argument('--candidate', type=int, help='Selected candidate id; omit if no letter is suitable')
    parser.add_argument('--note', help='Student visual review note')
    args = parser.parse_args()
    try:
        if args.review:
            if args.config:
                raise ValueError('--review and --config cannot be combined')
            record_review(args.review, args.model, args.candidate, args.note, args.dataset)
        elif args.config:
            explore_candidates(args.model, args.output, json.loads(args.config.read_text(encoding='utf-8')), args.dataset)
        else:
            visualize_latent(args.model, args.output, args.dataset)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f'Exploration artifacts: {args.review if args.review else args.output}')


if __name__ == '__main__':
    main()
