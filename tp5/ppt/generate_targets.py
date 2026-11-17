"""Compare four unseen target glyphs with outputs of the frozen TP5 decoder."""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

TP = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / 'assets'
MODEL = TP / 'results/final/model.npz'
sys.path.insert(0, str(TP))
from src.autoencoder import Autoencoder
from src.dataset import load_dataset

# Reference drawings for this experiment only: never added to training data.
TARGETS = [
    ('enie', 'Letra ñ', [
        '01010', '10100', '00000', '10110', '11001', '10001', '10001']),
    ('three', 'Número 3', [
        '01110', '10001', '00001', '00110', '00001', '10001', '01110']),
    ('d', 'Letra D', [
        '11110', '10001', '10001', '10001', '10001', '10001', '11110']),
    ('exclamation', 'Signo !', [
        '00100', '00100', '00100', '00100', '00100', '00000', '00100']),
]
RESOLUTION = 301
THRESHOLD = 0.5


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def novelty_distances(binary, originals):
    """Compare against every original, keeping temporary arrays bounded."""
    return np.concatenate([
        np.count_nonzero(chunk[:, None, :] != originals[None, :, :], axis=2).min(axis=1)
        for chunk in np.array_split(binary, max(1, (len(binary) + 4095) // 4096))
    ])


def best_candidate(coordinates, predictions, target, originals):
    """Minimize binary differences, then MSE, among outputs absent from Font3."""
    binary = (predictions >= THRESHOLD).astype(np.uint8)
    eligible = np.flatnonzero(novelty_distances(binary, originals) > 0)
    if not len(eligible):
        raise ValueError('No novel output in this search region')
    errors = np.count_nonzero(binary[eligible] != target, axis=1)
    mse = np.mean((predictions[eligible] - target) ** 2, axis=1)
    winner = np.lexsort((mse, errors))[0]
    index = eligible[winner]
    return coordinates[index], predictions[index], (int(errors[winner]), float(mse[winner]))


def plot_comparison(record, name):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 16,
        'text.color': '#26384a', 'savefig.facecolor': 'white',
        # Preserve typography when the SVG is imported into PowerPoint.
        'svg.fonttype': 'path',
    })
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(record['label'], y=0.96, fontsize=25, weight='bold')
    for left, title, values in [
        (0.16, 'Objetivo', record['target']),
        (0.57, 'Generado', record['binary']),
    ]:
        ax = fig.add_axes([left, 0.25, 0.27, 0.56])
        ax.imshow(np.array(values, dtype=float).reshape(7, 5), cmap='Blues',
                  vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(title, fontsize=19, pad=12)
        ax.axis('off')
    fig.text(0.5, 0.52, '→', ha='center', va='center', fontsize=33, color='#368dc4')
    fig.text(0.5, 0.145,
             f"Pixeles incorrectos: {record['target_pixel_errors']} de 35 pixels",
             ha='center', fontsize=17, weight='normal')
    for extension in ('png', 'svg'):
        fig.savefig(OUT / f'{name}.{extension}', dpi=240)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset()
    model_hash = sha256(MODEL)
    model = Autoencoder.load(MODEL)
    original_codes = model.encode(dataset.vectors)
    low, high = original_codes.min(axis=0), original_codes.max(axis=0)
    coordinates = np.array([
        [x, y] for y in np.linspace(low[1], high[1], RESOLUTION)
        for x in np.linspace(low[0], high[0], RESOLUTION)
    ])
    predictions = np.concatenate([model.decode(chunk) for chunk in np.array_split(coordinates, 20)])
    print(f'Searched {len(coordinates)} latent points; model weights frozen.', flush=True)
    records = []
    for number, (slug, label, rows) in enumerate(TARGETS, start=9):
        target = np.array([int(pixel) for row in rows for pixel in row], dtype=np.uint8)
        assert target.shape == (35,)
        assert novelty_distances(target[None, :], dataset.vectors)[0] > 0
        point, prediction, score = best_candidate(coordinates, predictions, target, dataset.vectors)
        span = (high - low) / (RESOLUTION - 1)
        for _ in range(3):
            local = np.array([
                [x, y]
                for y in np.linspace(max(low[1], point[1] - span[1]), min(high[1], point[1] + span[1]), 41)
                for x in np.linspace(max(low[0], point[0] - span[0]), min(high[0], point[0] + span[0]), 41)
            ])
            trial = best_candidate(local, model.decode(local), target, dataset.vectors)
            if trial[2] < score:
                point, prediction, score = trial
            span = span / 10
        regenerated = model.decode(point[None, :])[0]
        np.testing.assert_allclose(regenerated, prediction, atol=1e-12, rtol=0)
        binary = (regenerated >= THRESHOLD).astype(np.uint8)
        distances = np.count_nonzero(binary != dataset.vectors, axis=1)
        assert distances.min() > 0
        assert np.count_nonzero(binary != target) == score[0]
        assert np.all(point >= low) and np.all(point <= high)
        assert not np.any(np.all(np.isclose(original_codes, point, atol=1e-12, rtol=0), axis=1))
        name = f'{number:02d}-generation-{slug}'
        record = {
            'slug': slug, 'label': label,
            'target': target.tolist(), 'coordinates': point.tolist(),
            'continuous': regenerated.tolist(), 'binary': binary.tolist(),
            'target_pixel_errors': score[0], 'target_mse': score[1],
            'min_hamming': int(distances.min()),
            'nearest_originals': [dataset.labels[i] for i in np.flatnonzero(distances == distances.min())],
            'target_min_hamming': int(novelty_distances(target[None, :], dataset.vectors)[0]),
            'figures': [f'{name}.png', f'{name}.svg'],
        }
        plot_comparison(record, name)
        records.append(record)
        print(f"{label}: {score[0]}/35 target differences; novelty distance {distances.min()}.", flush=True)
    assert sha256(MODEL) == model_hash
    assert sha256(Path(dataset.path)) == dataset.sha256
    assert len({tuple(record['binary']) for record in records}) == len(records)
    provenance = {
        'model': 'results/final/model.npz', 'model_sha256': model_hash,
        'dataset': 'data/font.h', 'dataset_sha256': dataset.sha256,
        'threshold': THRESHOLD, 'image_shape': [7, 5],
        'search': {
            'method': 'target-guided latent search; decoder weights fixed',
            'grid_resolution_per_axis': RESOLUTION, 'grid_points': len(coordinates),
            'local_refinements': 3, 'local_resolution_per_axis': 41,
            'range_min': low.tolist(), 'range_max': high.tolist(),
            'ranking': ['binary_target_pixel_errors', 'target_mse'],
            'eligibility': 'minimum Hamming distance to the 32 originals > 0',
            'targets_are_training_data': False,
        },
        'verification': {
            'all_targets_absent_from_dataset': True,
            'all_outputs_absent_from_dataset': True,
            'all_outputs_regenerated_from_saved_coordinates': True,
            'model_and_dataset_unchanged': True,
        },
        'interpretation': 'Best novel outputs found by this finite search; no global optimum or semantic recognition claim.',
        'examples': records,
    }
    (OUT / 'generation-provenance.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
