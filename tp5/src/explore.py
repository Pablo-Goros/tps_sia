"""Reload-only exploration of the two-dimensional character codes."""
import argparse
from pathlib import Path

from .autoencoder import Autoencoder
from .dataset import DEFAULT_DATASET, load_dataset
from .plots import plot_latent
from .train import model_identity, write_json


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


def main():
    parser = argparse.ArgumentParser(description='Visualize saved autoencoder codes without training')
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--output', type=Path, default=Path('results/latent'))
    args = parser.parse_args()
    try:
        visualize_latent(args.model, args.output, args.dataset)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f'Latent codes and figure: {args.output}')


if __name__ == '__main__':
    main()
