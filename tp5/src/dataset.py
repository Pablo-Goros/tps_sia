"""Load the original Font3 C initializer without executing it."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import numpy as np


DEFAULT_DATASET = Path(__file__).resolve().parents[1] / 'data' / 'font.h'


@dataclass
class Dataset:
    images: np.ndarray
    vectors: np.ndarray
    labels: list[str]
    sha256: str
    path: str


def load_dataset(path=DEFAULT_DATASET):
    """Return ordered images and float row-major vectors from Font3[32][7]."""
    path = Path(path)
    raw = path.read_bytes()
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', raw.decode('utf-8'), flags=re.DOTALL)
    match = re.search(r'\bFont3\s*\[\s*32\s*\]\s*\[\s*7\s*\]\s*=\s*\{(.*?)\}\s*;', text, re.DOTALL)
    if match is None:
        raise ValueError('Expected a complete Font3[32][7] initializer')
    body = match.group(1)
    rows = re.findall(r'\{([^{}]*)\}', body)
    if re.sub(r'\{[^{}]*\}|[\s,]', '', body):
        raise ValueError('Unexpected content outside Font3 patterns')
    if len(rows) != 32:
        raise ValueError(f'Font3 must contain 32 patterns; found {len(rows)}')
    values = []
    for index, row in enumerate(rows):
        tokens = row.split(',')
        if tokens[-1].strip() == '':
            tokens.pop()
        if len(tokens) != 7:
            raise ValueError(f'Pattern {index} must contain 7 rows; found {len(tokens)}')
        try:
            integers = [int(token.strip(), 0) for token in tokens]
        except ValueError as error:
            raise ValueError(f'Pattern {index} contains an invalid integer') from error
        if any(value < 0 or value > 31 for value in integers):
            raise ValueError(f'Pattern {index} has a row outside 0..31 (five bits)')
        values.append(integers)
    images = ((np.array(values)[:, :, None] >> np.arange(4, -1, -1)) & 1).astype(np.uint8)
    labels = [f'0x{code:02x} ' + (chr(code) if code != 127 else 'DEL') for code in range(96, 128)]
    return Dataset(images, images.reshape(32, 35).astype(float), labels,
                   hashlib.sha256(raw).hexdigest(), str(path.resolve()))


def main():
    import argparse
    import json
    from .plots import plot_dataset

    parser = argparse.ArgumentParser(description='Inspect the original Font3 dataset')
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--output', type=Path, default=Path('results/dataset'))
    args = parser.parse_args()
    try:
        dataset = load_dataset(args.dataset)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    sheet = plot_dataset(dataset, args.output / 'dataset.png')
    (args.output / 'dataset.json').write_text(json.dumps({
        'dataset': dataset.path, 'sha256': dataset.sha256, 'labels': dataset.labels,
        'image_shape': list(dataset.images.shape), 'vector_shape': list(dataset.vectors.shape),
    }, indent=2), encoding='utf-8')
    print(f'Loaded 32 patterns; sheet: {sheet}')


if __name__ == '__main__':
    main()
