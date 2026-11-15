"""TP3 digit images as the new dataset of exercise 2.a (read-only reuse)."""
from pathlib import Path
import sys

import numpy as np

_TP5 = Path(__file__).resolve().parents[1]
_ROOT = _TP5.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from tp3.shared.digit_dataset import cargar

FILES = {'digits': 'digits.csv', 'more_digits': 'more_digits.csv', 'test': 'digits_test.csv'}


def load_digits(split='digits', cache_dir=None):
    """Return (images (N,784) float64 in [0,1], labels (N,) int).

    ``digits`` has no 8 and few 5; ``more_digits`` contains the ten digits;
    ``test`` is the held-out TP3 file. Parsing the CSV is slow, so the TP3 loader caches it as NPZ under
    ``results/cache`` (ignored by Git). The CSV files are never modified.
    """
    if split not in FILES:
        raise ValueError(f'split must be one of {sorted(FILES)}')
    cache_dir = Path(cache_dir) if cache_dir else _TP5 / 'results' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    x, y = cargar(_ROOT / 'tp3' / 'data' / FILES[split], cache_dir / f'digits-{split}.npz')
    x = np.asarray(x, dtype=float)
    if x.min() < 0 or x.max() > 1:
        raise ValueError('Digit pixels are expected in [0, 1]')
    return x, np.argmax(y, axis=1)
