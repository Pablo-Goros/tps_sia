"""Non-interactive figures for the public TP5 workflows."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_dataset(dataset, output):
    """Save all 32 original patterns, preserving image orientation."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(4, 8, figsize=(16, 9))
    for image, label, axis in zip(dataset.images, dataset.labels, axes.flat):
        axis.imshow(image, cmap='gray_r', vmin=0, vmax=1, interpolation='nearest', origin='upper')
        axis.set_title(label, fontsize=11)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle('Font3: 32 patrones originales (7 filas × 5 columnas)')
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)
    return output
