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


def plot_reconstructions(dataset, predictions, metrics, output):
    import numpy as np
    figure, axes = plt.subplots(4, 8, figsize=(16, 10))
    for index, axis in enumerate(axes.flat):
        pair = np.concatenate([dataset.images[index], (predictions[index] >= 0.5).reshape(7, 5)], axis=1)
        axis.imshow(pair, cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
        axis.axvline(4.5, color='red', linewidth=0.7)
        axis.set_title(f'{dataset.labels[index]}: {metrics["pixel_errors"][index]} errores', fontsize=9)
        axis.axis('off')
    figure.suptitle('Original | reconstrucción binaria (umbral >= 0.5)')
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def plot_training(history, output):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history['epochs'], history['mse'])
    axes[0].set(xlabel='Época', ylabel='MSE por píxel')
    points = history['evaluations']
    axes[1].plot([p['epoch'] for p in points], [p['max_pixel_errors'] for p in points])
    axes[1].axhline(1, color='red', linestyle='--')
    axes[1].set(xlabel='Época', ylabel='Máximo de errores por patrón')
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)
