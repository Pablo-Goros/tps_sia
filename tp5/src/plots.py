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


def plot_latent(result, output):
    """Use separate label lanes so coincident points retain all identities."""
    import numpy as np
    points = result['points']
    coordinates = np.array([point['coordinates'] for point in points])
    figure, axis = plt.subplots(figsize=(14, 11))
    figure.subplots_adjust(left=0.22, right=0.78, bottom=0.14, top=0.9)
    axis.scatter(coordinates[:, 0], coordinates[:, 1], s=30, color='tab:blue')
    # Split by x, sort each lane by y; every pattern gets its own label slot.
    ordered = np.argsort(coordinates[:, 0], kind='stable')
    for lane, indices in enumerate((ordered[:16], ordered[16:])):
        indices = sorted(indices, key=lambda index: coordinates[index, 1])
        for height, index in zip(np.linspace(0.025, 0.975, 16), indices):
            axis.annotate(points[index]['label'], xy=coordinates[index],
                          xytext=(-0.09 if lane == 0 else 1.09, height),
                          textcoords='axes fraction', ha='right' if lane == 0 else 'left',
                          va='center', fontsize=10, annotation_clip=False,
                          arrowprops={'arrowstyle': '-', 'color': '0.65', 'linewidth': 0.6})
    axis.set(xlabel='Código latente 1', ylabel='Código latente 2',
             title='Font3: los 32 patrones en el espacio latente')
    axis.yaxis.set_label_coords(-0.20, 0.5)
    axis.grid(alpha=0.2)
    figure.text(0.04, 0.05, 'Modelo SHA256: ' + result['model']['sha256'] + '\n'
                + 'Dataset SHA256: ' + result['dataset']['sha256'], fontsize=8,
                family='monospace')
    figure.savefig(output, dpi=140)
    plt.close(figure)
