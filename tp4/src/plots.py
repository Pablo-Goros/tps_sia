"""Helpers de gráficos (matplotlib) compartidos por los scripts."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save(fig, path: str | Path, dpi: int = 130):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def heatmap(M, title="", xlabel="", ylabel="", annot=None, cmap="viridis", fmt="{:.2f}",
            cbar_label="", figsize=(6, 5), xticks=None, yticks=None, ax=None, vmin=None, vmax=None,
            fontsize=6.5, text_color=None):
    """text_color: None -> blanco/negro según la luminancia de la celda; o un color fijo."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(xticks, rotation=90, fontsize=7)
    if yticks is not None:
        ax.set_yticks(range(len(yticks)))
        ax.set_yticklabels(yticks, fontsize=7)
    if annot is not None:
        M = np.asarray(M, dtype=float)
        # color del texto según la luminancia real del color de fondo de la celda
        norm = im.norm
        cm = im.cmap
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                a = annot[i][j] if not isinstance(annot, bool) else fmt.format(M[i, j])
                if np.isfinite(M[i, j]):
                    r, g, b, _ = cm(norm(M[i, j]))
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                else:
                    lum = 1.0
                ax.text(j, i, a, ha="center", va="center", fontsize=fontsize,
                        color=text_color or ("white" if lum < 0.5 else "black"))
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cb.set_label(cbar_label)
    return (fig, ax) if own else ax


def pattern_grid(patterns, titles, size, ncols=5, figsize=None, suptitle=""):
    n = len(patterns)
    nrows = int(np.ceil(n / ncols))
    figsize = figsize or (2.2 * ncols, 2.4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, p, t in zip(axes.ravel(), patterns, titles):
        ax.imshow(np.asarray(p).reshape(size, size), cmap="gray_r", vmin=-1, vmax=1)
        ax.set_title(t, fontsize=8)
    if suptitle:
        fig.suptitle(suptitle)
    return fig
