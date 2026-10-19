"""Carga de códigos QR como matrices binarias 37x37 en {1, -1}.

Convención: módulo oscuro (negro) = +1 (neurona activa), claro = -1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

QR_SIZE = 37


def load_qr(path: str | Path, size: int = QR_SIZE) -> np.ndarray:
    """Lee un PNG y lo convierte a una matriz size x size en {1, -1}.
    Si la imagen es más grande (p. ej. 370x370 con módulos de 10 px) se
    muestrea el centro de cada módulo."""
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=float)
    if arr.shape != (size, size):
        h, w = arr.shape
        ys = ((np.arange(size) + 0.5) * h / size).astype(int)
        xs = ((np.arange(size) + 0.5) * w / size).astype(int)
        arr = arr[np.ix_(ys, xs)]
    return np.where(arr < 128, 1.0, -1.0)


def load_qr_folder(folder: str | Path) -> dict[str, np.ndarray]:
    folder = Path(folder)
    out = {}
    for p in sorted(folder.glob("*.png")):
        out[p.stem.replace("_qr", "").replace("_menu", "")] = load_qr(p)
    return out


def to_image(pattern: np.ndarray, size: int = QR_SIZE, scale: int = 10, border: int = 4) -> np.ndarray:
    """Vuelve a imagen (uint8, 0 = negro) con zoom y margen silencioso para un lector."""
    m = np.asarray(pattern).reshape(size, size)
    img = np.where(m > 0, 0, 255).astype(np.uint8)
    img = np.kron(img, np.ones((scale, scale), dtype=np.uint8))
    img = np.pad(img, border * scale, constant_values=255)
    return img


def decode(pattern: np.ndarray, size: int = QR_SIZE) -> str:
    """Intenta decodificar el patrón con un lector de QR real (OpenCV).
    Devuelve '' si no se pudo leer."""
    try:
        import cv2
    except ImportError:          # pragma: no cover
        return "<cv2 no instalado>"
    img = to_image(pattern, size)
    txt, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
    return txt
