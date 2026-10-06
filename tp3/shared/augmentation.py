"""Train-only integer translations of flattened 28x28 images, with zero padding."""
import numpy as np


def validate_augmentation(config, input_size):
    if config is None:
        return None
    if not isinstance(config, dict) or input_size != 784:
        raise ValueError('Augmentation requires784 inputs and a configuration object.')
    name = config.get('name')
    keys = {'name', 'max_shift'}
    if name == 'translation_rotation':
        keys.add('max_angle_degrees')
        angle = config.get('max_angle_degrees')
        if (isinstance(angle, bool) or not isinstance(angle, (int, float))
                or not np.isfinite(angle) or not 0 < angle <= 5):
            raise ValueError('Rotation maximum must be finite and between0 and5 degrees.')
    elif name != 'translation':
        raise ValueError('Unknown augmentation name.')
    shift = config.get('max_shift')
    if (set(config) != keys or isinstance(shift, bool) or not isinstance(shift, int)
            or not 1 <= shift < 28):
        raise ValueError('Invalid augmentation settings.')
    return dict(config)


def rotate_images(X, max_angle_degrees, rng):
    """Independent uniform angles about image center; bilinear inverse sampling, zero padding."""
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 784:
        raise ValueError('Rotations require flattened28x28 images.')
    if (isinstance(max_angle_degrees, bool) or not isinstance(max_angle_degrees, (int, float))
            or not np.isfinite(max_angle_degrees) or not 0 <= max_angle_degrees <= 5):
        raise ValueError('Rotation maximum must be between0 and5 degrees.')
    if max_angle_degrees == 0 or len(X) == 0:
        return X.copy()
    dtype = X.dtype if X.dtype.kind == 'f' else np.dtype('float64')
    angles = np.deg2rad(rng.uniform(-max_angle_degrees, max_angle_degrees, len(X)))[:, None]
    yy, xx = np.indices((28, 28), dtype=float)
    dx, dy = xx.reshape(1, -1) - 13.5, yy.reshape(1, -1) - 13.5
    cosine, sine = np.cos(angles), np.sin(angles)
    sx = cosine * dx + sine * dy + 13.5
    sy = -sine * dx + cosine * dy + 13.5
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    wx, wy = sx-x0, sy-y0
    result = np.zeros(X.shape, dtype=dtype)
    rows = np.arange(len(X))[:, None]
    for ox, oy, weight in ((0, 0, (1-wx)*(1-wy)), (1, 0, wx*(1-wy)),
                           (0, 1, (1-wx)*wy), (1, 1, wx*wy)):
        xi, yi = x0+ox, y0+oy
        valid = (xi >= 0) & (xi < 28) & (yi >= 0) & (yi < 28)
        values = X[rows, np.clip(yi, 0, 27)*28 + np.clip(xi, 0, 27)]
        result += (values * weight * valid).astype(dtype)
    return result


def translate_images(X, max_shift, rng):
    """Uniform independent row/column shifts; no wrap, interpolation or label change."""
    if isinstance(max_shift, bool) or not isinstance(max_shift, int) or not 0 <= max_shift < 28:
        raise ValueError('max_shift must be an integer between 0 and 27.')
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 784:
        raise ValueError('Translations require flattened 28x28 images.')
    if max_shift == 0:
        return X.copy()
    shifts = rng.integers(-max_shift, max_shift + 1, size=(len(X), 2))
    source = X.reshape(-1, 28, 28)
    result = np.zeros_like(source)
    for dy, dx in np.unique(shifts, axis=0):
        indices = np.flatnonzero(np.all(shifts == (dy, dx), axis=1))
        sy, sx = max(0, -dy), max(0, -dx)
        ty, tx = max(0, dy), max(0, dx)
        height, width = 28 - abs(dy), 28 - abs(dx)
        result[indices, ty:ty+height, tx:tx+width] = source[indices, sy:sy+height, sx:sx+width]
    return result.reshape(X.shape)
