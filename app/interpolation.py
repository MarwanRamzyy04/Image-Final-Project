from __future__ import annotations

import numpy as np

from app.utils import clamp_index


def nearest_neighbor_resize(image: np.ndarray, scale: float) -> np.ndarray:
    """
    Nearest-neighbour resize implemented from scratch.

    FIX: Inverse mapping now uses floor (int truncation) — NOT round().
    Using round() caused half-pixel drift at boundaries.

    Correct formula:
        src_y = int(y / scale)   <- floor / truncation
        src_x = int(x / scale)

    This maps each output pixel back to the nearest source pixel by
    simply truncating the real-valued coordinate, which is the correct
    definition of nearest-neighbour interpolation.
    """
    if scale <= 0:
        raise ValueError("Scale must be positive")
    h, w = image.shape[:2]
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    output = np.zeros((new_h, new_w), dtype=np.float32)
    for y in range(new_h):
        src_y = clamp_index(int(y / scale), 0, h - 1)
        for x in range(new_w):
            src_x = clamp_index(int(x / scale), 0, w - 1)
            output[y, x] = float(image[src_y, src_x])
    return output


def bilinear_resize(image: np.ndarray, scale: float) -> np.ndarray:
    """
    Bilinear resize implemented from scratch.

    Inverse mapping: src = dst / scale  (continuous coordinate)
    4-neighbour interpolation:
        top    = v(y0,x0)*(1-wx) + v(y0,x1)*wx
        bottom = v(y1,x0)*(1-wx) + v(y1,x1)*wx
        out    = top*(1-wy)       + bottom*wy

    wy and wx are the fractional parts of the continuous source coordinates.
    All boundary accesses are clamped.
    """
    if scale <= 0:
        raise ValueError("Scale must be positive")
    h, w = image.shape[:2]
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    output = np.zeros((new_h, new_w), dtype=np.float32)
    for y in range(new_h):
        src_y = y / scale
        y0 = int(src_y)                          # floor
        wy = src_y - y0                          # fractional part (before clamping)
        y0 = clamp_index(y0,     0, h - 1)
        y1 = clamp_index(y0 + 1, 0, h - 1)
        for x in range(new_w):
            src_x = x / scale
            x0 = int(src_x)
            wx = src_x - x0
            x0 = clamp_index(x0,     0, w - 1)
            x1 = clamp_index(x0 + 1, 0, w - 1)
            v00 = float(image[y0, x0])
            v01 = float(image[y0, x1])
            v10 = float(image[y1, x0])
            v11 = float(image[y1, x1])
            top    = v00 * (1 - wx) + v01 * wx
            bottom = v10 * (1 - wx) + v11 * wx
            output[y, x] = top * (1 - wy) + bottom * wy
    return output