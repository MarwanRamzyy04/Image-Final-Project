from __future__ import annotations

import numpy as np


def _equalize_block(block: np.ndarray) -> np.ndarray:
    """
    Local histogram equalization for a single block.

    Histogram and CDF are built from scratch using plain loops (no np.histogram).

    FIX: The equalization formula now correctly subtracts cdf_min so that
    the output spans the full [0, 255] range:

        out = round( (cdf[v] - cdf_min) * 255 / (total - cdf_min) )

    The old formula  cdf[v] * 255 / total  was wrong — it compressed the
    dynamic range whenever the darkest pixel value was not 0.
    """
    h, w = block.shape
    total = h * w

    # --- build histogram from scratch ---
    hist = [0] * 256
    for y in range(h):
        for x in range(w):
            hist[int(block[y, x])] += 1

    # --- build cumulative distribution function ---
    cdf = [0] * 256
    cumulative = 0
    for i in range(256):
        cumulative += hist[i]
        cdf[i] = cumulative

    # --- find cdf_min (first non-zero CDF entry) ---
    cdf_min = 0
    for i in range(256):
        if cdf[i] > 0:
            cdf_min = cdf[i]
            break

    output = np.zeros((h, w), dtype=np.uint8)
    if total == 0:
        return output

    denom = total - cdf_min
    if denom == 0:
        # All pixels identical — return flat block
        return output

    # --- apply equalization mapping ---
    for y in range(h):
        for x in range(w):
            v = int(block[y, x])
            output[y, x] = int(round((cdf[v] - cdf_min) * 255 / denom))

    return output


def local_hist_equalization(image: np.ndarray, block_size: int) -> np.ndarray:
    """
    Divide image into non-overlapping blocks of size block_size x block_size
    and equalize each block independently.
    Partial (edge) blocks are handled automatically by NumPy slicing.
    """
    if block_size <= 0:
        raise ValueError("Block size must be positive")
    image = image.astype(np.uint8)
    h, w = image.shape
    output = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = image[y: y + block_size, x: x + block_size]
            output[y: y + block_size, x: x + block_size] = _equalize_block(block)
    return output