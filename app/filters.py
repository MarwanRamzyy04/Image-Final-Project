from __future__ import annotations

from typing import Tuple

import numpy as np

from app.utils import clamp_index


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    2D convolution implemented from scratch using 4 nested loops.
    Boundary handling: clamp (replicate border pixels).
    No library convolution functions are used.
    """
    image = image.astype(np.float32)
    kernel = kernel.astype(np.float32)
    h, w = image.shape
    kh, kw = kernel.shape
    pad_y = kh // 2
    pad_x = kw // 2
    output = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            acc = 0.0
            for ky in range(kh):
                for kx in range(kw):
                    iy = clamp_index(y + ky - pad_y, 0, h - 1)
                    ix = clamp_index(x + kx - pad_x, 0, w - 1)
                    acc += image[iy, ix] * kernel[ky, kx]
            output[y, x] = acc
    return output


def average_kernel(size: int) -> np.ndarray:
    if size <= 0 or size % 2 == 0:
        raise ValueError("Kernel size must be positive and odd")
    value = 1.0 / (size * size)
    return np.full((size, size), value, dtype=np.float32)


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    if size <= 0 or size % 2 == 0:
        raise ValueError("Kernel size must be positive and odd")
    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)
    two_sigma_sq = 2.0 * sigma * sigma
    total = 0.0
    for y in range(size):
        for x in range(size):
            dy = y - center
            dx = x - center
            value = np.exp(-(dx * dx + dy * dy) / two_sigma_sq)
            kernel[y, x] = value
            total += value
    if total > 0:
        kernel /= total
    return kernel


def average_filter(image: np.ndarray, size: int) -> np.ndarray:
    return convolve2d(image, average_kernel(size))


def gaussian_filter(image: np.ndarray, size: int, sigma: float) -> np.ndarray:
    return convolve2d(image, gaussian_kernel(size, sigma))


def median_filter(image: np.ndarray, size: int) -> np.ndarray:
    if size <= 0 or size % 2 == 0:
        raise ValueError("Kernel size must be positive and odd")
    image = image.astype(np.float32)
    h, w = image.shape
    pad = size // 2
    output = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            values = []
            for ky in range(-pad, pad + 1):
                for kx in range(-pad, pad + 1):
                    iy = clamp_index(y + ky, 0, h - 1)
                    ix = clamp_index(x + kx, 0, w - 1)
                    values.append(float(image[iy, ix]))
            values.sort()
            output[y, x] = values[len(values) // 2]
    return output


def sobel_kernels() -> Tuple[np.ndarray, np.ndarray]:
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    return gx, gy


def prewitt_kernels() -> Tuple[np.ndarray, np.ndarray]:
    gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    gy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    return gx, gy


def edge_components(
    image: np.ndarray, kernels: Tuple[np.ndarray, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Gx result, Gy result) separately so the GUI can display each independently."""
    gx_kernel, gy_kernel = kernels
    gx = convolve2d(image, gx_kernel)
    gy = convolve2d(image, gy_kernel)
    return gx, gy


def edge_magnitude(
    image: np.ndarray, kernels: Tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    gx, gy = edge_components(image, kernels)
    return np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)