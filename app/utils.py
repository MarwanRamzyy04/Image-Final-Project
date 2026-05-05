from __future__ import annotations

from typing import Dict

import numpy as np
from PyQt6.QtGui import QImage


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB/RGBA array to grayscale using the standard luminance formula."""
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.ndim == 3 and image.shape[2] >= 3:
        r = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        b = image[:, :, 2].astype(np.float32)
        return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
    return image.astype(np.float32)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Linearly scale any float/int array to the uint8 range [0, 255].
    Returns an array of zeros if max == min (flat image).
    """
    image = image.astype(np.float32)
    min_val = float(image.min())
    max_val = float(image.max())
    if max_val - min_val < 1e-6:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image - min_val) * (255.0 / (max_val - min_val))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return normalize_to_uint8(image)


def clamp_index(value: int, lower: int, upper: int) -> int:
    """Clamp an integer index to [lower, upper]. Pure Python — no numpy."""
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def numpy_to_qimage(image: np.ndarray) -> QImage:
    """
    Convert a uint8 numpy array to a QImage suitable for display.
    Calls .copy() to ensure Qt doesn't hold a dangling pointer into numpy memory.
    """
    image = ensure_uint8(image)
    if image.ndim == 2:
        h, w = image.shape
        qimage = QImage(image.data, w, h, w, QImage.Format.Format_Grayscale8)
        return qimage.copy()
    h, w, c = image.shape
    if c >= 3:
        rgb = np.ascontiguousarray(image[:, :, :3])
        bytes_per_line = w * 3
        qimage = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return qimage.copy()
    gray = np.ascontiguousarray(image[:, :, 0])
    qimage = QImage(gray.data, w, h, w, QImage.Format.Format_Grayscale8)
    return qimage.copy()


def metadata_to_text(metadata: Dict[str, str]) -> str:
    return "\n".join(f"{k}: {v}" for k, v in metadata.items())