from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pydicom
from PIL import Image

from app.utils import normalize_to_uint8, to_grayscale


SUPPORTED_EXTENSIONS = {".dcm", ".dicom", ".jpg", ".jpeg", ".bmp", ".png"}

# Maps PIL image mode to bits-per-channel (medically meaningful bit depth).
# "Bit depth" in medical imaging = bits stored per channel, not total bits per pixel.
_MODE_BIT_DEPTH: Dict[str, int] = {
    "1":    1,
    "L":    8,
    "P":    8,
    "RGB":  8,
    "RGBA": 8,
    "CMYK": 8,
    "YCbCr": 8,
    "I":   32,
    "F":   32,
}


def _dicom_metadata(dataset: pydicom.Dataset) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    metadata["Format"]       = "DICOM"
    metadata["Width"]        = str(getattr(dataset, "Columns",          "Unknown"))
    metadata["Height"]       = str(getattr(dataset, "Rows",             "Unknown"))
    metadata["Bit depth"]    = str(getattr(dataset, "BitsAllocated",    "Unknown"))
    metadata["Modality"]     = str(getattr(dataset, "Modality",         "Unknown"))
    metadata["Patient Name"] = str(getattr(dataset, "PatientName",      "Unknown"))
    metadata["Patient Age"]  = str(getattr(dataset, "PatientAge",       "Unknown"))
    metadata["Body Part"]    = str(getattr(dataset, "BodyPartExamined", "Unknown"))
    return metadata


def _standard_metadata(image: Image.Image) -> Dict[str, str]:
    """
    FIX: Bit depth is now reported as bits-per-channel, not total bits per pixel.

    Old (wrong):  len(image.getbands()) * 8  → gives 24 for RGB, 32 for RGBA
    New (correct): look up the mode in _MODE_BIT_DEPTH → gives 8 for RGB/RGBA,
                   consistent with how DICOM reports BitsAllocated.
    """
    metadata: Dict[str, str] = {}
    metadata["Format"]    = str(image.format or "Unknown")
    metadata["Width"]     = str(image.width)
    metadata["Height"]    = str(image.height)
    metadata["Bit depth"] = str(_MODE_BIT_DEPTH.get(image.mode, 8))
    return metadata


def load_image(file_path: str) -> Tuple[np.ndarray, Dict[str, str]]:
    """
    Load DICOM, JPEG, BMP, or PNG image.
    Returns (grayscale uint8 array, metadata dict).
    Raises on unsupported format or corrupted file — caller must catch.
    """
    lower = file_path.lower()
    if lower.endswith((".dcm", ".dicom")):
        dataset = pydicom.dcmread(file_path)
        pixel_array = dataset.pixel_array.astype(np.float32)
        slope     = float(getattr(dataset, "RescaleSlope",     1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        pixel_array = pixel_array * slope + intercept
        # Multi-frame DICOM: take first frame only
        if pixel_array.ndim > 2:
            pixel_array = pixel_array[0]
        metadata = _dicom_metadata(dataset)
        return normalize_to_uint8(pixel_array), metadata

    # Standard image formats
    image = Image.open(file_path)
    metadata = _standard_metadata(image)
    image_array = np.array(image)
    gray = to_grayscale(image_array)
    return normalize_to_uint8(gray), metadata


def save_image(file_path: str, image: np.ndarray) -> None:
    """Save the processed image (normalised to uint8) to disk."""
    image_uint8 = normalize_to_uint8(image)
    pil_image = Image.fromarray(image_uint8)
    pil_image.save(file_path)