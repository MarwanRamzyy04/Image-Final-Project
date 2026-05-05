# Medical Image Workbench - Phase 1

## Overview

Desktop GUI for spatial domain operations and core architecture.

## Setup

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Run the app:
   - `python main.py`

## Notes

- DICOM is loaded with pydicom, then normalized to 8-bit for processing.
- Zoom uses custom nearest-neighbor or bilinear interpolation.
- Convolution, median, and local histogram equalization are implemented from scratch.
