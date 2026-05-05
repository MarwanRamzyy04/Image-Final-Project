from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app import filters, histogram, image_io, interpolation
from app.pipeline import Pipeline
from app.utils import metadata_to_text, normalize_to_uint8, numpy_to_qimage


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Medical Image Workbench – Phase 1")
        self.pipeline = Pipeline()
        self.original_image: Optional[np.ndarray] = None
        self.current_scale = 1.0
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ── Image tabs (Original / Processed) ──────────────────────────
        self.tabs = QTabWidget()

        self.original_label = QLabel("Load an image to begin")
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.processed_label = QLabel("No processing applied yet")
        self.processed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        orig_scroll = QScrollArea()
        orig_scroll.setWidgetResizable(True)
        orig_scroll.setWidget(self.original_label)

        proc_scroll = QScrollArea()
        proc_scroll.setWidgetResizable(True)
        proc_scroll.setWidget(self.processed_label)

        self.tabs.addTab(orig_scroll, "Original")
        self.tabs.addTab(proc_scroll, "Processed")

        # ── Metadata panel ─────────────────────────────────────────────
        self.metadata_label = QLabel()
        self.metadata_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.metadata_label.setWordWrap(True)

        meta_scroll = QScrollArea()
        meta_scroll.setWidgetResizable(True)
        meta_scroll.setWidget(self.metadata_label)

        meta_group = QGroupBox("Image Metadata")
        meta_layout = QVBoxLayout()
        meta_layout.addWidget(meta_scroll)
        meta_group.setLayout(meta_layout)

        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.tabs, stretch=4)
        left_layout.addWidget(meta_group, stretch=1)
        left_widget.setLayout(left_layout)

        # ── Control panel (right side) ──────────────────────────────────
        control_panel = self._build_controls()

        splitter = QSplitter()
        splitter.addWidget(left_widget)
        splitter.addWidget(control_panel)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        container = QWidget()
        container_layout = QHBoxLayout()
        container_layout.addWidget(splitter)
        container.setLayout(container_layout)
        self.setCentralWidget(container)

    def _build_controls(self) -> QWidget:
        # ── Load / Save / Reset / Undo ─────────────────────────────────
        load_btn  = QPushButton("Load Image")
        save_btn  = QPushButton("Save Processed")
        reset_btn = QPushButton("Reset to Original")
        undo_btn  = QPushButton("Undo Last Step")

        load_btn.clicked.connect(self._on_load)
        save_btn.clicked.connect(self._on_save)
        reset_btn.clicked.connect(self._on_reset)
        undo_btn.clicked.connect(self._on_undo)

        io_row = QHBoxLayout()
        io_row.addWidget(load_btn)
        io_row.addWidget(save_btn)
        io_widget = QWidget()
        io_widget.setLayout(io_row)

        # ── Zoom controls ───────────────────────────────────────────────
        self.interp_combo = QComboBox()
        self.interp_combo.addItems(["Nearest-Neighbor", "Linear (Bilinear)"])

        zoom_in_btn  = QPushButton("Zoom In  (+25 %)")
        zoom_out_btn = QPushButton("Zoom Out (−20 %)")
        zoom_in_btn.clicked.connect(lambda: self._on_zoom(1.25))
        zoom_out_btn.clicked.connect(lambda: self._on_zoom(0.8))

        zoom_group = QGroupBox("Zoom / Interpolation")
        zoom_layout = QVBoxLayout()
        zoom_layout.addWidget(QLabel("Interpolation method:"))
        zoom_layout.addWidget(self.interp_combo)
        zoom_layout.addWidget(zoom_in_btn)
        zoom_layout.addWidget(zoom_out_btn)
        zoom_group.setLayout(zoom_layout)

        # ── Spatial filters ─────────────────────────────────────────────
        self.kernel_size = QComboBox()
        self.kernel_size.addItems(["3", "5", "7", "9"])

        self.gaussian_sigma = QDoubleSpinBox()
        self.gaussian_sigma.setRange(0.1, 10.0)
        self.gaussian_sigma.setValue(1.0)
        self.gaussian_sigma.setSingleStep(0.1)
        self.gaussian_sigma.setPrefix("σ = ")

        avg_btn    = QPushButton("Average")
        gauss_btn  = QPushButton("Gaussian")
        median_btn = QPushButton("Median")

        avg_btn.clicked.connect(self._apply_average)
        gauss_btn.clicked.connect(self._apply_gaussian)
        median_btn.clicked.connect(self._apply_median)

        # Sobel – three separate buttons (Gx, Gy, Magnitude)
        sobel_gx_btn  = QPushButton("Gx")
        sobel_gy_btn  = QPushButton("Gy")
        sobel_mag_btn = QPushButton("Magnitude")
        sobel_gx_btn.clicked.connect(self._apply_sobel_gx)
        sobel_gy_btn.clicked.connect(self._apply_sobel_gy)
        sobel_mag_btn.clicked.connect(self._apply_sobel_magnitude)

        sobel_row = QHBoxLayout()
        sobel_row.addWidget(sobel_gx_btn)
        sobel_row.addWidget(sobel_gy_btn)
        sobel_row.addWidget(sobel_mag_btn)

        # Prewitt – three separate buttons
        prewitt_gx_btn  = QPushButton("Gx")
        prewitt_gy_btn  = QPushButton("Gy")
        prewitt_mag_btn = QPushButton("Magnitude")
        prewitt_gx_btn.clicked.connect(self._apply_prewitt_gx)
        prewitt_gy_btn.clicked.connect(self._apply_prewitt_gy)
        prewitt_mag_btn.clicked.connect(self._apply_prewitt_magnitude)

        prewitt_row = QHBoxLayout()
        prewitt_row.addWidget(prewitt_gx_btn)
        prewitt_row.addWidget(prewitt_gy_btn)
        prewitt_row.addWidget(prewitt_mag_btn)

        filter_group = QGroupBox("Spatial Filters")
        filter_layout = QVBoxLayout()
        filter_layout.addWidget(QLabel("Kernel size:"))
        filter_layout.addWidget(self.kernel_size)
        filter_layout.addWidget(QLabel("Gaussian σ (std. deviation):"))
        filter_layout.addWidget(self.gaussian_sigma)
        filter_layout.addWidget(avg_btn)
        filter_layout.addWidget(gauss_btn)
        filter_layout.addWidget(median_btn)
        filter_layout.addWidget(QLabel("Sobel edge detection:"))
        filter_layout.addLayout(sobel_row)
        filter_layout.addWidget(QLabel("Prewitt edge detection:"))
        filter_layout.addLayout(prewitt_row)
        filter_group.setLayout(filter_layout)

        # ── Local histogram equalization ────────────────────────────────
        self.block_size_spin = QSpinBox()
        self.block_size_spin.setRange(4, 256)
        self.block_size_spin.setValue(8)
        self.block_size_spin.setSingleStep(8)
        self.block_size_spin.setSuffix(" px")

        hist_btn = QPushButton("Apply Local Equalization")
        hist_btn.clicked.connect(self._apply_local_equalization)

        hist_group = QGroupBox("Local Histogram Equalization")
        hist_layout = QVBoxLayout()
        hist_layout.addWidget(QLabel("Block size:"))
        hist_layout.addWidget(self.block_size_spin)
        hist_layout.addWidget(hist_btn)
        hist_group.setLayout(hist_layout)

        # ── Pipeline step list ──────────────────────────────────────────
        self.pipeline_list = QListWidget()
        pipe_group = QGroupBox("Enhancement Pipeline")
        pipe_layout = QVBoxLayout()
        pipe_layout.addWidget(self.pipeline_list)
        pipe_group.setLayout(pipe_layout)

        # ── Assemble control panel ──────────────────────────────────────
        control_layout = QVBoxLayout()
        control_layout.addWidget(io_widget)
        control_layout.addWidget(reset_btn)
        control_layout.addWidget(undo_btn)
        control_layout.addWidget(zoom_group)
        control_layout.addWidget(filter_group)
        control_layout.addWidget(hist_group)
        control_layout.addWidget(pipe_group)
        control_layout.addStretch()

        control_widget = QWidget()
        control_widget.setLayout(control_layout)

        # Wrap in a scroll area so the panel is accessible on small screens
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(control_widget)
        scroll.setMinimumWidth(260)
        return scroll

    # ------------------------------------------------------------------
    # Slot handlers – I/O
    # ------------------------------------------------------------------

    def _on_load(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Medical Image",
            "",
            "Images (*.dcm *.dicom *.jpg *.jpeg *.bmp *.png)",
        )
        if not file_path:
            return
        try:
            image, metadata = image_io.load_image(file_path)
        except Exception as exc:
            self._show_error(f"Failed to load image:\n{exc}")
            return

        self.original_image = image
        self.pipeline.reset(image)
        self.current_scale = 1.0
        self.metadata_label.setText(metadata_to_text(metadata))
        self._refresh_pipeline_list()
        self._update_display()

    def _on_save(self) -> None:
        current = self.pipeline.current()
        if current is None:
            self._show_error("No processed image to save.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processed Image",
            "",
            "PNG (*.png);;BMP (*.bmp);;JPEG (*.jpg *.jpeg)",
        )
        if not file_path:
            return
        try:
            image_io.save_image(file_path, current)
        except Exception as exc:
            self._show_error(f"Failed to save image:\n{exc}")

    # ------------------------------------------------------------------
    # Slot handlers – Pipeline management
    # ------------------------------------------------------------------

    def _on_reset(self) -> None:
        if self.original_image is None:
            return
        self.pipeline.reset(self.original_image)
        self.current_scale = 1.0
        self._refresh_pipeline_list()
        self._update_display()

    def _on_undo(self) -> None:
        """
        FIX: zoom scale is now reset to 1.0 on undo, consistent with
        Reset and Load behaviour. Previously the zoom level persisted
        after undoing a step, which was confusing.
        """
        if not self.pipeline.can_undo():
            return
        self.pipeline.undo()
        self.current_scale = 1.0          # <-- FIX
        self._refresh_pipeline_list()
        self._update_display()

    # ------------------------------------------------------------------
    # Slot handlers – Zoom
    # ------------------------------------------------------------------

    def _on_zoom(self, factor: float) -> None:
        """
        Zoom is a VIEW operation only — it scales the display image using the custom
        interpolation algorithm but does NOT push a new entry onto the pipeline stack.
        This preserves full pixel fidelity: subsequent filters always operate on the
        original-resolution image, not on a rescaled copy.
        """
        if self.pipeline.current() is None:
            return
        new_scale = self.current_scale * factor
        # Clamp to a sensible range
        if new_scale < 0.05 or new_scale > 16.0:
            return
        self.current_scale = new_scale
        self._update_display()

    # ------------------------------------------------------------------
    # Slot handlers – Filters (thin wrappers → _apply_filter)
    # ------------------------------------------------------------------

    def _apply_average(self)          -> None: self._apply_filter("Average Filter",        self._op_average)
    def _apply_gaussian(self)         -> None: self._apply_filter("Gaussian Filter",       self._op_gaussian)
    def _apply_median(self)           -> None: self._apply_filter("Median Filter",         self._op_median)
    def _apply_sobel_gx(self)         -> None: self._apply_filter("Sobel Gx",              self._op_sobel_gx)
    def _apply_sobel_gy(self)         -> None: self._apply_filter("Sobel Gy",              self._op_sobel_gy)
    def _apply_sobel_magnitude(self)  -> None: self._apply_filter("Sobel Magnitude",       self._op_sobel_mag)
    def _apply_prewitt_gx(self)       -> None: self._apply_filter("Prewitt Gx",            self._op_prewitt_gx)
    def _apply_prewitt_gy(self)       -> None: self._apply_filter("Prewitt Gy",            self._op_prewitt_gy)
    def _apply_prewitt_magnitude(self)-> None: self._apply_filter("Prewitt Magnitude",     self._op_prewitt_mag)
    def _apply_local_equalization(self)->None: self._apply_filter("Local Hist. Equal.",    self._op_local_eq)

    def _apply_filter(self, name: str, operation) -> None:
        """
        Generic filter dispatcher:
        1. Get the current pipeline image.
        2. Run the operation.
        3. Push result onto the pipeline stack.
        4. Refresh UI.
        All exceptions are caught and shown as error dialogs — app never crashes.
        """
        current = self.pipeline.current()
        if current is None:
            self._show_error("Please load an image first.")
            return
        try:
            result = operation(current)
        except Exception as exc:
            self._show_error(f"Operation '{name}' failed:\n{exc}")
            return
        self.pipeline.apply(name, result)
        self._refresh_pipeline_list()
        self._update_display()

    # ------------------------------------------------------------------
    # Image operation lambdas
    # ------------------------------------------------------------------

    def _op_average(self, img: np.ndarray) -> np.ndarray:
        return filters.average_filter(img, int(self.kernel_size.currentText()))

    def _op_gaussian(self, img: np.ndarray) -> np.ndarray:
        return filters.gaussian_filter(
            img,
            int(self.kernel_size.currentText()),
            float(self.gaussian_sigma.value()),
        )

    def _op_median(self, img: np.ndarray) -> np.ndarray:
        return filters.median_filter(img, int(self.kernel_size.currentText()))

    def _op_sobel_gx(self, img: np.ndarray) -> np.ndarray:
        gx, _ = filters.edge_components(img, filters.sobel_kernels())
        return gx

    def _op_sobel_gy(self, img: np.ndarray) -> np.ndarray:
        _, gy = filters.edge_components(img, filters.sobel_kernels())
        return gy

    def _op_sobel_mag(self, img: np.ndarray) -> np.ndarray:
        return filters.edge_magnitude(img, filters.sobel_kernels())

    def _op_prewitt_gx(self, img: np.ndarray) -> np.ndarray:
        gx, _ = filters.edge_components(img, filters.prewitt_kernels())
        return gx

    def _op_prewitt_gy(self, img: np.ndarray) -> np.ndarray:
        _, gy = filters.edge_components(img, filters.prewitt_kernels())
        return gy

    def _op_prewitt_mag(self, img: np.ndarray) -> np.ndarray:
        return filters.edge_magnitude(img, filters.prewitt_kernels())

    def _op_local_eq(self, img: np.ndarray) -> np.ndarray:
        block = self.block_size_spin.value()
        return histogram.local_hist_equalization(normalize_to_uint8(img), block)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _refresh_pipeline_list(self) -> None:
        self.pipeline_list.clear()
        for i, step in enumerate(self.pipeline.list_steps()):
            self.pipeline_list.addItem(f"{i}. {step}")
        # Scroll to most recent step
        self.pipeline_list.scrollToBottom()

    def _update_display(self) -> None:
        """
        Resize both the original and the processed image using the
        currently selected interpolation method and current zoom scale,
        then push them to their respective tab labels.
        """
        original = self.pipeline.original
        current  = self.pipeline.current()
        if original is None or current is None:
            return

        use_bilinear = self.interp_combo.currentText().startswith("Linear")
        resize_fn = (
            interpolation.bilinear_resize
            if use_bilinear
            else interpolation.nearest_neighbor_resize
        )

        try:
            disp_orig = resize_fn(original, self.current_scale)
            disp_curr = resize_fn(current,  self.current_scale)
        except Exception as exc:
            self._show_error(f"Zoom/resize failed:\n{exc}")
            return

        self.original_label.setPixmap(
            QPixmap.fromImage(numpy_to_qimage(disp_orig))
        )
        self.processed_label.setPixmap(
            QPixmap.fromImage(numpy_to_qimage(disp_curr))
        )

    def _show_error(self, message: str) -> None:
        QMessageBox.warning(self, "Error", message)