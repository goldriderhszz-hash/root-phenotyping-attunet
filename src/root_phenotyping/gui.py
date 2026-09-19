"""PyQt desktop interface for segmentation and mask-derived measurements."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import onnxruntime as ort
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from skimage.morphology import skeletonize

from .onnx_inference import read_grayscale, segment_image
from .phenotypes import extract_phenotypes_decoupled

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
MODEL_NAME = "precision_boost_attention_unet.onnx"


def default_model_path() -> Path:
    """Find the model in a source checkout or a PyInstaller bundle."""
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
    candidates = (
        bundle_root / "models" / MODEL_NAME,
        bundle_root / "software" / "models" / MODEL_NAME,
        Path.cwd() / "software" / "models" / MODEL_NAME,
    )
    return next((path for path in candidates if path.exists()), candidates[0])


def to_pixmap(image: np.ndarray) -> QPixmap:
    """Convert a NumPy grayscale or BGR image without retaining its buffer."""
    if image.ndim == 2:
        height, width = image.shape
        qimage = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


class RootPhenotyperWindow(QMainWindow):
    """Single-image review interface reconstructed from the archived executable."""

    def __init__(self, model_path: Path):
        super().__init__()
        self.model_path = model_path
        self.session = None
        self.image_path: Path | None = None
        self.original_image: np.ndarray | None = None
        self._build_ui()
        self._load_model()

    def _build_ui(self) -> None:
        self.setWindowTitle("AI Root Phenotype Tool (reconstructed, manuscript edition)")
        self.resize(1280, 820)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        controls = QGroupBox("Analysis")
        row = QHBoxLayout(controls)
        self.load_button = QPushButton("1. Load root image")
        self.run_button = QPushButton("2. Segment and measure")
        self.run_button.setEnabled(False)
        self.load_button.clicked.connect(self.load_image)
        self.run_button.clicked.connect(self.run_analysis)
        row.addWidget(self.load_button)
        row.addWidget(self.run_button)
        row.addStretch()
        layout.addWidget(controls)

        images = QGroupBox("Original image | predicted mask | skeleton")
        image_row = QHBoxLayout(images)
        self.image_labels = [QLabel(text) for text in ("Original", "Mask", "Skeleton")]
        for label in self.image_labels:
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(320, 360)
            label.setStyleSheet("background:#f2f2f2; border:1px solid #bbb;")
            image_row.addWidget(label)
        layout.addWidget(images, stretch=2)

        report = QGroupBox("Image-derived descriptor report")
        report_layout = QVBoxLayout(report)
        self.report = QTextBrowser()
        report_layout.addWidget(self.report)
        layout.addWidget(report, stretch=1)

    def _load_model(self) -> None:
        try:
            self.session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
            self.report.setHtml(
                f"<b>Ready.</b> Loaded the archived ONNX model:<br>{self.model_path}"
            )
        except Exception as exc:
            self.report.setHtml(
                "<b style='color:#b00020'>Model could not be loaded.</b><br>"
                f"{exc}<br><br>Expected: {self.model_path}"
            )

    def _show(self, image: np.ndarray, label: QLabel) -> None:
        label.setPixmap(
            to_pixmap(image).scaled(
                label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def load_image(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select a root image",
            "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)",
        )
        if not filename:
            return
        try:
            self.original_image = read_grayscale(filename)
            self.image_path = Path(filename)
            self._show(self.original_image, self.image_labels[0])
            self.image_labels[1].setText("Mask")
            self.image_labels[2].setText("Skeleton")
            self.run_button.setEnabled(self.session is not None)
            self.report.setHtml(f"Loaded: <b>{self.image_path.name}</b>")
        except Exception as exc:
            self.report.setHtml(f"<b style='color:#b00020'>Image error:</b> {exc}")

    def run_analysis(self) -> None:
        if self.session is None or self.original_image is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.run_button.setEnabled(False)
        try:
            _, mask = segment_image(self.session, self.original_image, threshold=0.53)
            length, segments, junctions, angle = extract_phenotypes_decoupled(mask)
            skeleton = skeletonize(mask > 127).astype(np.uint8) * 255
            self._show(mask, self.image_labels[1])
            self._show(skeleton, self.image_labels[2])
            self.report.setHtml(
                "<h3>Results</h3>"
                f"<p><b>Dominant-axis path index:</b> {length:.2f} px-equivalent</p>"
                f"<p><b>Retained segment count:</b> {segments}</p>"
                f"<p><b>Junction-region count:</b> {junctions}</p>"
                f"<p><b>Mean local acute angle:</b> {angle:.2f}&deg;</p>"
                "<p><i>These are two-dimensional image-derived descriptors. "
                "No physical-length calibration or root-identity tracking is applied.</i></p>"
            )
        except Exception as exc:
            self.report.setHtml(f"<b style='color:#b00020'>Analysis error:</b> {exc}")
        finally:
            self.run_button.setEnabled(True)
            QApplication.restoreOverrideCursor()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().resizeEvent(event)
        if self.original_image is not None:
            self._show(self.original_image, self.image_labels[0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the AI root phenotype desktop tool.")
    parser.add_argument("--model", type=Path, default=default_model_path())
    args = parser.parse_args(argv)
    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    window = RootPhenotyperWindow(args.model.resolve())
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
