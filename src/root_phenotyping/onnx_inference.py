"""ONNX inference used by the reconstructed desktop application."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def gaussian_window(patch_size: int = 256, sigma: float = 64.0) -> np.ndarray:
    """Return the Gaussian blending window used in the manuscript pipeline."""
    x = np.arange(patch_size)
    mean = patch_size // 2
    gauss_1d = np.exp(-((x - mean) ** 2) / (2 * sigma**2))
    window = np.outer(gauss_1d, gauss_1d)
    window = (window - window.min()) / (window.max() - window.min() + 1e-8)
    return (window * 0.9 + 0.1).astype(np.float32)


def sliding_window_predict(
    session,
    image: np.ndarray,
    *,
    patch_size: int = 256,
    overlap_ratio: float = 0.5,
) -> np.ndarray:
    """Predict a padded grayscale image with Gaussian-weighted overlap."""
    if image.ndim != 2:
        raise ValueError("Expected a two-dimensional grayscale image.")
    if image.shape[0] < patch_size or image.shape[1] < patch_size:
        raise ValueError("Image must be padded to at least one model patch.")
    if not 0 <= overlap_ratio < 1:
        raise ValueError("overlap_ratio must be in [0, 1).")

    height, width = image.shape
    stride = max(1, int(patch_size * (1 - overlap_ratio)))
    prediction = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    window = gaussian_window(patch_size)

    y_coords = list(range(0, height - patch_size + 1, stride))
    x_coords = list(range(0, width - patch_size + 1, stride))
    if y_coords[-1] != height - patch_size:
        y_coords.append(height - patch_size)
    if x_coords[-1] != width - patch_size:
        x_coords.append(width - patch_size)

    input_name = session.get_inputs()[0].name
    for y in y_coords:
        for x in x_coords:
            patch = image[y : y + patch_size, x : x + patch_size]
            tensor = patch[np.newaxis, np.newaxis].astype(np.float32, copy=False)
            patch_prediction = np.squeeze(session.run(None, {input_name: tensor})[0])
            prediction[y : y + patch_size, x : x + patch_size] += patch_prediction * window
            weights[y : y + patch_size, x : x + patch_size] += window

    return prediction / (weights + 1e-8)


def segment_image(
    session,
    image: np.ndarray,
    *,
    threshold: float = 0.53,
    patch_size: int = 256,
    overlap_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess, infer, crop, and threshold one grayscale root image."""
    if image is None or image.ndim != 2:
        raise ValueError("A readable grayscale image is required.")
    height, width = image.shape
    target_height = max(patch_size, height + (-height % patch_size))
    target_width = max(patch_size, width + (-width % patch_size))
    pad_h = target_height - height
    pad_w = target_width - width
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    prepared = clahe.apply(padded).astype(np.float32) / 255.0
    probability = sliding_window_predict(
        session,
        prepared,
        patch_size=patch_size,
        overlap_ratio=overlap_ratio,
    )[:height, :width]
    mask = (probability > threshold).astype(np.uint8) * 255
    return probability, mask


def read_grayscale(path: str | Path) -> np.ndarray:
    """Read paths containing non-ASCII characters on Windows."""
    encoded = np.fromfile(Path(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image
