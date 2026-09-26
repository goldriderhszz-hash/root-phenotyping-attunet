"""Deterministic inference, image descriptors, and optional reference evaluation."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from .phenotype_extractor import ExtractorSettings, dominant_path, extract_descriptors, prune_skeleton

MODEL_SHA256 = "f81770d5c8e2e710da1b30f3354fef4e69d4641079b291fd7e857d4c647b8892"
DEFAULT_THRESHOLD = 0.47
PATCH_SIZE = 256
STRIDE = 128


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2:
        raise ValueError(f"无法读取灰度图像：{path.name}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"无法保存图像：{path.name}")
    encoded.tofile(str(path))


def preview(image: np.ndarray, maximum: int = 1400) -> np.ndarray:
    height, width = image.shape[:2]
    if max(height, width) <= maximum:
        return image
    scale = maximum / max(height, width)
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def gaussian_window() -> np.ndarray:
    axis = np.arange(PATCH_SIZE, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    weight = np.exp(-((xx - 128) ** 2 + (yy - 128) ** 2) / (2 * 64.0 ** 2))
    return (0.1 + 0.9 * (weight - weight.min()) / (weight.max() - weight.min())).astype(np.float32)


class InferenceEngine:
    def __init__(self, checkpoint: Path, expected_sha256: str = MODEL_SHA256):
        actual = sha256(checkpoint)
        if actual != expected_sha256:
            raise ValueError("模型校验失败：ONNX 文件 SHA-256 与发布记录不一致")
        options = ort.SessionOptions()
        options.intra_op_num_threads = min(os.cpu_count() or 1, 8)
        available = ort.get_available_providers()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(checkpoint), sess_options=options, providers=providers)
        self.device = self.session.get_providers()[0]
        self.weight = gaussian_window()

    def predict(self, image: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        if batch_size is None:
            batch_size = 8 if self.device == "CUDAExecutionProvider" else 2
        normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16)).apply(image).astype(np.float32) / 255.0
        height, width = normalized.shape
        pad_h, pad_w = (-height) % PATCH_SIZE, (-width) % PATCH_SIZE
        padded = np.pad(normalized, ((0, pad_h), (0, pad_w)), mode="reflect")
        coordinates = [(y, x) for y in range(0, padded.shape[0] - PATCH_SIZE + 1, STRIDE)
                       for x in range(0, padded.shape[1] - PATCH_SIZE + 1, STRIDE)]
        total = np.zeros_like(padded, dtype=np.float32)
        weights = np.zeros_like(padded, dtype=np.float32)
        for start in range(0, len(coordinates), batch_size):
            positions = coordinates[start:start + batch_size]
            tiles = np.stack([padded[y:y + PATCH_SIZE, x:x + PATCH_SIZE] for y, x in positions])[:, None].astype(np.float32)
            outputs = self.session.run(["probability"], {"image": tiles})[0][:, 0]
            for output, (y, x) in zip(outputs, positions):
                total[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += output * self.weight
                weights[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += self.weight
        return (total / (weights + 1e-8))[:height, :width]


def evaluate(prediction: np.ndarray, reference: np.ndarray) -> dict:
    pred, ref = prediction.astype(bool), reference.astype(bool)
    tp = int((pred & ref).sum())
    fp = int((pred & ~ref).sum())
    fn = int((~pred & ref).sum())
    def ratio(n, d):
        return float(n / d) if d else None
    sk_pred, sk_ref = skeletonize(pred), skeletonize(ref)
    tprec = ratio((sk_pred & ref).sum(), sk_pred.sum())
    tsens = ratio((sk_ref & pred).sum(), sk_ref.sum())
    cldice = ratio(2 * tprec * tsens, tprec + tsens) if tprec is not None and tsens is not None else None
    metrics = {"precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
               "dice": ratio(2 * tp, 2 * tp + fp + fn), "iou": ratio(tp, tp + fp + fn),
               "hard_cldice": cldice, "tp": tp, "fp": fp, "fn": fn}
    width = 2 * distance_transform_edt(ref)
    for key, lower, upper in (("width_le3", 0, 3), ("width_3_6", 3, 6),
                              ("width_6_12", 6, 12), ("width_gt12", 12, math.inf)):
        support = sk_ref & (width > lower) & (width <= upper)
        n = int(support.sum())
        metrics[key + "_n"] = n
        metrics[key + "_recall"] = ratio((support & pred).sum(), n)
    return metrics


def analyze(image_path: Path, output: Path, engine: InferenceEngine, threshold: float,
            reference_path: Path | None = None, save_probability: bool = False) -> dict:
    image = read_gray(image_path)
    reference = read_gray(reference_path) > 127 if reference_path is not None else None
    if reference is not None and reference.shape != image.shape:
        raise ValueError("参考掩膜与原图尺寸不一致")
    output.mkdir(parents=True, exist_ok=True)
    probability = engine.predict(image)
    mask = probability > threshold
    descriptors = extract_descriptors(mask, ExtractorSettings())
    raw = skeletonize(mask).astype(np.uint8)
    path, _, _, _ = dominant_path(raw)
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    pruned = prune_skeleton(skeletonize(closed).astype(np.uint8), 15)
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array([130, 164, 28])).astype(np.uint8)
    if path:
        points = np.asarray(path)
        overlay[points[:, 0], points[:, 1]] = (229, 144, 47)
    skeleton_view = np.zeros((*image.shape, 3), dtype=np.uint8)
    skeleton_view[pruned > 0] = (117, 173, 63)
    if path:
        skeleton_view[points[:, 0], points[:, 1]] = (229, 144, 47)
    write_image(output / "prediction_mask.png", mask.astype(np.uint8) * 255)
    write_image(output / "overlay.png", overlay)
    write_image(output / "skeleton.png", skeleton_view)
    write_image(output / "source_preview.jpg", preview(image))
    write_image(output / "mask_preview.png", preview(mask.astype(np.uint8) * 255))
    write_image(output / "overlay_preview.jpg", preview(overlay))
    write_image(output / "skeleton_preview.png", preview(skeleton_view))
    if save_probability:
        np.savez_compressed(output / "probability_float32.npz", probability=probability)
    result = {"image_name": image_path.name, "image_sha256": sha256(image_path),
              "width": int(image.shape[1]), "height": int(image.shape[0]),
              "foreground_fraction": float(mask.mean()), "threshold": threshold,
              "descriptors": descriptors, "qc_flags": []}
    if not mask.any(): result["qc_flags"].append("empty_prediction")
    if mask.mean() < 0.001: result["qc_flags"].append("very_low_foreground")
    if mask.mean() > 0.2: result["qc_flags"].append("high_foreground")
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        result["qc_flags"].append("touches_image_border")
    if descriptors["path_status"] != "ok": result["qc_flags"].append("path_unavailable")
    if descriptors["valid_angle_n"] == 0: result["qc_flags"].append("angle_unavailable")
    if reference is not None:
        result["reference_sha256"] = sha256(reference_path)
        result["evaluation"] = evaluate(mask, reference)
        result["reference_descriptors"] = extract_descriptors(reference, ExtractorSettings())
    return result
