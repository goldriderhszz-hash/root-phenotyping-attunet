"""Batch workflow shared by the desktop interface and command-line entry point."""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import sys
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import onnxruntime as ort
import scipy
import skimage

from . import __version__
from .engine import InferenceEngine, analyze
from .models import DEFAULT_MODEL_ID, model_path, model_spec

MODEL_PATH = model_path(DEFAULT_MODEL_ID)
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
Progress = Callable[[int, int, str], None]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def finite(value):
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items() if not key.startswith("_")}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def match_reference(image: Path, references: list[Path]) -> Path | None:
    stem = image.stem.casefold()
    for path in references:
        if path.stem.casefold() in {stem, stem + "_mask", stem + "-mask"}:
            return path
    return None


def row_for(result: dict) -> dict:
    row = {key: result.get(key) for key in ("image_name", "image_sha256", "width", "height", "foreground_fraction", "threshold")}
    row.update(result["descriptors"])
    row["qc_flags"] = ";".join(result["qc_flags"])
    row["reference_sha256"] = result.get("reference_sha256")
    for key, value in result.get("evaluation", {}).items():
        row["eval_" + key] = value
    for key in ("dominant_path_length", "retained_segment_count", "junction_region_count", "mean_local_acute_angle_deg"):
        if "reference_descriptors" in result:
            reference_value = result["reference_descriptors"].get(key)
            prediction_value = result["descriptors"].get(key)
            row["reference_" + key] = reference_value
            row["error_" + key] = prediction_value - reference_value if prediction_value is not None and reference_value is not None else None
    return finite(row)


def write_csv(path: Path, rows: list[dict], fallback_fields: list[str]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else fallback_fields
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_batch(images: list[Path], references: list[Path], output_root: Path,
              threshold: float | None = None, save_probability: bool = False,
              cancel_event: threading.Event | None = None, progress: Progress | None = None,
              make_zip: bool = True, model_id: str = DEFAULT_MODEL_ID) -> dict:
    if not images:
        raise ValueError("请先导入至少一张根系图片")
    spec = model_spec(model_id)
    if threshold is None:
        threshold = float(spec["threshold"])
    if not 0.1 <= threshold <= 0.9:
        raise ValueError("阈值必须在 0.10 到 0.90 之间")
    for image in images + references:
        if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"图片不存在或格式不支持：{image}")
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    folder = output_root / ("RootScope_" + run_id)
    folder.mkdir()
    (folder / "images").mkdir()
    started = utc_now()
    engine = InferenceEngine(model_path(model_id), spec["sha256"])
    results: list[dict] = []
    errors: list[dict] = []
    stopped = False
    processed = 0
    for index, image in enumerate(images, 1):
        if cancel_event is not None and cancel_event.is_set():
            stopped = True
            break
        if progress:
            progress(index - 1, len(images), f"正在分析：{image.name}")
        image_id = uuid.uuid4().hex[:10]
        reference = match_reference(image, references)
        try:
            result = analyze(image, folder / "images" / image_id, engine, threshold,
                             reference, save_probability)
            result["image_id"] = image_id
            result["_source_path"] = str(image)
            results.append(finite(result) | {"_source_path": str(image)})
        except Exception as exc:
            errors.append({"image_name": image.name, "reason": f"{type(exc).__name__}: {exc}"})
        processed = index
        if progress:
            progress(index, len(images), f"已完成 {index}/{len(images)} 张")
    if progress:
        progress(processed, len(images), "正在保存结果与来源记录…")
    public_results = [finite(item) for item in results]
    rows = [row_for(item) for item in results]
    write_csv(folder / "results.csv", rows, ["image_name", "dominant_path_length", "retained_segment_count",
                                                 "junction_region_count", "mean_local_acute_angle_deg", "qc_flags"])
    write_csv(folder / "errors.csv", errors, ["image_name", "reason"])
    (folder / "results.json").write_text(json.dumps(public_results, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "application": f"RootScope Desktop {__version__}", "created_utc": started, "completed_utc": utc_now(),
        "status": "cancelled" if stopped else "complete", "requested_images": len(images),
        "successful_images": len(results), "failed_images": len(errors), "errors": errors,
        "model": "Attention U-Net + fixed soft-clDice", "model_id": model_id,
        "study_seed": 42, "study_fold": spec["fold"],
        "model_format": "ONNX opset 17", "onnx_sha256": spec["sha256"],
        "source_checkpoint_sha256": spec["source_checkpoint_sha256"],
        "default_threshold": spec["threshold"], "used_threshold": threshold,
        "image_preprocessing": "grayscale; CLAHE clip=2.0 grid=16x16; intensity / 255",
        "inference": "256x256 tiles; stride 128; reflect pad; Gaussian sigma 64; fused probabilities",
        "descriptors": "raw skeleton weighted dominant path; 5x5 closing; 15 pruning; >=10 segment pixels",
        "measurement_scale": "pixel-space only; no physical calibration",
        "save_probability": bool(save_probability), "provider": engine.device,
        "software": {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
                     "opencv": cv2.__version__, "scipy": scipy.__version__, "scikit_image": skimage.__version__,
                     "onnxruntime": ort.__version__},
        "images": [{"name": item["image_name"], "sha256": item["image_sha256"],
                    "reference_sha256": item.get("reference_sha256")} for item in public_results],
    }
    (folder / "provenance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = None
    if make_zip:
        if progress:
            progress(processed, len(images), "正在生成结果压缩包…")
        archive = folder / ("RootScope_" + run_id + "_results.zip")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
            for path in folder.rglob("*"):
                if path.is_file() and path != archive:
                    output.write(path, path.relative_to(folder))
    return {"folder": str(folder), "archive": str(archive) if archive else None,
            "results": results, "errors": errors, "report": report}
