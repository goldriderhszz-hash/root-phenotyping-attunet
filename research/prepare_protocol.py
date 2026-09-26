from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("ROOTSCOPE_DATA_DIR", str(ROOT / "data"))).expanduser().resolve()
SEED = 20260920


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_unchanged(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Cannot read {path}")
    return image


def mask_from_labelme(payload: dict, width: int, height: int) -> np.ndarray:
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for index, shape in enumerate(payload.get("shapes", [])):
        if shape.get("shape_type", "polygon") != "polygon":
            raise ValueError(f"shape {index} is not a polygon")
        points = [(float(x), float(y)) for x, y in shape.get("points", [])]
        if len(points) < 2:
            raise ValueError(f"shape {index} has fewer than 2 points")
        draw.polygon(points, outline=1, fill=1)
    return np.asarray(canvas, dtype=np.uint8)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    images = {p.stem: p for p in DATA_DIR.glob("*.tif") if not p.stem.lower().endswith("_mask")}
    annotations = {p.stem: p for p in DATA_DIR.glob("*.json")}
    missing_json = sorted(set(images) - set(annotations))
    missing_image = sorted(set(annotations) - set(images))
    errors: list[str] = []
    if len(images) != 50 or len(annotations) != 50:
        errors.append(f"expected 50 TIFF and 50 JSON files, found {len(images)} and {len(annotations)}")
    if missing_json or missing_image:
        errors.append(f"unpaired files: missing_json={missing_json}, missing_image={missing_image}")
    if any(name.startswith("10-1-4") for name in images | annotations):
        errors.append("excluded sample 10-1-4 remains in the active dataset")

    rows: list[dict] = []
    image_byte_groups: dict[str, list[str]] = defaultdict(list)
    image_pixel_groups: dict[str, list[str]] = defaultdict(list)
    for name in sorted(set(images) & set(annotations)):
        image_path = images[name]
        json_path = annotations[name]
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        image = read_unchanged(image_path)
        height, width = image.shape[:2]
        if int(payload.get("imageWidth", width)) != width or int(payload.get("imageHeight", height)) != height:
            errors.append(f"{name}: JSON dimensions do not match TIFF")
        if Path(str(payload.get("imagePath", ""))).name != image_path.name:
            errors.append(f"{name}: imagePath={payload.get('imagePath')!r} does not name {image_path.name}")

        coordinate_count = 0
        for shape_index, shape in enumerate(payload.get("shapes", [])):
            for point_index, point in enumerate(shape.get("points", [])):
                coordinate_count += 1
                x, y = map(float, point)
                # Labelme coordinates are continuous and may lie exactly on the
                # right/bottom image boundary; rasterization clips that boundary.
                if not (0 <= x <= width and 0 <= y <= height):
                    errors.append(f"{name}: shape {shape_index} point {point_index} ({x}, {y}) outside {width}x{height}")

        regenerated = mask_from_labelme(payload, width, height)
        mask_path = DATA_DIR / f"{name}_mask.png"
        old_mask_dice = float("nan")
        if not mask_path.exists():
            current_mask = np.zeros_like(regenerated)
        else:
            current_mask = (read_unchanged(mask_path) > 0).astype(np.uint8)
            if current_mask.shape == regenerated.shape:
                denominator = int(current_mask.sum() + regenerated.sum())
                old_mask_dice = float(2 * np.logical_and(current_mask, regenerated).sum() / denominator) if denominator else 1.0
            backup_dir = ROOT / "mask_cache_before_regeneration"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / mask_path.name
            if not backup_path.exists():
                shutil.copy2(mask_path, backup_path)
        Image.fromarray((regenerated * 255).astype(np.uint8), mode="L").save(mask_path)

        image_byte_hash = sha256_file(image_path)
        image_pixel_hash = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
        image_byte_groups[image_byte_hash].append(name)
        image_pixel_groups[image_pixel_hash].append(name)
        rows.append(
            {
                "sample_name": name,
                "image_file": image_path.name,
                "annotation_file": json_path.name,
                "mask_file": mask_path.name,
                "image_sha256": image_byte_hash,
                "annotation_sha256": sha256_file(json_path),
                "image_pixel_sha256": image_pixel_hash,
                "mask_binary_sha256": hashlib.sha256(regenerated.tobytes()).hexdigest(),
                "width": width,
                "height": height,
                "annotation_count": len(payload.get("shapes", [])),
                "annotation_point_count": coordinate_count,
                "foreground_pixels": int(regenerated.sum()),
                "foreground_ratio": float(regenerated.mean()),
                "previous_mask_dice_vs_json": old_mask_dice,
                "included": True,
                "exclusion_reason": "",
            }
        )

    byte_duplicates = [members for members in image_byte_groups.values() if len(members) > 1]
    pixel_duplicates = [members for members in image_pixel_groups.values() if len(members) > 1]
    if byte_duplicates:
        errors.append(f"byte-identical TIFF groups: {byte_duplicates}")
    if pixel_duplicates:
        errors.append(f"pixel-identical TIFF groups: {pixel_duplicates}")
    if errors:
        raise RuntimeError("Dataset audit failed:\n- " + "\n- ".join(errors))

    write_csv(ROOT / "dataset_manifest.csv", rows)
    write_csv(
        ROOT / "manifests" / "mask_regeneration_audit.csv",
        [
            {
                "sample_name": row["sample_name"],
                "previous_mask_dice_vs_json": row["previous_mask_dice_vs_json"],
                "regenerated_mask_sha256": row["mask_binary_sha256"],
            }
            for row in rows
        ],
    )

    # Rank-block stratification: every consecutive block of five foreground ratios
    # contributes one sample to each fold; the within-block allocation is seeded.
    ordered = sorted(rows, key=lambda row: (float(row["foreground_ratio"]), row["sample_name"]))
    rng = random.Random(SEED)
    folds: dict[int, list[str]] = {fold: [] for fold in range(5)}
    sample_to_fold: dict[str, int] = {}
    for start in range(0, len(ordered), 5):
        block = ordered[start : start + 5]
        labels = list(range(5))
        rng.shuffle(labels)
        for row, fold in zip(block, labels):
            name = str(row["sample_name"])
            folds[fold].append(name)
            sample_to_fold[name] = fold
    for names in folds.values():
        names.sort()

    assignment = {
        "protocol_name": "internal_5_fold_rank_block_stratified_cross_validation",
        "split_seed": SEED,
        "stratification_variable": "annotation foreground_ratio",
        "stratification_method": "sort by foreground_ratio, partition into ten consecutive blocks of five, and assign one seeded-random sample per block to each fold",
        "rotation_rule": "outer fold k is test; fold (k+1) mod 5 is validation; the remaining folds are training",
        "fold_sizes": {str(fold): len(names) for fold, names in folds.items()},
        "folds": {str(fold): names for fold, names in folds.items()},
        "sample_to_fold": sample_to_fold,
        "dataset_manifest_sha256": sha256_file(ROOT / "dataset_manifest.csv"),
    }
    (ROOT / "fold_assignments.json").write_text(json.dumps(assignment, ensure_ascii=False, indent=2), encoding="utf-8")

    run_config = {
        "dataset_n": 50,
        "folds": 5,
        "split_seed": SEED,
        "seeds": [42, 3407, 2026],
        "epochs": 50,
        "checkpoint_selection": "highest validation macro Dice at threshold 0.50, evaluated after every epoch",
        "threshold_selection": {"split": "validation only", "minimum": 0.10, "maximum": 0.90, "step": 0.01, "tie_break": "lower threshold"},
        "test_policy": "each held-out test fold is evaluated once after checkpoint and threshold lock",
        "optimizer": {"name": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001},
        "scheduler": {"name": "CosineAnnealingLR", "T_max": 50, "eta_min": 0.00001},
        "batch_size": 12,
        "patch_size": 256,
        "training_stride": 128,
        "inference_stride": 128,
        "fusion": "Gaussian-weighted overlap",
        "configs": {
            "unet_pixel": "U-Net; 0.2 BCE + 0.8 Dice",
            "attunet_pixel": "Attention U-Net; 0.2 BCE + 0.8 Dice",
            "attunet_cl_fixed": "Attention U-Net; 0.2 BCE + 0.8 Dice + 0.3 corrected soft-clDice for epochs 1-50",
            "attunet_cl_schedule": "Attention U-Net; fixed loss for epochs 1-25; BCE weight changes from 0.2 to 0.4 for epochs 26-50",
        },
    }
    (ROOT / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "status": "PASS",
        "n_pairs": len(rows),
        "byte_duplicate_groups": byte_duplicates,
        "pixel_duplicate_groups": pixel_duplicates,
        "fold_sizes": assignment["fold_sizes"],
        "foreground_ratio_range": [min(float(r["foreground_ratio"]) for r in rows), max(float(r["foreground_ratio"]) for r in rows)],
    }
    (ROOT / "manifests" / "protocol_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
