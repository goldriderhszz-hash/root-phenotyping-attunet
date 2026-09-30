"""Recompute mask-derived descriptors for the frozen 5-fold OOF predictions.

The biological reference in these outputs is the annotation mask processed by
the same extractor; no human trace or physical calibration is inferred.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import os

import cv2
import numpy as np

from fast_extractor import extract_descriptors_fast


ROOT = Path(os.environ.get("ROOTSCOPE_STUDY_DIR", str(Path(__file__).resolve().parents[1] / "research")))
DATA = Path(os.environ.get("ROOTSCOPE_DATA_DIR", str(Path(__file__).resolve().parents[1] / "research" / "data")))
OUT = Path(os.environ.get("ROOTSCOPE_ANALYSIS_OUT", str(Path(__file__).resolve().parent / "tables")))
PARTS = OUT / "descriptor_parts"
CONFIGS = ("attunet_cl_fixed", "attunet_cl_schedule", "attunet_pixel", "unet_pixel")
SEEDS = (42, 3407, 2026)
TRAITS = ("dominant_path_length", "retained_segment_count", "junction_region_count",
          "mean_local_acute_angle_deg")
DESCRIPTOR_FIELDS = ("scale_status", "length_unit", "closing_kernel", "pruning_iterations",
                     "min_component_pixels", "dominant_path_length", "retained_segment_count",
                     "junction_region_count", "mean_local_acute_angle_deg", "valid_angle_n",
                     "angle_failure_reason", "path_status")


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mask(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot decode {path}")
    return image


def make_record(sample: str, fold: int, source: str, config: str, seed: int | str,
                path: Path, expected_shape: tuple[int, int]) -> dict:
    path_hash = sha256(path)
    part = PARTS / source / config / f"seed_{seed}" / f"{sample}.json"
    if part.exists():
        candidate = json.loads(part.read_text(encoding="utf-8"))
        if candidate.get("source_sha256") == path_hash and candidate.get("extractor") == "fast_equivalent_v1":
            return candidate
    mask = load_mask(path)
    if mask.shape != expected_shape:
        raise RuntimeError(f"Shape mismatch for {path}: {mask.shape} != {expected_shape}")
    record = {
        "sample_name": sample, "fold": fold, "source": source, "config": config, "seed": seed,
        "mask_path": str(path), "source_sha256": path_hash, "extractor": "fast_equivalent_v1",
        **extract_descriptors_fast(mask),
    }
    part.parent.mkdir(parents=True, exist_ok=True)
    temporary = part.with_name(part.name + ".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    temporary.replace(part)
    return record


def is_finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def lin_ccc(reference: np.ndarray, estimate: np.ndarray) -> float:
    if len(reference) < 2:
        return float("nan")
    mx = float(np.mean(reference))
    my = float(np.mean(estimate))
    vx = float(np.mean((reference - mx) ** 2))
    vy = float(np.mean((estimate - my) ** 2))
    covariance = float(np.mean((reference - mx) * (estimate - my)))
    denominator = vx + vy + (mx - my) ** 2
    return 2 * covariance / denominator if denominator > 0 else float("nan")


def summarize(pairs: list[dict], config: str, seed: int | str, fold: int | str,
              trait: str) -> dict:
    selected = [row for row in pairs if row["config"] == config and row["trait"] == trait
                and (seed == "all" or row["seed"] == seed)
                and (fold == "all" or row["fold"] == fold)]
    valid = [row for row in selected if row["status"] == "valid"]
    ref = np.asarray([row["reference_value"] for row in valid], dtype=float)
    est = np.asarray([row["estimate_value"] for row in valid], dtype=float)
    diff = est - ref
    nonzero = ref != 0
    return {
        "config": config, "seed": seed, "fold": fold, "trait": trait,
        "reference_type": "annotation_mask_same_extractor", "unit": "pixel" if trait == "dominant_path_length" else "degree" if trait == "mean_local_acute_angle_deg" else "count",
        "n_predictions": len(selected), "n_valid_pairs": len(valid), "n_invalid_pairs": len(selected) - len(valid),
        "n_zero_reference": int(np.count_nonzero(~nonzero)), "n_mape": int(np.count_nonzero(nonzero)),
        "mae": float(np.mean(np.abs(diff))) if len(valid) else float("nan"),
        "bias_estimate_minus_reference": float(np.mean(diff)) if len(valid) else float("nan"),
        "rmse": float(np.sqrt(np.mean(diff ** 2))) if len(valid) else float("nan"),
        "ccc": lin_ccc(ref, est),
        "mape_percent_nonzero_reference": float(np.mean(np.abs(diff[nonzero] / ref[nonzero])) * 100) if np.any(nonzero) else float("nan"),
        "mean_reference": float(np.mean(ref)) if len(valid) else float("nan"),
        "mean_estimate": float(np.mean(est)) if len(valid) else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("fixed", "all"), default="all")
    args = parser.parse_args()
    manifest = read_csv(ROOT / "dataset_manifest.csv")
    included = [row for row in manifest if row["included"].lower() == "true"]
    if len(included) != 50:
        raise RuntimeError(f"Expected 50 included images, got {len(included)}")
    fold_assignment = json.loads((ROOT / "fold_assignments.json").read_text(encoding="utf-8"))["sample_to_fold"]
    reference_rows = []
    for index, row in enumerate(included, 1):
        name = row["sample_name"]
        shape = (int(row["height"]), int(row["width"]))
        reference_rows.append(make_record(name, int(fold_assignment[name]), "annotation_mask",
                                          "annotation", "reference", DATA / row["mask_file"], shape))
        if index % 10 == 0:
            print(f"reference {index}/50", flush=True)
    reference_by_name = {row["sample_name"]: row for row in reference_rows}

    configs = ("attunet_cl_fixed",) if args.phase == "fixed" else CONFIGS
    predictions = []
    for config in configs:
        for seed in SEEDS:
            for row in included:
                name = row["sample_name"]
                path = ROOT / "oof_predictions" / config / f"seed_{seed}" / f"{name}.png"
                if not path.exists():
                    raise FileNotFoundError(path)
                shape = (int(row["height"]), int(row["width"]))
                predictions.append(make_record(name, int(fold_assignment[name]), "oof_prediction",
                                               config, seed, path, shape))
            print(f"predictions {config} seed={seed} 50/50", flush=True)

    pairs = []
    for pred in predictions:
        ref = reference_by_name[pred["sample_name"]]
        for trait in TRAITS:
            a, b = ref[trait], pred[trait]
            valid = is_finite(a) and is_finite(b)
            difference = float(b) - float(a) if valid else float("nan")
            ape = abs(difference / float(a)) * 100 if valid and float(a) != 0 else float("nan")
            pairs.append({
                "sample_name": pred["sample_name"], "fold": pred["fold"], "config": pred["config"],
                "seed": pred["seed"], "trait": trait, "reference_type": "annotation_mask_same_extractor",
                "reference_value": a, "estimate_value": b, "difference_estimate_minus_reference": difference,
                "absolute_error": abs(difference) if valid else float("nan"),
                "ape_percent_nonzero_reference": ape, "status": "valid" if valid else "missing_reference_or_estimate",
                "reference_path_status": ref["path_status"], "prediction_path_status": pred["path_status"],
                "reference_angle_failure_reason": ref["angle_failure_reason"],
                "prediction_angle_failure_reason": pred["angle_failure_reason"],
                "reference_mask_sha256": ref["source_sha256"], "prediction_mask_sha256": pred["source_sha256"],
            })
    summary_by_seed = [summarize(pairs, config, seed, "all", trait)
                       for config in configs for seed in SEEDS for trait in TRAITS]
    summary_by_fold_seed = [summarize(pairs, config, seed, fold, trait)
                            for config in configs for seed in SEEDS for fold in range(5) for trait in TRAITS]
    summary_across_seeds = []
    for config in configs:
        for trait in TRAITS:
            rows = [row for row in summary_by_seed if row["config"] == config and row["trait"] == trait]
            overall = {"config": config, "trait": trait, "reference_type": "annotation_mask_same_extractor",
                       "n_distinct_images": 50, "n_training_seeds": 3,
                       "n_valid_pairs_min_seed": min(row["n_valid_pairs"] for row in rows),
                       "n_valid_pairs_max_seed": max(row["n_valid_pairs"] for row in rows)}
            for field in ("mae", "bias_estimate_minus_reference", "rmse", "ccc",
                          "mape_percent_nonzero_reference", "n_mape"):
                values = np.asarray([row[field] for row in rows], dtype=float)
                overall[f"{field}_seed_mean"] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
                overall[f"{field}_seed_sd"] = float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else float("nan")
            summary_across_seeds.append(overall)

    reference_fields = ["sample_name", "fold", "source", "config", "seed", "mask_path", "source_sha256", "extractor", *DESCRIPTOR_FIELDS]
    pair_fields = list(pairs[0])
    summary_fields = list(summary_by_seed[0])
    overall_fields = list(summary_across_seeds[0])
    write_csv(OUT / "annotation_descriptors.csv", reference_rows, reference_fields)
    write_csv(OUT / "oof_descriptors.csv", predictions, reference_fields)
    write_csv(OUT / "descriptor_comparison_long.csv", pairs, pair_fields)
    write_csv(OUT / "descriptor_summary_by_seed.csv", summary_by_seed, summary_fields)
    write_csv(OUT / "descriptor_summary_by_fold_seed.csv", summary_by_fold_seed, summary_fields)
    write_csv(OUT / "descriptor_summary_across_seeds.csv", summary_across_seeds, overall_fields)
    print(json.dumps({"phase": args.phase, "reference_images": len(reference_rows),
                      "oof_predictions": len(predictions), "pair_rows": len(pairs),
                      "missing_pairs": sum(row["status"] != "valid" for row in pairs)},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
