"""36-setting extractor sensitivity on annotations and fixed-loss OOF masks."""
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

from compute_oof_descriptors import (DATA, DESCRIPTOR_FIELDS, OUT, ROOT, SEEDS,
                                     TRAITS, lin_ccc, read_csv, write_csv)
from grid_extractor import extract_grid

PARTS = OUT / "sensitivity_parts"
DEFAULT = (5, 15, 10)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grid_for(sample: str, fold: int, source: str, seed: int | str, path: Path,
             shape: tuple[int, int]) -> list[dict]:
    path_hash = sha256(path)
    part = PARTS / source / f"seed_{seed}" / f"{sample}.json"
    if part.exists():
        candidate = json.loads(part.read_text(encoding="utf-8"))
        if (candidate.get("source_sha256") == path_hash
                and candidate.get("extractor") == "grid_equivalent_v1"
                and len(candidate.get("rows", [])) == 36):
            return candidate["rows"]
    mask = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != shape:
        raise RuntimeError(f"Invalid mask or shape at {path}")
    rows = [{"sample_name": sample, "fold": fold, "source": source,
             "config": "annotation" if source == "annotation_mask" else "attunet_cl_fixed",
             "seed": seed, "mask_path": str(path), "source_sha256": path_hash,
             "extractor": "grid_equivalent_v1", **entry}
            for entry in extract_grid(mask)]
    lengths = [row["dominant_path_length"] for row in rows]
    if not all((math.isnan(lengths[0]) and math.isnan(value)) or lengths[0] == value for value in lengths):
        raise RuntimeError(f"Path length changed across Channel B settings: {path}")
    part.parent.mkdir(parents=True, exist_ok=True)
    temporary = part.with_name(part.name + ".tmp")
    temporary.write_text(json.dumps({"source_sha256": path_hash, "extractor": "grid_equivalent_v1",
                                     "rows": rows}, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    temporary.replace(part)
    return rows


def setting_key(row: dict) -> tuple[int, int, int]:
    return (int(row["closing_kernel"]), int(row["pruning_iterations"]),
            int(row["min_component_pixels"]))


def summary(pairs: list[dict], seed: int, setting: tuple[int, int, int], trait: str) -> dict:
    selected = [row for row in pairs if row["seed"] == seed and row["trait"] == trait
                and setting_key(row) == setting]
    if len(selected) != 50:
        raise RuntimeError(f"Expected 50 pairs for {seed}, {setting}, {trait}; got {len(selected)}")
    valid = [row for row in selected if row["status"] == "valid"]
    x = np.asarray([row["reference_value"] for row in valid], dtype=float)
    y = np.asarray([row["estimate_value"] for row in valid], dtype=float)
    d = y - x
    nonzero = x != 0
    return {"seed": seed, "closing_kernel": setting[0], "pruning_iterations": setting[1],
            "min_component_pixels": setting[2], "trait": trait,
            "reference_type": "annotation_mask_same_extractor", "n_distinct_images": 50,
            "n_valid_pairs": len(valid), "n_invalid_pairs": 50 - len(valid),
            "n_mape": int(np.count_nonzero(nonzero)), "n_zero_reference": int(np.count_nonzero(~nonzero)),
            "mae": float(np.mean(np.abs(d))) if len(valid) else float("nan"),
            "bias_estimate_minus_reference": float(np.mean(d)) if len(valid) else float("nan"),
            "ccc": lin_ccc(x, y),
            "mape_percent_nonzero_reference": float(np.mean(np.abs(d[nonzero] / x[nonzero])) * 100) if np.any(nonzero) else float("nan"),
            "reference_valid_angle_n": sum(row["reference_valid_angle_n"] > 0 for row in selected),
            "prediction_valid_angle_n": sum(row["prediction_valid_angle_n"] > 0 for row in selected),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("annotation", "all"), default="all")
    args = parser.parse_args()
    manifest = [row for row in read_csv(ROOT / "dataset_manifest.csv") if row["included"].lower() == "true"]
    if len(manifest) != 50:
        raise RuntimeError("Manifest does not have exactly 50 included images")
    assignments = json.loads((ROOT / "fold_assignments.json").read_text(encoding="utf-8"))["sample_to_fold"]
    annotation_rows = []
    for i, row in enumerate(manifest, 1):
        name = row["sample_name"]
        annotation_rows.extend(grid_for(name, int(assignments[name]), "annotation_mask", "reference",
                                        DATA / row["mask_file"], (int(row["height"]), int(row["width"]))))
        if i % 10 == 0:
            print(f"annotation grid {i}/50", flush=True)
    reference = {(row["sample_name"], setting_key(row)): row for row in annotation_rows}
    if len(reference) != 1800:
        raise RuntimeError(f"Annotation grid key count is {len(reference)} instead of 1800")
    settings = sorted(set(setting_key(row) for row in annotation_rows))
    annotation_summary = []
    for setting in settings:
        selected = [row for row in annotation_rows if setting_key(row) == setting]
        result = {"closing_kernel": setting[0], "pruning_iterations": setting[1],
                  "min_component_pixels": setting[2], "n_distinct_images": len(selected),
                  "n_valid_angle_images": sum(int(row["valid_angle_n"]) > 0 for row in selected),
                  "default_parameter_setting": setting == DEFAULT}
        for trait in TRAITS:
            values = np.asarray([row[trait] for row in selected], dtype=float)
            result[f"{trait}_mean"] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
            result[f"{trait}_sd"] = float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else float("nan")
        annotation_summary.append(result)
    default_annotation = next(row for row in annotation_summary if setting_key(row) == DEFAULT)
    for row in annotation_summary:
        for trait in TRAITS:
            row[f"{trait}_mean_delta_from_5_15_10"] = row[f"{trait}_mean"] - default_annotation[f"{trait}_mean"]
        row["n_valid_angle_images_delta_from_5_15_10"] = row["n_valid_angle_images"] - default_annotation["n_valid_angle_images"]
    write_csv(OUT / "sensitivity_annotation_summary.csv", annotation_summary, list(annotation_summary[0]))

    predictions = []
    if args.phase == "all":
        for seed in SEEDS:
            for i, row in enumerate(manifest, 1):
                name = row["sample_name"]
                path = ROOT / "oof_predictions" / "attunet_cl_fixed" / f"seed_{seed}" / f"{name}.png"
                predictions.extend(grid_for(name, int(assignments[name]), "oof_prediction", seed,
                                            path, (int(row["height"]), int(row["width"]))))
                if i % 10 == 0:
                    print(f"prediction grid seed={seed} {i}/50", flush=True)

    base_fields = ["sample_name", "fold", "source", "config", "seed", "mask_path", "source_sha256", "extractor", *DESCRIPTOR_FIELDS]
    write_csv(OUT / "sensitivity_annotation_grid.csv", annotation_rows, base_fields)
    if args.phase == "annotation":
        print(json.dumps({"annotation_grid_rows": len(annotation_rows), "prediction_grid_rows": 0}), flush=True)
        return
    write_csv(OUT / "sensitivity_fixed_oof_grid.csv", predictions, base_fields)

    pairs = []
    for pred in predictions:
        ref = reference[(pred["sample_name"], setting_key(pred))]
        for trait in TRAITS:
            x = float(ref[trait])
            y = float(pred[trait])
            valid = math.isfinite(x) and math.isfinite(y)
            difference = y - x if valid else float("nan")
            pairs.append({
                "sample_name": pred["sample_name"], "fold": pred["fold"], "seed": pred["seed"],
                "closing_kernel": pred["closing_kernel"], "pruning_iterations": pred["pruning_iterations"],
                "min_component_pixels": pred["min_component_pixels"], "trait": trait,
                "reference_value": x, "estimate_value": y, "difference_estimate_minus_reference": difference,
                "absolute_error": abs(difference) if valid else float("nan"),
                "ape_percent_nonzero_reference": abs(difference / x) * 100 if valid and x != 0 else float("nan"),
                "status": "valid" if valid else "missing_reference_or_estimate",
                "reference_valid_angle_n": int(ref["valid_angle_n"]),
                "prediction_valid_angle_n": int(pred["valid_angle_n"]),
                "reference_angle_failure_reason": ref["angle_failure_reason"],
                "prediction_angle_failure_reason": pred["angle_failure_reason"],
                "reference_mask_sha256": ref["source_sha256"],
                "prediction_mask_sha256": pred["source_sha256"],
            })
    write_csv(OUT / "sensitivity_comparison_long.csv", pairs, list(pairs[0]))
    by_seed = [summary(pairs, seed, setting, trait)
               for seed in SEEDS for setting in settings for trait in TRAITS]
    defaults = {(row["seed"], row["trait"]): row for row in by_seed if setting_key(row) == DEFAULT}
    for row in by_seed:
        baseline = defaults[(row["seed"], row["trait"])]
        row["delta_mae_from_5_15_10"] = row["mae"] - baseline["mae"]
        row["delta_valid_pairs_from_5_15_10"] = row["n_valid_pairs"] - baseline["n_valid_pairs"]
    write_csv(OUT / "sensitivity_summary_by_seed.csv", by_seed, list(by_seed[0]))

    across = []
    for setting in settings:
        for trait in TRAITS:
            selected = [row for row in by_seed if setting_key(row) == setting and row["trait"] == trait]
            result = {"closing_kernel": setting[0], "pruning_iterations": setting[1],
                      "min_component_pixels": setting[2], "trait": trait,
                      "reference_type": "annotation_mask_same_extractor", "n_distinct_images": 50,
                      "n_training_seeds": 3, "default_parameter_setting": setting == DEFAULT,
                      "n_valid_pairs_min_seed": min(row["n_valid_pairs"] for row in selected),
                      "n_valid_pairs_max_seed": max(row["n_valid_pairs"] for row in selected)}
            for field in ("mae", "bias_estimate_minus_reference", "ccc",
                          "mape_percent_nonzero_reference", "reference_valid_angle_n",
                          "prediction_valid_angle_n", "delta_mae_from_5_15_10"):
                values = np.asarray([row[field] for row in selected], dtype=float)
                result[f"{field}_seed_mean"] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
                result[f"{field}_seed_sd"] = float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else float("nan")
            across.append(result)
    write_csv(OUT / "sensitivity_summary_across_seeds.csv", across, list(across[0]))
    print(json.dumps({"annotation_grid_rows": len(annotation_rows),
                      "prediction_grid_rows": len(predictions), "comparison_rows": len(pairs),
                      "settings": len(settings), "summary_rows": len(across)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
