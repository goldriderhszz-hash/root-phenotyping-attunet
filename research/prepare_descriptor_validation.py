from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from experiment_core import DATA_DIR, ROOT, read_gray
from phenotype_extractor import ExtractorSettings, extract_descriptors


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or list(rows[0])
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_csv_new(path: Path, rows: list[dict], fields: list[str]) -> None:
    if not path.exists():
        write_csv(path, rows, fields)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_two(values: list[dict]) -> list[dict]:
    ordered = sorted(values, key=lambda row: (float(row["foreground_ratio"]), row["sample_name"]))
    ratios = np.asarray([float(row["foreground_ratio"]) for row in ordered])
    targets = np.quantile(ratios, [1 / 3, 2 / 3])
    selected = []
    available = list(range(len(ordered)))
    for target in targets:
        index = min(available, key=lambda item: (abs(ratios[item] - target), ordered[item]["sample_name"]))
        selected.append({**ordered[index], "target_quantile": float(target)})
        available.remove(index)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args()
    manifest = list(csv.DictReader((ROOT / "dataset_manifest.csv").open("r", encoding="utf-8-sig", newline="")))
    assignment = json.loads((ROOT / "fold_assignments.json").read_text(encoding="utf-8"))["sample_to_fold"]
    selected = []
    for fold in range(5):
        fold_rows = [row for row in manifest if int(assignment[row["sample_name"]]) == fold]
        for order, row in enumerate(select_two(fold_rows), 1):
            selected.append({**row, "fold": fold, "fold_quantile_label": "q33" if order == 1 else "q67"})

    randomized = selected.copy()
    random.Random(20260920).shuffle(randomized)
    output = ROOT / "manual_validation"
    observer = output / "observer_packet"
    coordinator = output / "coordinator_only"
    images = observer / "blinded_images"
    images.mkdir(parents=True, exist_ok=True)
    coordinator.mkdir(parents=True, exist_ok=True)
    mapping = []
    for index, row in enumerate(randomized, 1):
        blind = f"RP{index:03d}"
        source = DATA_DIR / f"{row['sample_name']}.tif"
        destination = images / f"{blind}.tif"
        if not destination.exists():
            shutil.copy2(source, destination)
        mapping.append({"blind_id": blind, "sample_name": row["sample_name"], "fold": row["fold"], "fold_quantile_label": row["fold_quantile_label"], "foreground_ratio": row["foreground_ratio"], "source_sha256": sha256(source), "blinded_copy_sha256": sha256(destination)})
    mapping_path = coordinator / "blind_id_mapping.csv"
    if mapping_path.exists():
        existing = list(csv.DictReader(mapping_path.open("r", encoding="utf-8-sig", newline="")))
        if existing != [{key: str(value) for key, value in row.items()} for row in mapping]:
            raise RuntimeError("Existing blind-ID mapping differs; refusing to overwrite it")
    else:
        write_csv(mapping_path, mapping)
    selection = [{key: value for key, value in row.items() if key not in {"image_sha256", "annotation_sha256", "image_pixel_sha256", "mask_binary_sha256"}} for row in selected]
    write_csv(ROOT / "results" / "descriptor_validation_selection.csv", selection)

    fields = ["blind_id", "observer_id", "measurement_date", "measurement_software", "dominant_path_length_pixel", "length_status", "length_failure_reason", "first_order_lateral_count", "lateral_status", "lateral_failure_reason", "anatomical_branch_origin_count", "junction_status", "junction_failure_reason", "mean_acute_angle_deg", "valid_angle_n", "angle_status", "angle_failure_reason", "notes"]
    write_csv_new(observer / "manual_measurements_blank.csv", [{"blind_id": row["blind_id"]} for row in mapping], fields)
    protocol = """Blind manual descriptor validation protocol

The observer receives only this folder. Do not provide model predictions, segmentation masks, automatic descriptor values, fold assignments, or the coordinator mapping before all measurements are frozen.

Measure the visible primary-root centerline as a polyline in original-image pixels. Report its Euclidean segment sum in pixel units; do not multiply by 1.12 and do not convert to millimetres. Count directly identifiable first-order laterals and anatomical branch origins using the frozen biological definitions. Measure acute insertion angles only where both direction vectors are valid. If a quantity cannot be measured, enter NA and a reason; never substitute 60 degrees. Record the number of valid angles. This n=10 exercise is preliminary biological validation and is not a basis for robust regression or broad generalization claims.
"""
    protocol_path = observer / "README_blinded_measurement.txt"
    if not protocol_path.exists():
        protocol_path.write_text(protocol, encoding="utf-8")

    if args.selection_only:
        print(json.dumps({"selected_images": 10, "human_measurements_generated": 0, "observer_packet": str(observer)}, ensure_ascii=False, indent=2))
        return

    annotation_rows = []
    settings_grid = [ExtractorSettings(kernel, prune, minimum) for kernel in (3, 5, 7) for prune in (5, 10, 15, 20) for minimum in (5, 10, 20)]
    sensitivity_detail = []
    parts_dir = ROOT / "results" / "parameter_sensitivity_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for row in manifest:
        name = row["sample_name"]
        part_path = parts_dir / f"{name}.json"
        part = None
        if part_path.exists():
            try:
                candidate = json.loads(part_path.read_text(encoding="utf-8"))
                if candidate.get("sample_name") == name and len(candidate.get("sensitivity_rows", [])) == len(settings_grid):
                    part = candidate
            except (OSError, json.JSONDecodeError):
                part = None
        if part is None:
            mask = read_gray(DATA_DIR / f"{name}_mask.png")
            annotation = {"sample_name": name, "source": "annotation_mask", **extract_descriptors(mask)}
            sensitivity_rows = [{"sample_name": name, **extract_descriptors(mask, settings)} for settings in settings_grid]
            part = {"sample_name": name, "annotation": annotation, "sensitivity_rows": sensitivity_rows}
            atomic_write_json(part_path, part)
        annotation_rows.append(part["annotation"])
        sensitivity_detail.extend(part["sensitivity_rows"])
    write_csv(ROOT / "results" / "annotation_descriptor_values.csv", annotation_rows)
    write_csv(ROOT / "results" / "parameter_sensitivity_detail.csv", sensitivity_detail)

    summary = []
    for settings in settings_grid:
        subset = [row for row in sensitivity_detail if int(row["closing_kernel"]) == settings.closing_kernel and int(row["pruning_iterations"]) == settings.pruning_iterations and int(row["min_component_pixels"]) == settings.min_component_pixels]
        summary.append(
            {
                "closing_kernel": settings.closing_kernel,
                "pruning_iterations": settings.pruning_iterations,
                "min_component_pixels": settings.min_component_pixels,
                "n_images": len(subset),
                "mean_dominant_path_length_pixel": float(np.nanmean([float(row["dominant_path_length"]) for row in subset])),
                "mean_retained_segment_count": float(np.mean([float(row["retained_segment_count"]) for row in subset])),
                "mean_junction_region_count": float(np.mean([float(row["junction_region_count"]) for row in subset])),
                "mean_local_acute_angle_deg": float(np.nanmean([float(row["mean_local_acute_angle_deg"]) for row in subset])),
                "images_with_valid_angle": int(sum(int(row["valid_angle_n"]) > 0 for row in subset)),
                "is_default_not_claimed_optimal": settings == ExtractorSettings(),
            }
        )
    write_csv(ROOT / "parameter_sensitivity.csv", summary)
    descriptor_fields = ["blind_id", "sample_name", "source", "trait", "reference_value", "estimated_value", "difference", "absolute_error", "percentage_error_if_reference_nonzero", "status", "failure_reason"]
    write_csv_new(ROOT / "descriptor_validation.csv", [], descriptor_fields)
    print(json.dumps({"selected_images": 10, "sensitivity_combinations": len(summary), "human_measurements_generated": 0, "scale_status": "unknown", "length_unit": "pixel"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
