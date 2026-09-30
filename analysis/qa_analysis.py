"""Check completeness, pairing, and grid/default agreement for manuscript data."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import os

from compute_oof_descriptors import CONFIGS, OUT, ROOT, SEEDS, TRAITS, read_csv


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nfloat(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def equal_number(a, b, tolerance=1e-8):
    x, y = nfloat(a), nfloat(b)
    return (math.isnan(x) and math.isnan(y)) or math.isclose(x, y, rel_tol=tolerance, abs_tol=tolerance)


def key(row):
    return row["sample_name"], row["config"], int(row["seed"])


def main():
    references = read_csv(OUT / "annotation_descriptors.csv")
    predictions = read_csv(OUT / "oof_descriptors.csv")
    pairs = read_csv(OUT / "descriptor_comparison_long.csv")
    summary_seeds = read_csv(OUT / "descriptor_summary_by_seed.csv")
    summary_folds = read_csv(OUT / "descriptor_summary_by_fold_seed.csv")
    summary_across = read_csv(OUT / "descriptor_summary_across_seeds.csv")
    paired_long = read_csv(OUT / "descriptor_paired_error_long.csv")
    paired_summary = read_csv(OUT / "descriptor_paired_effects_summary.csv")
    assert len(references) == 50
    assert len(predictions) == 600
    assert len(pairs) == 2400
    assert len(summary_seeds) == 4 * 3 * 4
    assert len(summary_folds) == 4 * 3 * 5 * 4
    assert len(summary_across) == 4 * 4
    assert len(paired_long) == 3 * 4 * 3 * 50
    assert len(paired_summary) == 3 * 4
    assert len({row["sample_name"] for row in references}) == 50
    assert len({key(row) for row in predictions}) == 600
    assert len({(row["sample_name"], row["config"], row["seed"], row["trait"]) for row in pairs}) == 2400
    expected = {(config, seed): set() for config in CONFIGS for seed in SEEDS}
    for row in predictions:
        expected[(row["config"], int(row["seed"]))].add(row["sample_name"])
    assert all(len(names) == 50 for names in expected.values())
    reference_names = {row["sample_name"] for row in references}
    assert all(names == reference_names for names in expected.values())
    for row in pairs:
        if row["status"] == "valid":
            assert equal_number(float(row["estimate_value"]) - float(row["reference_value"]),
                                row["difference_estimate_minus_reference"])
    for row in paired_long:
        assert equal_number(float(row["treatment_absolute_error"]) - float(row["control_absolute_error"]),
                            row["paired_difference_treatment_minus_control"])
    for row in predictions:
        if row["source"] != "oof_prediction" or row["length_unit"] != "pixel":
            raise AssertionError(row)

    grid_ref = read_csv(OUT / "sensitivity_annotation_grid.csv")
    grid_pred = read_csv(OUT / "sensitivity_fixed_oof_grid.csv")
    grid_pairs = read_csv(OUT / "sensitivity_comparison_long.csv")
    grid_by_seed = read_csv(OUT / "sensitivity_summary_by_seed.csv")
    grid_across = read_csv(OUT / "sensitivity_summary_across_seeds.csv")
    grid_ref_summary = read_csv(OUT / "sensitivity_annotation_summary.csv")
    assert len(grid_ref) == 50 * 36
    assert len(grid_pred) == 3 * 50 * 36
    assert len(grid_pairs) == 3 * 50 * 36 * 4
    assert len(grid_by_seed) == 3 * 36 * 4
    assert len(grid_across) == 36 * 4
    assert len(grid_ref_summary) == 36
    default = lambda row: (int(row["closing_kernel"]), int(row["pruning_iterations"]),
                           int(row["min_component_pixels"])) == (5, 15, 10)
    base_reference = {row["sample_name"]: row for row in references}
    base_prediction = {key(row): row for row in predictions if row["config"] == "attunet_cl_fixed"}
    default_reference = [row for row in grid_ref if default(row)]
    default_prediction = [row for row in grid_pred if default(row)]
    assert len(default_reference) == 50
    assert len(default_prediction) == 150
    for row in default_reference:
        reference = base_reference[row["sample_name"]]
        for trait in TRAITS:
            assert equal_number(row[trait], reference[trait]), (row["sample_name"], trait)
    for row in default_prediction:
        baseline = base_prediction[key(row)]
        for trait in TRAITS:
            assert equal_number(row[trait], baseline[trait]), (key(row), trait)
    for source in (grid_ref, grid_pred):
        by_image = {}
        for row in source:
            by_image.setdefault((row["sample_name"], row["seed"]), []).append(row)
        assert all(len(rows) == 36 for rows in by_image.values())
        for rows in by_image.values():
            lengths = [nfloat(row["dominant_path_length"]) for row in rows]
            assert all(equal_number(lengths[0], x) for x in lengths)

    report = {
        "reference_images": len(references), "prediction_masks": len(predictions),
        "configurations": list(CONFIGS), "seeds": list(SEEDS),
        "descriptor_pair_rows": len(pairs),
        "paired_config_error_rows": len(paired_long),
        "valid_descriptor_pairs": sum(row["status"] == "valid" for row in pairs),
        "grid_settings": 36, "grid_reference_rows": len(grid_ref),
        "grid_prediction_rows": len(grid_pred), "grid_pair_rows": len(grid_pairs),
        "default_setting_matches_base_extractor": True,
        "dominant_path_invariant_across_channel_b_parameters": True,
        "reference_type": "annotation_mask_same_extractor",
        "human_reference_available": False, "physical_calibration_available": False,
        "source_manifest_sha256": sha256(ROOT / "dataset_manifest.csv"),
        "source_fold_assignments_sha256": sha256(ROOT / "fold_assignments.json"),
        "source_extractor_sha256": sha256(Path(__file__).resolve().parents[1] / "research" / "phenotype_extractor.py"),
    }
    (OUT / "analysis_qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
