"""Paired differences in descriptor absolute error across OOF configurations.

The interval resamples 50 distinct images in their test folds while keeping
their three-seed observations together. It is descriptive and conditional on
the existing folds, training runs and image collection.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from compute_oof_descriptors import OUT, SEEDS, TRAITS, read_csv, write_csv

COMPARISONS = (
    ("attention_effect", "attunet_pixel", "unet_pixel"),
    ("fixed_skeleton_effect", "attunet_cl_fixed", "attunet_pixel"),
    ("schedule_vs_fixed", "attunet_cl_schedule", "attunet_cl_fixed"),
)


def main() -> None:
    rows = read_csv(OUT / "descriptor_comparison_long.csv")
    data = {(row["sample_name"], int(row["fold"]), int(row["seed"]), row["trait"], row["config"]): row
            for row in rows}
    if len(data) != 2400:
        raise RuntimeError(f"Expected 2400 unique descriptor rows, got {len(data)}")
    images = {(row["sample_name"], int(row["fold"])) for row in rows}
    if len(images) != 50:
        raise RuntimeError(f"Expected 50 unique images, got {len(images)}")
    effects = []
    for label, treatment, control in COMPARISONS:
        for name, fold in sorted(images):
            for seed in SEEDS:
                for trait in TRAITS:
                    a = data[(name, fold, seed, trait, treatment)]
                    b = data[(name, fold, seed, trait, control)]
                    if a["status"] != "valid" or b["status"] != "valid":
                        raise RuntimeError(f"Invalid pair: {name}, {seed}, {trait}")
                    a_error = float(a["absolute_error"])
                    b_error = float(b["absolute_error"])
                    effects.append({"comparison": label, "treatment": treatment, "control": control,
                                    "sample_name": name, "fold": fold, "seed": seed, "trait": trait,
                                    "treatment_absolute_error": a_error, "control_absolute_error": b_error,
                                    "paired_difference_treatment_minus_control": a_error - b_error})
    write_csv(OUT / "descriptor_paired_error_long.csv", effects, list(effects[0]))

    by_seed = []
    by_fold = []
    overall = []
    rng = np.random.default_rng(20260925)
    for label, treatment, control in COMPARISONS:
        for trait in TRAITS:
            selected = [row for row in effects if row["comparison"] == label and row["trait"] == trait]
            seed_means = []
            for seed in SEEDS:
                values = np.asarray([row["paired_difference_treatment_minus_control"]
                                     for row in selected if row["seed"] == seed], dtype=float)
                if len(values) != 50:
                    raise RuntimeError(f"Seed count mismatch: {label}, {trait}, {seed}")
                seed_means.append(float(np.mean(values)))
                by_seed.append({"comparison": label, "treatment": treatment, "control": control,
                                "trait": trait, "seed": seed, "n_distinct_images": 50,
                                "mean_paired_absolute_error_difference": seed_means[-1]})
            image_means = []
            for fold in range(5):
                fold_values = []
                for name in sorted(name for name, f in images if f == fold):
                    values = [row["paired_difference_treatment_minus_control"] for row in selected
                              if row["sample_name"] == name]
                    if len(values) != 3:
                        raise RuntimeError(f"Missing seed in {name}, {label}, {trait}")
                    fold_values.append(float(np.mean(values)))
                    image_means.append((name, fold, float(np.mean(values))))
                by_fold.append({"comparison": label, "treatment": treatment, "control": control,
                                "trait": trait, "fold": fold, "n_distinct_images": 10,
                                "mean_paired_absolute_error_difference": float(np.mean(fold_values))})
            fold_arrays = [np.asarray([value for _, f, value in image_means if f == fold], dtype=float)
                           for fold in range(5)]
            bootstrap = np.empty(10000, dtype=float)
            for iteration in range(len(bootstrap)):
                bootstrap[iteration] = float(np.mean(np.concatenate([
                    array[rng.integers(0, len(array), size=len(array))] for array in fold_arrays
                ])))
            mean_effect = float(np.mean(seed_means))
            overall.append({"comparison": label, "treatment": treatment, "control": control,
                            "trait": trait, "n_distinct_images": 50, "n_training_seeds": 3,
                            "n_paired_predictions": 150,
                            "mean_paired_absolute_error_difference": mean_effect,
                            "seed_sd": float(np.std(seed_means, ddof=1)),
                            "bootstrap_image_within_fold_ci95_low": float(np.percentile(bootstrap, 2.5)),
                            "bootstrap_image_within_fold_ci95_high": float(np.percentile(bootstrap, 97.5)),
                            "positive_seed_count": sum(value > 0 for value in seed_means),
                            "positive_fold_count": sum(row["mean_paired_absolute_error_difference"] > 0
                                                       for row in by_fold if row["comparison"] == label
                                                       and row["trait"] == trait),
                            "direction": "positive means treatment larger absolute error"})
    write_csv(OUT / "descriptor_paired_effects_by_seed.csv", by_seed, list(by_seed[0]))
    write_csv(OUT / "descriptor_paired_effects_by_fold.csv", by_fold, list(by_fold[0]))
    write_csv(OUT / "descriptor_paired_effects_summary.csv", overall, list(overall[0]))
    print(json.dumps({"pair_rows": len(effects), "comparison_rows": len(overall),
                      "bootstrap_draws": 10000}, ensure_ascii=False))


if __name__ == "__main__":
    main()
