from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from experiment_core import CONFIGS, ROOT


SEEDS = [42, 3407, 2026]
METRICS = [
    "dice",
    "iou",
    "hard_cldice",
    "precision",
    "recall",
    "width_le3_recall",
    "width_3_6_recall",
    "width_6_12_recall",
    "width_gt12_recall",
]
COMPARISONS = [
    ("attunet_pixel", "unet_pixel", "attention_architecture_effect"),
    ("attunet_cl_fixed", "attunet_pixel", "fixed_skeleton_loss_effect"),
    ("attunet_cl_schedule", "attunet_cl_fixed", "two_stage_schedule_effect"),
]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def numeric(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.zeros(len(p_values), dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        candidate = (m - rank) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def main() -> None:
    all_rows: list[dict] = []
    expected_runs = len(CONFIGS) * len(SEEDS) * 5
    complete_files = sorted((ROOT / "results" / "runs").glob("*/seed_*/fold_*/complete.json"))
    if len(complete_files) != expected_runs:
        raise RuntimeError(f"Expected {expected_runs} complete runs, found {len(complete_files)}")

    for config in CONFIGS:
        for seed in SEEDS:
            seen: set[str] = set()
            for fold in range(5):
                path = ROOT / "results" / "runs" / config / f"seed_{seed}" / f"fold_{fold}" / "test_metrics.csv"
                for row in read_csv(path):
                    name = row["name"]
                    if name in seen:
                        raise RuntimeError(f"Duplicate OOF prediction: {config} seed={seed} sample={name}")
                    seen.add(name)
                    all_rows.append(row)
            if len(seen) != 50:
                raise RuntimeError(f"Expected 50 OOF images for {config} seed={seed}, found {len(seen)}")
            prediction_dir = ROOT / "oof_predictions" / config / f"seed_{seed}"
            if len(list(prediction_dir.glob("*.png"))) != 50:
                raise RuntimeError(f"Expected 50 OOF masks in {prediction_dir}")

    write_csv(ROOT / "results" / "all_oof_image_metrics.csv", all_rows)
    lookup = {(row["config"], int(row["seed"]), row["name"]): row for row in all_rows}
    sample_names = sorted({row["name"] for row in all_rows})

    per_seed_rows: list[dict] = []
    for config in CONFIGS:
        for seed in SEEDS:
            subset = [lookup[(config, seed, name)] for name in sample_names]
            per_seed_rows.append(
                {
                    "config": config,
                    "seed": seed,
                    "n_images": len(subset),
                    **{f"macro_{metric}": float(np.nanmean([numeric(row[metric]) for row in subset])) for metric in METRICS},
                }
            )
    write_csv(ROOT / "results" / "per_seed_oof_summary.csv", per_seed_rows)

    summary_rows: list[dict] = []
    for config in CONFIGS:
        subset = [row for row in per_seed_rows if row["config"] == config]
        for metric in METRICS:
            values = np.asarray([float(row[f"macro_{metric}"]) for row in subset], dtype=float)
            summary_rows.append(
                {
                    "config": config,
                    "endpoint": f"OOF macro {metric}",
                    "n_images_per_seed": 50,
                    "n_seeds": 3,
                    "mean": float(np.nanmean(values)),
                    "sd_across_seeds": float(np.nanstd(values, ddof=1)),
                }
            )
    write_csv(ROOT / "run_summary.csv", summary_rows)

    rng = np.random.default_rng(20260920)
    bootstrap_n = 10000
    effect_rows: list[dict] = []
    raw_p: list[float] = []
    for treatment, control, label in COMPARISONS:
        difference = np.asarray(
            [
                [numeric(lookup[(treatment, seed, name)]["dice"]) - numeric(lookup[(control, seed, name)]["dice"]) for name in sample_names]
                for seed in SEEDS
            ],
            dtype=float,
        )
        observed = float(np.nanmean(difference))
        boot = np.empty(bootstrap_n, dtype=float)
        for index in range(bootstrap_n):
            seed_index = rng.integers(0, len(SEEDS), len(SEEDS))
            image_index = rng.integers(0, len(sample_names), len(sample_names))
            boot[index] = float(np.nanmean(difference[np.ix_(seed_index, image_index)]))
        low, high = np.quantile(boot, [0.025, 0.975])
        p_value = float(min(1.0, 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))))
        raw_p.append(p_value)
        fold_effects = []
        for fold in range(5):
            values = [numeric(row["dice"]) for row in all_rows if row["config"] == treatment and int(row["fold"]) == fold]
            controls = [numeric(row["dice"]) for row in all_rows if row["config"] == control and int(row["fold"]) == fold]
            fold_effects.append(float(np.nanmean(values) - np.nanmean(controls)))
        effect_rows.append(
            {
                "comparison": label,
                "treatment": treatment,
                "control": control,
                "endpoint": "OOF macro Dice",
                "mean_paired_difference": observed,
                "bootstrap_ci95_low": float(low),
                "bootstrap_ci95_high": float(high),
                "bootstrap_p_two_sided": p_value,
                "holm_adjusted_p": "",
                "positive_folds": int(sum(value > 0 for value in fold_effects)),
                "fold_effects": ";".join(f"{value:.8f}" for value in fold_effects),
                "bootstrap_replicates": bootstrap_n,
                "bootstrap_seed": 20260920,
            }
        )
    for row, adjusted in zip(effect_rows, holm_adjust(raw_p)):
        row["holm_adjusted_p"] = adjusted
    write_csv(ROOT / "paired_config_effects.csv", effect_rows)

    schedule = next(row for row in effect_rows if row["comparison"] == "two_stage_schedule_effect")
    stable = float(schedule["bootstrap_ci95_low"]) > 0 and int(schedule["positive_folds"]) >= 4
    conclusion = {
        "schedule_stable_improvement_rule_met": stable,
        "allowed_wording": (
            "在本数据集上取得稳定改善" if stable else
            "均值较高，但未证明具有稳定优势" if float(schedule["mean_paired_difference"]) > 0 else
            "均值较低，未见相对固定权重的稳定改善"
        ),
        "rule": "CI lower bound > 0 and positive direction in at least 4 of 5 folds",
    }
    atomic_write_json(ROOT / "results" / "conclusion_rule.json", conclusion)
    print(json.dumps({"status": "complete", "runs": expected_runs, **conclusion}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
