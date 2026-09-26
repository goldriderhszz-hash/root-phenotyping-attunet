"""Run the released ONNX desktop pipeline on each fold's held-out full images.

This is out-of-fold validation of five frozen models, not external validation or
accuracy evidence for an ensemble. Raw images and masks stay outside the repo.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rootscope.batch import run_batch  # noqa: E402
from rootscope.models import model_spec  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_root", type=Path)
    parser.add_argument("image_root", type=Path)
    parser.add_argument("--run-root", type=Path, default=ROOT / "validation_runs")
    parser.add_argument("--report-root", type=Path, default=ROOT / "validation")
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    args = parser.parse_args()
    assignments = json.loads((args.study_root / "fold_assignments.json").read_text(encoding="utf-8"))
    args.report_root.mkdir(parents=True, exist_ok=True)
    existing = args.report_root / "deployment_seed42_oof.csv"
    all_rows = [row for row in read_csv(existing) if int(row["fold"]) not in args.folds] if existing.exists() else []
    for fold in args.folds:
        names = assignments["folds"][str(fold)]
        if len(names) != 10:
            raise ValueError(f"Expected 10 held-out images for fold {fold}")
        images = [args.image_root / f"{name}.tif" for name in names]
        references = [args.image_root / f"{name}_mask.png" for name in names]
        if any(not path.is_file() for path in images + references):
            raise FileNotFoundError(f"Missing full-size input or mask for fold {fold}")
        spec = model_spec(f"fold_{fold}")
        print(f"Running fold {fold}, n=10, threshold={spec['threshold']}", flush=True)
        result = run_batch(images, references, args.run_root, model_id=spec["id"], make_zip=False,
                           progress=lambda done, total, msg: print(f"  {done}/{total} {msg}", flush=True))
        if result["errors"] or len(result["results"]) != 10:
            raise RuntimeError(f"Fold {fold} incomplete: {result['errors']}")
        rows = read_csv(Path(result["folder"]) / "results.csv")
        compact_metrics = args.study_root / "reference_metrics" / f"fold_{fold}_test_metrics.csv"
        original_metrics = compact_metrics if compact_metrics.exists() else (
            args.study_root / "results" / "runs" / "attunet_cl_fixed" / "seed_42" /
            f"fold_{fold}" / "test_metrics.csv")
        original = {row["name"]: row for row in read_csv(original_metrics)}
        for row in rows:
            name = Path(row["image_name"]).stem
            study_row = original[name]
            row["fold"] = fold
            row["model_id"] = spec["id"]
            row["onnx_sha256"] = spec["sha256"]
            row["research_dice"] = study_row["dice"]
            row["research_iou"] = study_row["iou"]
            row["dice_absolute_difference_from_research"] = abs(float(row["eval_dice"]) - float(study_row["dice"]))
            row["iou_absolute_difference_from_research"] = abs(float(row["eval_iou"]) - float(study_row["iou"]))
            all_rows.append(row)
        print(f"Fold {fold} complete: macro Dice={np.mean([float(r['eval_dice']) for r in rows]):.6f}", flush=True)
        write_reports(args.report_root, all_rows, list(range(5)))


def write_reports(report_root: Path, rows: list[dict], requested_folds: list[int]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path = report_root / "deployment_seed42_oof.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    values = lambda key: np.array([float(row[key]) for row in rows], dtype=float)
    tp, fp, fn = (sum(int(row[f"eval_{key}"]) for row in rows) for key in ("tp", "fp", "fn"))
    dice = values("eval_dice")
    rng = np.random.default_rng(20260926)
    bootstrap = np.mean(dice[rng.integers(0, len(dice), size=(10000, len(dice)))], axis=1)
    summary = {
        "protocol": "Locked seed-42 five-fold out-of-fold validation through released ONNX desktop pipeline",
        "completed_folds": sorted(set(int(row["fold"]) for row in rows)),
        "requested_folds": requested_folds,
        "n_full_images": len(rows),
        "image_level_macro_dice": float(np.mean(dice)),
        "image_level_macro_dice_bootstrap_95_percentile_ci": [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
        "image_level_macro_iou": float(np.mean(values("eval_iou"))),
        "image_level_macro_hard_cldice": float(np.mean(values("eval_hard_cldice"))),
        "pooled_dice": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None,
        "pooled_iou": (tp / (tp + fp + fn)) if (tp + fp + fn) else None,
        "max_dice_absolute_difference_from_research": float(max(values("dice_absolute_difference_from_research"))),
        "max_iou_absolute_difference_from_research": float(max(values("iou_absolute_difference_from_research"))),
        "descriptor_image_space_mae": {},
        "folds": {},
        "interpretation": "These are internal OOF scores of five separate checkpoints, not accuracy of one checkpoint on 50 images or external generalization."
    }
    for descriptor in ("dominant_path_length", "retained_segment_count",
                       "junction_region_count", "mean_local_acute_angle_deg"):
        key = "error_" + descriptor
        errors = [abs(float(row[key])) for row in rows if row.get(key) not in (None, "")
                  and math.isfinite(float(row[key]))]
        summary["descriptor_image_space_mae"][descriptor] = {
            "n_evaluable": len(errors), "mae": float(np.mean(errors)) if errors else None,
            "units": {"dominant_path_length": "pixels", "retained_segment_count": "segments",
                      "junction_region_count": "regions", "mean_local_acute_angle_deg": "degrees"}[descriptor],
        }
    for fold in sorted(set(int(row["fold"]) for row in rows)):
        subset = [row for row in rows if int(row["fold"]) == fold]
        summary["folds"][str(fold)] = {
            "n": len(subset), "threshold": float(model_spec(f"fold_{fold}")["threshold"]),
            "macro_dice": float(np.mean([float(row["eval_dice"]) for row in subset])),
            "macro_iou": float(np.mean([float(row["eval_iou"]) for row in subset])),
        }
    (report_root / "deployment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
