"""Score actual RhizoVision Explorer output against the same held-out masks.

RVE must be run separately. This script selects its threshold on fold 1
(validation for RootScope fold 0), then compares both programs on fold 0.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rootscope.engine import read_gray  # noqa: E402

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scores(prediction: np.ndarray, reference: np.ndarray) -> dict:
    if prediction.shape != reference.shape:
        raise ValueError("Prediction/reference image sizes differ")
    if prediction.dtype != np.bool_ or reference.dtype != np.bool_:
        raise TypeError("Pass explicit Boolean foreground masks to scores()")
    pred, ref = prediction, reference
    tp = int(np.logical_and(pred, ref).sum())
    fp = int(np.logical_and(pred, ~ref).sum())
    fn = int(np.logical_and(~pred, ref).sum())
    return {"precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
            "tp": tp, "fp": fp, "fn": fn}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_rve_input(image_root: Path, output_root: Path, role: str, name: str) -> None:
    source = read_gray(image_root / f"{name}.tif")
    prepared = read_gray(output_root / f"rve_{role}_fold0" / f"{name}.tif")
    if source.shape != prepared.shape or not np.array_equal(source, prepared):
        raise ValueError(f"RVE input polarity or geometry differs from the original grayscale image: {role}/{name}")


def verify_rve_metadata(output_dir: Path, threshold: int) -> None:
    path = output_dir / "metadata.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        settings = {row[0].strip(): row[1].strip() for row in csv.reader(stream) if len(row) >= 2}
    expected = {
        "RhizoVision Explorer Version": "2.0.3",
        "Root type": "Whole root",
        "Image Thresholding Level": str(threshold),
        "Invert images": "false",
        "Keep largest component": "true",
        "Save segmented images": "true",
    }
    for key, value in expected.items():
        if settings.get(key) != value:
            raise ValueError(f"{path}: expected {key}={value!r}, got {settings.get(key)!r}")


def rve_foreground(path: Path) -> np.ndarray:
    # RVE's saved PNG uses black (0) for its segmented root and white (255)
    # for background. Convert explicitly before comparing with white-root refs.
    mask = read_gray(path)
    if not np.isin(mask, (0, 255)).all():
        raise ValueError(f"RVE output is not a binary 0/255 image: {path}")
    return mask == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_root", type=Path)
    parser.add_argument("rve_output_parent", type=Path)
    parser.add_argument("--thresholds", type=int, nargs="+", required=True,
                        help="Threshold grid fixed on validation fold 1 before examining fold 0")
    parser.add_argument("--assignments", type=Path, default=ROOT / "research" / "fold_assignments.json")
    parser.add_argument("--rootscope-results", type=Path,
                        default=ROOT / "validation" / "deployment_seed42_oof.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark")
    args = parser.parse_args()
    thresholds = tuple(sorted(set(args.thresholds)))
    if not thresholds or any(level < 0 or level > 255 for level in thresholds):
        parser.error("--thresholds must contain integers between 0 and 255")
    folds = json.loads(args.assignments.read_text(encoding="utf-8"))["folds"]
    for role, fold in (("validation", "1"), ("test", "0")):
        for name in folds[fold]:
            verify_rve_input(args.image_root, args.rve_output_parent, role, name)
    validation_rows = []
    output_hashes = []
    for threshold in thresholds:
        verify_rve_metadata(args.rve_output_parent / f"rve_validation_t{threshold}", threshold)
        for name in folds["1"]:
            mask_path = args.rve_output_parent / f"rve_validation_t{threshold}" / f"{name}_seg.png"
            pred = rve_foreground(mask_path)
            ref = read_gray(args.image_root / f"{name}_mask.png") > 127
            validation_rows.append({"name": name, "threshold": threshold, **scores(pred, ref)})
            output_hashes.append({"role": "validation", "name": name, "threshold": threshold,
                                  "segmentation_sha256": sha256(mask_path)})
    macro = {threshold: float(np.mean([r["dice"] for r in validation_rows if r["threshold"] == threshold]))
             for threshold in thresholds}
    selected = min(thresholds, key=lambda threshold: (-macro[threshold], threshold))
    if not (args.rve_output_parent / f"rve_test_t{selected}").is_dir():
        raise FileNotFoundError(f"RVE held-out test output for selected threshold {selected} is absent")
    verify_rve_metadata(args.rve_output_parent / f"rve_test_t{selected}", selected)
    with args.rootscope_results.open(encoding="utf-8-sig", newline="") as stream:
        rootscope = {Path(row["image_name"]).stem: row for row in csv.DictReader(stream)
                     if int(row["fold"]) == 0}
    if set(rootscope) != set(folds["0"]):
        raise ValueError("RootScope held-out fold-0 rows are incomplete or mismatched")
    paired = []
    for name in folds["0"]:
        mask_path = args.rve_output_parent / f"rve_test_t{selected}" / f"{name}_seg.png"
        pred = rve_foreground(mask_path)
        ref = read_gray(args.image_root / f"{name}_mask.png") > 127
        rve = scores(pred, ref)
        own = rootscope[name]
        output_hashes.append({"role": "test", "name": name, "threshold": selected,
                              "segmentation_sha256": sha256(mask_path)})
        paired.append({"name": name, "rve_threshold": selected,
                       "rve_dice": rve["dice"], "rve_iou": rve["iou"],
                       "rve_precision": rve["precision"], "rve_recall": rve["recall"],
                       "rve_tp": rve["tp"], "rve_fp": rve["fp"], "rve_fn": rve["fn"],
                       "rootscope_model": "fold_0", "rootscope_threshold": float(own["threshold"]),
                       "rootscope_dice": float(own["eval_dice"]),
                       "rootscope_iou": float(own["eval_iou"]),
                       "paired_dice_difference": float(own["eval_dice"]) - rve["dice"],
                       "paired_iou_difference": float(own["eval_iou"]) - rve["iou"]})
    args.output.mkdir(exist_ok=True)
    write_csv(args.output / "rve_validation_thresholds.csv", validation_rows)
    write_csv(args.output / "rve_fold0_per_image.csv", paired)
    write_csv(args.output / "rve_output_hashes.csv", output_hashes)
    differences = np.array([row["paired_dice_difference"] for row in paired])
    rng = np.random.default_rng(20260926)
    boot = np.mean(differences[rng.integers(0, len(differences), (10000, len(differences)))], axis=1)
    summary = {
        "comparator": "RhizoVision Explorer 2.0.3, official Zenodo Windows archive",
        "comparator_doi": "https://doi.org/10.5281/zenodo.3747697",
        "comparator_archive_sha256": "c4822980cdbfbb213f88cd1d738d4e7423f4064ce926ae880b5401d58dc896fd",
        "domain": "full-resolution rhizobag images from the manuscript",
        "rve_settings": "Whole root mode; Keep largest component on; RVE inversion off; original grayscale with dark roots on light background; save binary segmentations",
        "rve_export_foreground": "black (pixel value 0); converted to Boolean root foreground before scoring",
        "threshold_selection": f"Highest validation macro Dice on fold 1 over prespecified {list(thresholds)}; lowest threshold breaks exact ties",
        "rve_validation_macro_dice_by_threshold": macro,
        "selected_rve_threshold": selected,
        "test_fold": 0, "test_n": len(paired),
        "rootscope_test_macro_dice": float(np.mean([r["rootscope_dice"] for r in paired])),
        "rve_test_macro_dice": float(np.mean([r["rve_dice"] for r in paired])),
        "rootscope_test_macro_iou": float(np.mean([r["rootscope_iou"] for r in paired])),
        "rve_test_macro_iou": float(np.mean([r["rve_iou"] for r in paired])),
        "paired_mean_dice_difference": float(np.mean(differences)),
        "paired_mean_dice_difference_bootstrap_95_percentile_ci": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
        "wilcoxon_two_sided_p": float(wilcoxon(differences).pvalue),
        "interpretation": "Same-domain operational comparison on one held-out fold. RVE is designed for high-contrast scanned/excavated roots; this does not establish general superiority across other imaging domains."
    }
    (args.output / "rve_fold0_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
