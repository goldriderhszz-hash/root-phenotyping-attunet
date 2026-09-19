"""Read-only supplementary segmentation evaluation of the supplied masks.

Run with the supplementary environment's Python; no training is performed.
Original data, original code, checkpoints and saved predictions are never written.
Resampling units are complete images; filename grouping is a sensitivity analysis
only, because a filename stem is not a verified biological plant identifier.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

SEED = 20260903
BOOTSTRAPS = 10000
THRESHOLD = 127
MODELS = {
    "Base U-Net": "base_unet",
    "Attention U-Net": "attention_unet",
    "U-Net + clDice": "unet_skeleton",
    "Ours": "combined",
}
WIDTH_BINS = ((0, 3, "<=3"), (3, 6, "(3,6]"),
              (6, 12, "(6,12]"), (12, np.inf, ">12"))
METRICS = ("precision", "recall", "dice", "iou", "cldice",
           "topology_precision", "topology_recall", "tolerance2_f1")


def progress(message: str) -> None:
    print(message, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        array = np.array(image)
    if array.ndim != 2:
        raise ValueError(f"Expected a single-channel binary mask: {path}: {array.shape}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"Image dimensions differ: {path}: {array.shape} != {shape}")
    if not np.all(np.isin(np.unique(array), (0, 255))):
        raise ValueError(f"Expected saved binary mask with levels 0 and 255: {path}")
    return array > THRESHOLD


def ratio(num: int | float, den: int | float) -> float:
    return float(num / den) if den else 0.0


def harmonic(a: float, b: float) -> float:
    return ratio(2 * a * b, a + b)


def confusion(gt: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(np.count_nonzero(gt & pred))
    fp = int(np.count_nonzero(~gt & pred))
    fn = int(np.count_nonzero(gt & ~pred))
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "dice": ratio(2 * tp, 2 * tp + fp + fn),
            "iou": ratio(tp, tp + fp + fn)}


def bootstrap_means(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    indices = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    return values[indices].mean(axis=1)


def interval(values: np.ndarray) -> tuple[float, float]:
    return tuple(float(x) for x in np.percentile(values, (2.5, 97.5)))


def inventory(root: Path) -> tuple[list[str], list[str], list[dict]]:
    images = root / "data/images"
    masks = root / "data/masks"
    annotations = root / "data/annotations"
    raw = sorted(p.stem for p in images.glob("*.tif"))
    if len(raw) != 50 or len(set(raw)) != 50:
        raise ValueError(f"Expected the documented 50 source image files, got {len(raw)}")
    split = json.loads((root / "data/splits/seed42.json").read_text(encoding="utf-8"))
    names = sorted(split["test"])
    if len(names) != 10 or len(set(names)) != 10:
        raise ValueError(f"Expected 10 unique saved test masks, got {len(names)}")
    files = [Path(__file__).resolve()]
    for name in raw:
        files.extend((images / f"{name}.tif", masks / f"{name}_mask.png", annotations / f"{name}.json"))
    for name in names:
        for experiment in MODELS.values():
            files.append(root / "artifacts/ablation" / experiment / "predictions" / f"{name}.tif")
    manifest = []
    for file in files:
        if not file.is_file():
            raise FileNotFoundError(file)
        stat = file.stat()
        manifest.append({"path": str(file), "bytes": stat.st_size,
                         "mtime_ns": stat.st_mtime_ns, "sha256": sha256(file)})
    return raw, names, manifest


def evaluate(root: Path, raw_names: list[str], names: list[str],
             results: Path, manifest: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    images = root / "data/images"
    masks = root / "data/masks"
    annotations = root / "data/annotations"
    rows, widths, profile = [], [], []
    for index, name in enumerate(raw_names, 1):
        progress(f"GT profile {index:02d}/50: {name}")
        gt = binary(masks / f"{name}_mask.png")
        with Image.open(images / f"{name}.tif") as image:
            if image.size != (gt.shape[1], gt.shape[0]):
                raise ValueError(f"Original image and mask dimensions differ: {name}")
        skel_gt = skeletonize(gt)
        distance_gt = ndimage.distance_transform_edt(gt)
        local_width = 2.0 * distance_gt[skel_gt]
        labels, components = ndimage.label(gt, structure=np.ones((3, 3), dtype=np.uint8))
        sizes = np.bincount(labels.ravel())[1:]
        del labels
        annotation = json.loads((annotations / f"{name}.json").read_text(encoding="utf-8"))
        row = {"image": name, "in_test": name in names, "height": gt.shape[0],
               "width": gt.shape[1], "foreground_pixels": int(gt.sum()),
               "foreground_fraction": float(gt.mean()), "components_8": int(components),
               "components_lt10_pixels": int((sizes < 10).sum()),
               "largest_component_fraction": ratio(int(sizes.max()) if sizes.size else 0, int(gt.sum())),
               "skeleton_pixels": int(skel_gt.sum()),
               "skeleton_width_p10_px": float(np.percentile(local_width, 10)),
               "skeleton_width_median_px": float(np.median(local_width)),
               "skeleton_width_p90_px": float(np.percentile(local_width, 90)),
               "annotation_polygons": len(annotation.get("shapes", []))}
        for low, high, label in WIDTH_BINS:
            row[f"skeleton_pixels_width_{label}"] = int(((local_width > low) & (local_width <= high)).sum())
        profile.append(row)
        if name not in names:
            continue
        # Tolerance is a diagnostic at a fixed 2-pixel Euclidean distance.
        # It is reported alongside strict metrics, never used to replace them.
        near_gt = ndimage.distance_transform_edt(~gt) <= 2.0
        for model, experiment in MODELS.items():
            progress(f"Test {names.index(name) + 1:02d}/10 | {model}: {name}")
            pred = binary(root / "artifacts/ablation" / experiment / "predictions" / f"{name}.tif", gt.shape)
            skel_pred = skeletonize(pred)
            n_sp, n_sg = int(skel_pred.sum()), int(skel_gt.sum())
            hit_sp = int((skel_pred & gt).sum())
            hit_sg = int((skel_gt & pred).sum())
            top_p, top_r = ratio(hit_sp, n_sp), ratio(hit_sg, n_sg)
            near_pred = ndimage.distance_transform_edt(~pred) <= 2.0
            hit_tol_p = int((pred & near_gt).sum())
            hit_tol_g = int((gt & near_pred).sum())
            tol_p, tol_r = ratio(hit_tol_p, int(pred.sum())), ratio(hit_tol_g, int(gt.sum()))
            rows.append({"image": name, "filename_group": re.sub(r"\(\d+\)$", "", name),
                         "model": model, **confusion(gt, pred),
                         "skeleton_pred_pixels": n_sp, "skeleton_gt_pixels": n_sg,
                         "skeleton_pred_in_gt": hit_sp, "skeleton_gt_in_pred": hit_sg,
                         "topology_precision": top_p, "topology_recall": top_r,
                         "cldice": harmonic(top_p, top_r),
                         "tolerance2_pred_hits": hit_tol_p, "tolerance2_gt_hits": hit_tol_g,
                         "tolerance2_precision": tol_p, "tolerance2_recall": tol_r,
                         "tolerance2_f1": harmonic(tol_p, tol_r)})
            covered = pred[skel_gt]
            for low, high, label in WIDTH_BINS:
                selected = (local_width > low) & (local_width <= high)
                denominator = int(selected.sum())
                numerator = int(covered[selected].sum())
                widths.append({"image": name, "model": model, "width_bin_px": label,
                               "gt_skeleton_pixels": denominator,
                               "covered_skeleton_pixels": numerator,
                               "exact_centerline_recall": numerator / denominator if denominator else np.nan})
            del pred, skel_pred, near_pred
        del gt, skel_gt, distance_gt, near_gt
    df = pd.DataFrame(rows)
    width_df = pd.DataFrame(widths)
    df.to_csv(results / "per_image_metrics.csv", index=False)
    width_df.to_csv(results / "per_image_width_recall.csv", index=False)
    pd.DataFrame(profile).to_csv(results / "all50_gt_profile.csv", index=False)
    duplicate_rows = []
    by_hash = defaultdict(list)
    for item in manifest:
        path = Path(item["path"])
        if path.parent == images and path.suffix.lower() == ".tif":
            by_hash[item["sha256"]].append(path.stem)
    for duplicate_names in by_hash.values():
        for left, right in itertools.combinations(duplicate_names, 2):
            gt_left, gt_right = binary(masks / f"{left}_mask.png"), binary(masks / f"{right}_mask.png")
            duplicate_rows.append({"image_a": left, "image_b": right,
                                   "both_in_test": left in names and right in names,
                                   "cross_train_test": (left in names) != (right in names),
                                   "different_mask_pixels": int((gt_left != gt_right).sum()),
                                   **confusion(gt_left, gt_right)})
    pd.DataFrame(duplicate_rows).to_csv(results / "duplicate_image_annotation_diagnostic.csv", index=False)
    return df, width_df


def summarize(df: pd.DataFrame, widths: pd.DataFrame, results: Path) -> None:
    rng = np.random.default_rng(SEED)
    summary = []
    for model in MODELS:
        subset = df.loc[df.model == model].sort_values("image")
        for metric in METRICS:
            values = subset[metric].to_numpy()
            lo, hi = interval(bootstrap_means(values, rng))
            summary.append({"model": model, "aggregation": "macro_image",
                            "metric": metric, "value": float(values.mean()),
                            "ci95_low": lo, "ci95_high": hi, "n_images": len(values)})
        tp, fp, fn = (int(subset[k].sum()) for k in ("tp", "fp", "fn"))
        tp_top = ratio(subset.skeleton_pred_in_gt.sum(), subset.skeleton_pred_pixels.sum())
        tr_top = ratio(subset.skeleton_gt_in_pred.sum(), subset.skeleton_gt_pixels.sum())
        tol_p = ratio(subset.tolerance2_pred_hits.sum(), tp + fp)
        tol_r = ratio(subset.tolerance2_gt_hits.sum(), tp + fn)
        micro = {"precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
                 "dice": ratio(2 * tp, 2 * tp + fp + fn), "iou": ratio(tp, tp + fp + fn),
                 "topology_precision": tp_top, "topology_recall": tr_top,
                 "cldice": harmonic(tp_top, tr_top), "tolerance2_f1": harmonic(tol_p, tol_r)}
        for metric, value in micro.items():
            summary.append({"model": model, "aggregation": "micro_pooled_counts",
                            "metric": metric, "value": value, "ci95_low": np.nan,
                            "ci95_high": np.nan, "n_images": len(subset)})
    pd.DataFrame(summary).to_csv(results / "summary_macro_micro.csv", index=False)
    paired = []
    for metric in METRICS:
        wide = df.pivot(index="image", columns="model", values=metric).sort_index()
        filename_groups = np.array([re.sub(r"\(\d+\)$", "", name) for name in wide.index])
        group_names = np.unique(filename_groups)
        for baseline in list(MODELS)[:-1]:
            differences = (wide["Ours"] - wide[baseline]).to_numpy()
            lo, hi = interval(bootstrap_means(differences, rng))
            paired.append({"baseline": baseline, "metric": metric,
                           "resampling_unit": "complete_image", "n_units": len(differences),
                           "difference_ours_minus_baseline": float(differences.mean()),
                           "ci95_low": lo, "ci95_high": hi,
                           "n_positive_images": int((differences > 0).sum()),
                           "n_negative_images": int((differences < 0).sum()),
                           "bootstrap_replicates": BOOTSTRAPS, "seed": SEED})
            # Sensitivity: resample filename groups, then include all their images.
            # This preserves the image-weighted estimand, rather than treating the
            # three 10-1-2 files as three independently sampled groups.
            sums = np.array([differences[filename_groups == g].sum() for g in group_names])
            counts = np.array([(filename_groups == g).sum() for g in group_names])
            sample = rng.integers(0, len(group_names), (BOOTSTRAPS, len(group_names)))
            means = sums[sample].sum(axis=1) / counts[sample].sum(axis=1)
            lo, hi = interval(means)
            paired.append({"baseline": baseline, "metric": metric,
                           "resampling_unit": "filename_group_UNVERIFIED_biological_identity",
                           "n_units": len(group_names),
                           "difference_ours_minus_baseline": float(differences.mean()),
                           "ci95_low": lo, "ci95_high": hi,
                           "n_positive_images": int((differences > 0).sum()),
                           "n_negative_images": int((differences < 0).sum()),
                           "bootstrap_replicates": BOOTSTRAPS, "seed": SEED})
    pd.DataFrame(paired).to_csv(results / "paired_bootstrap_differences.csv", index=False)
    width_summary = []
    for model in MODELS:
        for _, _, label in WIDTH_BINS:
            subset = widths.loc[(widths.model == model) & (widths.width_bin_px == label)]
            values = subset.exact_centerline_recall.dropna().to_numpy()
            lo, hi = interval(bootstrap_means(values, rng)) if values.size else (np.nan, np.nan)
            den = int(subset.gt_skeleton_pixels.sum())
            num = int(subset.covered_skeleton_pixels.sum())
            width_summary.append({"model": model, "width_bin_px": label,
                                  "n_images_with_bin": len(values),
                                  "gt_skeleton_pixels": den, "covered_skeleton_pixels": num,
                                  "macro_recall": float(values.mean()) if values.size else np.nan,
                                  "ci95_low": lo, "ci95_high": hi,
                                  "micro_recall": num / den if den else np.nan})
    pd.DataFrame(width_summary).to_csv(results / "width_recall_summary.csv", index=False)


def figures(df: pd.DataFrame, results: Path, out: Path, root: Path,
            names: list[str], publication_only: bool = False) -> None:
    # All matplotlib caches stay inside the dedicated supplementary figure folder.
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.family": "Arial", "font.size": 9.5,
                         "axes.labelsize": 9.5, "axes.titlesize": 10,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                         "legend.fontsize": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    colors = ["#476C9B", "#4C9A9A", "#B78035", "#963C63"]
    labels = ["U-Net", "Attention\nU-Net", "U-Net +\nclDice", "Ours"]
    summary = pd.read_csv(results / "summary_macro_micro.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.05), constrained_layout=True)
    for panel, ax, metric, title in zip(("a", "b"), axes, ["dice", "cldice"], ["Pixel Dice", "Hard-skeleton clDice"]):
        wide = df.pivot(index="image", columns="model", values=metric).reindex(columns=MODELS)
        ax.plot(np.arange(4), wide.to_numpy().T * 100, color="#C6CCD1", lw=0.6, alpha=0.7, zorder=1)
        for x, model in enumerate(MODELS):
            ax.scatter(np.full(len(wide), x), wide[model] * 100, s=20, facecolors="white",
                       edgecolors=colors[x], linewidths=0.9, zorder=3)
            row = summary[(summary.model == model) & (summary.metric == metric) &
                          (summary.aggregation == "macro_image")].iloc[0]
            ax.errorbar(x + 0.13, row.value * 100,
                        yerr=[[100 * (row.value - row.ci95_low)], [100 * (row.ci95_high - row.value)]],
                        fmt="s", ms=5, capsize=3.5, color=colors[x], lw=1.5, zorder=4)
        ax.set_xticks(range(4), labels)
        ax.set_ylabel(f"{title} (%)")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.15)
        ax.set_title(f"({panel})", loc="left", fontsize=10, fontweight="bold", pad=4)
    fig.suptitle("All 10 test images; squares show means and 95% bootstrap CIs", fontsize=10)
    for ext in ("png", "tiff", "pdf", "svg"):
        kwargs = {"dpi": 600, "facecolor": "white"}
        if ext == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(out / f"paired_image_dice_cldice.{ext}", **kwargs)
    plt.close(fig)
    width = pd.read_csv(results / "width_recall_summary.csv")
    fig, ax = plt.subplots(figsize=(6.0, 3.25), constrained_layout=True)
    for i, (model, color) in enumerate(zip(MODELS, colors)):
        selected = width.loc[width.model == model].set_index("width_bin_px").loc[[x[2] for x in WIDTH_BINS]]
        x = np.arange(4) + (i - 1.5) * 0.12
        y = selected.macro_recall.to_numpy() * 100
        lo, hi = selected.ci95_low.to_numpy() * 100, selected.ci95_high.to_numpy() * 100
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o-", ms=5, lw=1.5,
                    color=color, capsize=3.5, label=model)
    ax.set_xticks(range(4), ["≤3", "(3, 6]", "(6, 12]", ">12"])
    ax.set_xlabel("GT local width at skeleton pixels (2 × Euclidean distance, px)")
    ax.set_ylabel("Exact centerline recall (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.15)
    ax.legend(ncol=2, loc="lower right", frameon=False)
    ax.set_title("Image means and 95% bootstrap CIs; no spatial tolerance", fontsize=10)
    for ext in ("png", "tiff", "pdf", "svg"):
        kwargs = {"dpi": 600, "facecolor": "white"}
        if ext == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(out / f"width_stratified_centerline_recall.{ext}", **kwargs)
    plt.close(fig)
    if publication_only:
        return
    # Every held-out image is shown, in filename order, with no selection by score.
    fig, axes = plt.subplots(5, 6, figsize=(15, 17), constrained_layout=True)
    for index, name in enumerate(names):
        row, start = index // 2, (index % 2) * 3
        gt = binary(root / "data/masks" / f"{name}_mask.png")
        pred = binary(root / "artifacts/ablation" / MODELS["Ours"] / "predictions" / f"{name}.tif", gt.shape)
        with Image.open(root / "data/images" / f"{name}.tif") as im:
            image = im.copy()
            image.thumbnail((560, 780))
        errors = np.full((*gt.shape, 3), 255, np.uint8)
        errors[gt & pred] = (65, 86, 101)
        errors[pred & ~gt] = (192, 67, 116)
        errors[gt & ~pred] = (223, 145, 30)
        # Maximize legibility while retaining true pixels; preview resampling is
        # display-only and is never used for any reported numerical measurement.
        error_image = Image.fromarray(errors)
        error_image.thumbnail((560, 780))
        mask_image = Image.fromarray(gt.astype(np.uint8) * 255)
        mask_image.thumbnail((560, 780))
        axes[row, start].imshow(image)
        axes[row, start + 1].imshow(mask_image, cmap="gray", vmin=0, vmax=255)
        axes[row, start + 2].imshow(error_image)
        for offset, heading in enumerate(("Original", "Manual mask", "Ours error map")):
            axes[row, start + offset].set_title(f"{name}\n{heading}", fontsize=9)
            axes[row, start + offset].axis("off")
        del errors, gt, pred
    legend = [Patch(color=c, label=l) for c, l in
              [("#415665", "True positive"), ("#C04374", "False positive"), ("#DF911E", "False negative")]]
    fig.legend(handles=legend, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(out / "all10_test_original_mask_errors.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Recompute, even if the complete hash-matched result exists")
    args = parser.parse_args()
    root = args.workspace.resolve()
    results = (args.output_dir or (root / "runs/evaluation/segmentation")).resolve()
    plots = results / "figures"
    for output in (results, plots):
        if not output.resolve().is_relative_to(root):
            raise ValueError(f"Unsafe output location: {output}")
        output.mkdir(parents=True, exist_ok=True)
    raw_names, names, manifest = inventory(root)
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    complete = results / "complete.json"
    if complete.exists() and not args.force:
        previous = json.loads(complete.read_text(encoding="utf-8"))
        if previous.get("input_signature") == signature and all(Path(p).is_file() for p in previous.get("outputs", [])):
            progress("Complete matching results already exist; no computation repeated.")
            return
    (results / "input_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    config = {"threshold": THRESHOLD, "seed": SEED, "bootstrap_replicates": BOOTSTRAPS,
              "models": MODELS, "test_images": names,
              "cldice_definition": "2*tprec*tsens/(tprec+tsens); tprec=|S(P) intersect G|/|S(P)|; tsens=|S(G) intersect P|/|S(G)|; skimage.skeletonize",
              "width_definition": "2*scipy.ndimage.distance_transform_edt(G), sampled at GT skeleton pixels; bins <=3,(3,6],(6,12],>12 px; exact prediction coverage",
              "diagnostic_tolerance": "Euclidean distance <=2 px to the other complete foreground mask; conventional relaxed precision/recall harmonic mean",
              "zero_denominator_policy": "zero for binary overlap/topology ratios; NaN for empty width strata",
              "resampling_unit": "complete image; additional unverified filename-group sensitivity",
              "evaluation_scope": "all four supplied ablation prediction sets, all ten supplied test images; no new training; no tuning"}
    (results / "evaluation_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    df, width_df = evaluate(root, raw_names, names, results, manifest)
    summarize(df, width_df, results)
    figures(df, results, plots, root, names)
    # Recheck every input's content and timestamps after all analysis and plotting.
    for item in manifest:
        path = Path(item["path"])
        stat = path.stat()
        if stat.st_size != item["bytes"] or stat.st_mtime_ns != item["mtime_ns"] or sha256(path) != item["sha256"]:
            raise RuntimeError(f"An input changed during evaluation: {path}")
    outputs = [str(p) for folder in (results, plots) for p in folder.glob("*") if p.is_file() and p != complete]
    complete.write_text(json.dumps({"completed_utc": datetime.now(timezone.utc).isoformat(),
                                    "input_signature": signature, "outputs": outputs,
                                    "input_integrity_verified": True,
                                    "n_test_images": len(names), "n_source_image_files": len(raw_names)},
                                   indent=2, ensure_ascii=False), encoding="utf-8")
    progress(f"COMPLETE: {results}; all input hashes and modification times unchanged.")


if __name__ == "__main__":
    main()
