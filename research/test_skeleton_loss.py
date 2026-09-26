from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "cache" / "matplotlib"))
import matplotlib.pyplot as plt

from experiment_core import ROOT, corrected_soft_skeleton, legacy_soft_skeleton


def pattern(kind: str, width: int = 1) -> torch.Tensor:
    canvas = np.zeros((1, 1, 64, 64), dtype=np.float32)
    if kind == "horizontal":
        canvas[:, :, 32:32 + width, 8:56] = 1
    elif kind == "vertical":
        canvas[:, :, 8:56, 32:32 + width] = 1
    elif kind == "gap":
        canvas[:, :, 32:32 + width, 8:29] = 1
        canvas[:, :, 32:32 + width, 35:56] = 1
    elif kind == "branch":
        canvas[:, :, 8:56, 31:31 + width] = 1
        for offset in range(24):
            canvas[:, :, 31 + offset:31 + offset + width, 31 + offset] = 1
    elif kind == "crossing":
        canvas[:, :, 31:31 + width, 8:56] = 1
        canvas[:, :, 8:56, 31:31 + width] = 1
    else:
        raise ValueError(kind)
    return torch.from_numpy(canvas)


def main() -> None:
    result_dir = ROOT / "results" / "skeleton_unit_tests"
    result_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind in ("horizontal", "vertical", "gap", "branch", "crossing"):
        for width in (1, 2, 3, 4):
            image = pattern(kind, width)
            legacy = legacy_soft_skeleton(image)
            corrected = corrected_soft_skeleton(image)
            row = {
                "pattern": kind,
                "width_px": width,
                "input_mass": float(image.sum()),
                "legacy_skeleton_mass": float(legacy.sum()),
                "corrected_skeleton_mass": float(corrected.sum()),
                "legacy_nonzero_pixels": int(torch.count_nonzero(legacy)),
                "corrected_nonzero_pixels": int(torch.count_nonzero(corrected)),
            }
            rows.append(row)
    with (result_dir / "synthetic_skeleton_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    thin = [row for row in rows if row["width_px"] <= 2]
    checks = {
        "corrected_retains_all_thin_patterns": all(row["corrected_skeleton_mass"] > 0 for row in thin),
        "legacy_loses_at_least_one_thin_pattern": any(row["legacy_skeleton_mass"] == 0 for row in thin),
        "n_cases": len(rows),
    }
    (result_dir / "unit_test_summary.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5})
    fig, ax = plt.subplots(figsize=(6.0, 2.8), dpi=600)
    positions = np.arange(1, 5)
    legacy_means = [np.mean([row["legacy_skeleton_mass"] / row["input_mass"] for row in rows if row["width_px"] == width]) for width in positions]
    corrected_means = [np.mean([row["corrected_skeleton_mass"] / row["input_mass"] for row in rows if row["width_px"] == width]) for width in positions]
    ax.plot(positions, np.asarray(legacy_means) * 100, marker="o", lw=1.7, color="#D55E00", label="Former recurrence")
    ax.plot(positions, np.asarray(corrected_means) * 100, marker="s", lw=1.7, color="#0072B2", label="Corrected recurrence")
    for row in rows:
        ax.scatter(row["width_px"], row["legacy_skeleton_mass"] / row["input_mass"] * 100, color="#D55E00", alpha=0.3, s=12)
        ax.scatter(row["width_px"], row["corrected_skeleton_mass"] / row["input_mass"] * 100, color="#0072B2", alpha=0.3, s=12)
    ax.set_xticks(positions)
    ax.set_xlabel("Synthetic structure width (pixels)")
    ax.set_ylabel("Soft-skeleton mass / input mass (%)")
    ax.grid(axis="y", color="#E6E6E6", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    output = ROOT / "figures" / "supplementary_soft_skeleton_unit_tests"
    for suffix in ("png", "tif", "pdf", "svg"):
        options = {"dpi": 600} if suffix in {"png", "tif"} else {}
        if suffix == "tif":
            options["pil_kwargs"] = {"compression": "tiff_lzw", "dpi": (600, 600)}
        fig.savefig(output.with_suffix(f".{suffix}"), bbox_inches="tight", facecolor="white", **options)
    plt.close(fig)
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not all((checks["corrected_retains_all_thin_patterns"], checks["legacy_loses_at_least_one_thin_pattern"])):
        sys.exit(1)


if __name__ == "__main__":
    main()
