"""Prepare dark-root, light-background grayscale images for RVE."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--assignments", type=Path, default=ROOT / "research" / "fold_assignments.json")
    args = parser.parse_args()
    folds = json.loads(args.assignments.read_text(encoding="utf-8"))["folds"]
    rows = []
    for role, fold in (("validation", 1), ("test", 0)):
        target = args.output_root / f"rve_{role}_fold0"
        target.mkdir(parents=True, exist_ok=True)
        for name in folds[str(fold)]:
            source = args.image_root / f"{name}.tif"
            destination = target / source.name
            gray = cv2.imdecode(np.fromfile(str(source), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError(f"Cannot read {source}")
            # RVE 2.0.3 classifies intensities below its threshold as roots.
            # The source images already have dark roots on a lighter background.
            ok, encoded = cv2.imencode(".tif", gray)
            if not ok:
                raise ValueError(f"Cannot encode {destination}")
            encoded.tofile(str(destination))
            rows.append({"role": role, "name": name,
                         "source_sha256": sha256(source),
                         "rve_input_sha256": sha256(destination),
                         "rve_input_polarity": "dark roots on light background"})
    with (args.output_root / "rve_input_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Prepared {len(rows)} full-resolution grayscale images without intensity inversion")


if __name__ == "__main__":
    main()

