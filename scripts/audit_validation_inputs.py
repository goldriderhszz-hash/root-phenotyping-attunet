"""Verify all local validation images and masks against the locked manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rootscope.engine import read_gray  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_root", type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT / "research" / "dataset_manifest.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "validation" / "input_integrity.json")
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["included"].lower() == "true"]
    if len(rows) != 50:
        raise ValueError(f"Expected 50 included images; got {len(rows)}")
    for row in rows:
        image_path = args.image_root / row["image_file"]
        mask_path = args.image_root / row["mask_file"]
        if sha256(image_path) != row["image_sha256"]:
            raise ValueError(f"Image bytes differ from locked manifest: {row['sample_name']}")
        image = read_gray(image_path)
        mask = (read_gray(mask_path) > 127).astype(np.uint8)
        if image.shape != mask.shape or image.shape != (int(row["height"]), int(row["width"])):
            raise ValueError(f"Image/mask geometry differs: {row['sample_name']}")
        if hashlib.sha256(mask.tobytes()).hexdigest() != row["mask_binary_sha256"]:
            raise ValueError(f"Mask pixels differ from locked manifest: {row['sample_name']}")
    args.output.parent.mkdir(exist_ok=True)
    report = {"status": "verified", "included_images": len(rows),
              "manifest_sha256": sha256(args.manifest),
              "checks": ["image file SHA-256", "binary mask pixel SHA-256", "image/mask dimensions"],
              "raw_data_in_repository": False}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
