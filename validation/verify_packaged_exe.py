"""Run the actual Windows EXE on one held-out full image and record its scores."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--run-root", type=Path, default=ROOT / "validation_runs" / "packaged_binary")
    parser.add_argument("--report", type=Path, default=ROOT / "validation" / "packaged_binary_full_image.json")
    args = parser.parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    before = set(args.run_root.glob("RootScope_*"))
    subprocess.run([str(args.exe.resolve()), "--batch", str(args.image.resolve()),
                    "--references", str(args.reference.resolve()), "--output", str(args.run_root.resolve()),
                    "--model", "fold_0", "--no-zip"], check=True, timeout=900)
    folders = set(args.run_root.glob("RootScope_*")) - before
    if len(folders) != 1:
        raise RuntimeError(f"Expected one new result directory; found {len(folders)}")
    folder = folders.pop()
    provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    results = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    if len(results) != 1 or provenance["successful_images"] != 1:
        raise RuntimeError("Packaged run did not finish one full image")
    item = results[0]
    if item["image_sha256"] != sha256(args.image) or item["reference_sha256"] != sha256(args.reference):
        raise ValueError("Packaged run input hashes differ")
    if provenance["model_id"] != "fold_0" or provenance["onnx_sha256"] != json.loads(
            (ROOT / "model" / "catalog.json").read_text(encoding="utf-8"))["models"][0]["sha256"]:
        raise ValueError("Packaged run used an unexpected model")
    with (ROOT / "validation" / "deployment_seed42_oof.csv").open(encoding="utf-8-sig", newline="") as stream:
        matching = [row for row in csv.DictReader(stream) if row["image_name"] == args.image.name]
    if len(matching) != 1 or int(matching[0]["fold"]) != 0:
        raise ValueError("The image is not in held-out fold 0")
    study = matching[0]
    report = {"status": "verified", "software_version": provenance["application"],
              "binary_sha256": sha256(args.exe), "image_name": args.image.name,
              "image_sha256": item["image_sha256"], "reference_sha256": item["reference_sha256"],
              "width": item["width"], "height": item["height"], "model_id": provenance["model_id"],
              "onnx_sha256": provenance["onnx_sha256"], "threshold": provenance["used_threshold"],
              "packaged_dice": item["evaluation"]["dice"],
              "packaged_iou": item["evaluation"]["iou"],
              "source_pipeline_dice": float(study["eval_dice"]),
              "source_pipeline_iou": float(study["eval_iou"]),
              "dice_absolute_difference": abs(item["evaluation"]["dice"] - float(study["eval_dice"])),
              "iou_absolute_difference": abs(item["evaluation"]["iou"] - float(study["eval_iou"])),
              "interpretation": "Single full-image smoke and parity test of the packaged Windows binary; the 50-image OOF validation uses the same source pipeline and released ONNX files."}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
