"""Export the five locked, seed-42 study checkpoints to deployment ONNX files.

This script never selects epochs or thresholds. It verifies the previously
selected checkpoint against the supplied study record before conversion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rootscope.network import AttentionUNet  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_root", type=Path, help="Root of the completed five-fold experiment")
    parser.add_argument("--checkpoint-root", type=Path, help="Optional minimal archive checkpoints/ directory")
    parser.add_argument("--record-root", type=Path, help="Optional companion research/run_records/ directory")
    parser.add_argument("--output", type=Path, help="Export into a separate directory to preserve locked release models")
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    args = parser.parse_args()
    model_dir = args.output or (ROOT / "model")
    model_dir.mkdir(exist_ok=True, parents=True)
    catalog_path = model_dir / "catalog.json"
    catalog = json.loads((ROOT / "model" / "catalog.json").read_text(encoding="utf-8"))
    by_fold = {item["fold"]: item for item in catalog["models"]}
    for fold in args.folds:
        if fold not in by_fold:
            parser.error(f"Fold {fold} absent from catalog")
        relative = Path("attunet_cl_fixed") / "seed_42" / f"fold_{fold}"
        record_dir = (args.record_root or (args.study_root / "results" / "runs")) / relative
        checkpoint = (args.checkpoint_root / relative / "best_model.pth") if args.checkpoint_root else (record_dir / "best_model.pth")
        record = json.loads((record_dir / "complete.json").read_text(encoding="utf-8"))
        item = by_fold[fold]
        if record["config"] != "attunet_cl_fixed" or record["seed"] != 42 or record["fold"] != fold:
            raise ValueError(f"Training record mismatch for fold {fold}")
        if sha256(checkpoint) != record["best_model_sha256"]:
            raise ValueError(f"Checkpoint SHA-256 mismatch for fold {fold}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (payload["config"], payload["seed"], payload["fold"]) != ("attunet_cl_fixed", 42, fold):
            raise ValueError(f"Checkpoint metadata mismatch for fold {fold}")
        model = AttentionUNet().eval()
        model.load_state_dict(payload["model"], strict=True)
        destination = model_dir / item["filename"]
        example = torch.zeros(1, 1, 256, 256, dtype=torch.float32)
        torch.onnx.export(
            model, example, destination, export_params=True, opset_version=17,
            do_constant_folding=True, input_names=["image"], output_names=["probability"],
            dynamic_axes={"image": {0: "batch"}, "probability": {0: "batch"}}, dynamo=False,
        )
        onnx.checker.check_model(str(destination))
        test = np.random.default_rng(20260926 + fold).random((1, 1, 256, 256), dtype=np.float32)
        with torch.inference_mode():
            expected = model(torch.from_numpy(test)).numpy()
        session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
        actual = session.run(["probability"], {"image": test})[0]
        maximum_error = float(np.max(np.abs(expected - actual)))
        if maximum_error > 1e-4:
            raise AssertionError(f"ONNX conversion diverged for fold {fold}: {maximum_error}")
        item["sha256"] = sha256(destination)
        item["source_checkpoint_sha256"] = record["best_model_sha256"]
        item["threshold"] = record["selected_threshold"]
        item["conversion_max_absolute_error"] = maximum_error
        print(json.dumps({"fold": fold, "onnx_sha256": item["sha256"],
                          "threshold": item["threshold"], "conversion_max_absolute_error": maximum_error}))
        catalog_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
