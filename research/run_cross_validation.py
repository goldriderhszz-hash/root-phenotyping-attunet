from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from experiment_core import (
    CONFIGS,
    ROOT,
    PatchDataset,
    UNet,
    composite_loss,
    confusion,
    count_model_complexity,
    load_fold,
    preprocess_full_image,
    ratio,
    segmentation_metrics,
    set_seed,
    sha256_file,
    sliding_predict,
    threshold_search,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(payload: dict, path: Path, keep_previous: bool = False) -> None:
    """Write a checkpoint atomically and optionally retain the prior good file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    previous = path.with_name(path.stem + "_previous" + path.suffix)
    torch.save(payload, temporary)
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    if keep_previous and path.exists():
        if previous.exists():
            previous.unlink()
        os.replace(path, previous)
    os.replace(temporary, path)


def load_resumable_checkpoint(paths: list[Path], device: torch.device, expected: dict) -> tuple[dict | None, Path | None]:
    errors = []
    for path in paths:
        if not path.exists():
            continue
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            if any(checkpoint.get(key) != value for key, value in expected.items()):
                errors.append(f"{path.name}: metadata mismatch")
                continue
            return checkpoint, path
        except Exception as exc:  # fallback is specifically for interrupted/corrupt writes
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
    if errors:
        print("CHECKPOINT_FALLBACK " + " | ".join(errors), flush=True)
    return None, None


def validate_at_threshold(model: UNet, names: list[str], device: torch.device, threshold: float = 0.5) -> tuple[float, list[dict]]:
    rows = []
    for name in names:
        image, target = preprocess_full_image(name)
        probability, elapsed = sliding_predict(model, image, device, fusion="gaussian")
        # Checkpoint selection uses Dice only. The full skeleton, component and
        # width metrics are computed for the selected model in evaluate_split.
        tp, fp, fn, tn = confusion(probability > threshold, target)
        row = {
            "name": name,
            "seconds": elapsed,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "dice": ratio(2 * tp, 2 * tp + fp + fn),
        }
        rows.append(row)
    return float(np.mean([row["dice"] for row in rows])), rows


def evaluate_split(model: UNet, names: list[str], device: torch.device, threshold: float) -> tuple[list[dict], dict[str, np.ndarray], dict[str, np.ndarray]]:
    rows: list[dict] = []
    probabilities: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    for name in names:
        image, target = preprocess_full_image(name)
        probability, elapsed = sliding_predict(model, image, device, fusion="gaussian")
        probabilities[name] = probability
        targets[name] = target
        rows.append({"name": name, "fusion": "gaussian", "threshold": threshold, "seconds": elapsed, **segmentation_metrics(probability, target, threshold)})
    return rows, probabilities, targets


def pooled_summary(rows: list[dict]) -> dict[str, float | int]:
    tp = int(sum(row["tp"] for row in rows))
    fp = int(sum(row["fp"] for row in rows))
    fn = int(sum(row["fn"] for row in rows))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n_images": len(rows),
        "macro_dice": float(np.mean([row["dice"] for row in rows])),
        "macro_iou": float(np.mean([row["iou"] for row in rows])),
        "macro_hard_cldice": float(np.mean([row["hard_cldice"] for row in rows])),
        "pooled_precision": precision,
        "pooled_recall": recall,
        "pooled_dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "pooled_iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "mean_seconds": float(np.mean([row["seconds"] for row in rows])),
    }


def run_one(config_name: str, seed: int, fold: int, split: dict, epochs: int, validate_every: int, batch_size: int, dataset: PatchDataset, device: torch.device, benchmark_steps: int) -> None:
    config = CONFIGS[config_name]
    run_dir = ROOT / "results" / "runs" / config_name / f"seed_{seed}" / f"fold_{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    completion = run_dir / "complete.json"
    if completion.exists() and benchmark_steps == 0:
        print(f"SKIP complete fold={fold} config={config_name} seed={seed}", flush=True)
        return

    set_seed(seed)
    model = UNet(attention=config.attention).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 1
    best_val_dice = -1.0
    best_epoch = 0
    training_rows: list[dict] = []
    validation_rows: list[dict] = []
    latest_path = run_dir / "checkpoint_latest.pth"
    previous_path = run_dir / "checkpoint_latest_previous.pth"
    best_path = run_dir / "best_model.pth"
    if benchmark_steps == 0:
        checkpoint, loaded_path = load_resumable_checkpoint(
            [latest_path, previous_path],
            device,
            {"epochs": epochs, "config": config_name, "seed": seed, "fold": fold},
        )
        if checkpoint is not None:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            if "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_val_dice = float(checkpoint["best_val_dice"])
            best_epoch = int(checkpoint["best_epoch"])
            training_rows = checkpoint.get("training_rows", [])
            validation_rows = checkpoint.get("validation_rows", [])
            print(f"RESUME fold={fold} config={config_name} seed={seed} from={loaded_path.name} at epoch={start_epoch}", flush=True)

    if benchmark_steps:
        model.train()
        dataset.set_epoch(1)
        generator = torch.Generator().manual_seed(seed * 1000 + 1)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, generator=generator)
        torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
        started = time.perf_counter()
        completed_steps = 0
        for images, targets, target_skeletons in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            target_skeletons = target_skeletons.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prediction = model(images)
            # Keep the original sigmoid + BCE formulation. BCE and the iterative
            # skeleton loss are evaluated in float32 because BCELoss is not safe
            # under CUDA autocast.
            loss, _ = composite_loss(prediction.float(), targets.float(), 1, config, target_skeletons.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            completed_steps += 1
            if completed_steps >= benchmark_steps:
                break
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        report = {"fold": fold, "config": config_name, "seed": seed, "batch_size": batch_size, "steps": completed_steps, "seconds": elapsed, "seconds_per_step": elapsed / completed_steps, "estimated_minutes_50_epochs": elapsed / completed_steps * (len(loader) * 50) / 60, "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None}
        atomic_write_text(run_dir / "benchmark.json", json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        return

    overall_started = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        dataset.set_epoch(epoch)
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, generator=generator)
        totals = {"loss": 0.0, "bce": 0.0, "dice_loss": 0.0, "cldice_loss": 0.0}
        started = time.perf_counter()
        for images, targets, target_skeletons in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            target_skeletons = target_skeletons.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prediction = model(images)
            loss, parts = composite_loss(prediction.float(), targets.float(), epoch, config, target_skeletons.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(loss.detach())
            for key in ("bce", "dice_loss", "cldice_loss"):
                totals[key] += parts[key]
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        row = {
            "epoch": epoch,
            **{key: value / len(loader) for key, value in totals.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
            "weight_bce": config.weights(epoch)[0],
            "weight_dice": config.weights(epoch)[1],
            "weight_cldice": config.weights(epoch)[2],
        }
        training_rows.append(row)

        should_validate = epoch == 1 or epoch % validate_every == 0 or epoch == epochs
        if should_validate:
            model.eval()
            val_dice, val_images = validate_at_threshold(model, split["validation"], device, threshold=0.5)
            validation_rows.append({"epoch": epoch, "threshold": 0.5, "macro_dice": val_dice, "image_rows": val_images})
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_epoch = epoch
                atomic_torch_save({"model": model.state_dict(), "fold": fold, "config": config_name, "seed": seed, "epoch": epoch, "validation_macro_dice_at_0.5": val_dice}, best_path, keep_previous=True)
            checkpoint = {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch, "epochs": epochs, "fold": fold, "config": config_name, "seed": seed,
                "best_val_dice": best_val_dice, "best_epoch": best_epoch,
                "training_rows": training_rows, "validation_rows": validation_rows,
            }
            atomic_torch_save(checkpoint, latest_path, keep_previous=True)
        write_csv(run_dir / "training_log.csv", training_rows)
        atomic_write_text(
            run_dir / "run_state.json",
            json.dumps(
                {
                    "status": "checkpointed",
                    "fold": fold,
                    "config": config_name,
                    "seed": seed,
                    "last_complete_epoch": epoch,
                    "epochs": epochs,
                    "best_epoch": best_epoch,
                    "best_val_dice": best_val_dice,
                    "resume_from_epoch": epoch + 1,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        atomic_write_text(
            ROOT / "experiment_status.json",
            json.dumps(
                {
                    "status": "running",
                    "process_id": os.getpid(),
                    "fold": fold,
                    "config": config_name,
                    "seed": seed,
                    "last_complete_epoch": epoch,
                    "epochs": epochs,
                    "resume_from_epoch": epoch + 1,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        print(f"TRAIN fold={fold} config={config_name} seed={seed} epoch={epoch:02d}/{epochs} loss={row['loss']:.5f} val_best={best_val_dice:.5f} epoch_s={row['seconds']:.1f}", flush=True)

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    model.eval()
    val_rows, val_probabilities, val_targets = evaluate_split(model, split["validation"], device, 0.5)
    thresholds = np.round(np.arange(0.10, 0.901, 0.01), 2)
    selected_threshold, threshold_rows = threshold_search(val_probabilities, val_targets, thresholds)
    val_rows = [{"name": name, "split": "validation", "fold": fold, "config": config_name, "seed": seed, "fusion": "gaussian", "threshold": selected_threshold, "seconds": next(row["seconds"] for row in val_rows if row["name"] == name), **segmentation_metrics(val_probabilities[name], val_targets[name], selected_threshold)} for name in sorted(val_probabilities)]
    test_rows, test_probabilities, test_targets = evaluate_split(model, split["test"], device, selected_threshold)
    test_rows = [{"name": row["name"], "split": "test", "fold": fold, "config": config_name, "seed": seed, **{key: value for key, value in row.items() if key != "name"}} for row in test_rows]
    prediction_dir = ROOT / "oof_predictions" / config_name / f"seed_{seed}"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(test_probabilities):
        destination = prediction_dir / f"{name}.png"
        encoded = cv2.imencode(".png", ((test_probabilities[name] > selected_threshold) * 255).astype(np.uint8))[1]
        temporary = destination.with_name(destination.name + ".tmp")
        encoded.tofile(temporary)
        os.replace(temporary, destination)
    write_csv(run_dir / "training_log.csv", training_rows)
    write_csv(run_dir / "threshold_search.csv", threshold_rows)
    write_csv(run_dir / "validation_metrics.csv", val_rows)
    write_csv(run_dir / "test_metrics.csv", test_rows)
    atomic_write_text(run_dir / "validation_checkpoints.json", json.dumps(validation_rows, ensure_ascii=False, indent=2))
    metadata = {
        "fold": fold, "config": config_name, "config_values": config.__dict__, "seed": seed, "epochs": epochs,
        "batch_size": batch_size, "training_tiles": len(dataset), "best_epoch": best_epoch,
        "checkpoint_selection": f"highest validation macro Dice at threshold 0.5 evaluated every {validate_every} epoch(s); test fold not accessed",
        "threshold_selection": "highest validation macro Dice among thresholds 0.10 to 0.90 in increments of 0.01; lower threshold resolves exact ties",
        "selected_threshold": selected_threshold,
        "validation": pooled_summary(val_rows),
        "test": pooled_summary(test_rows),
        "complexity": count_model_complexity(UNet(attention=config.attention)),
        "best_model_sha256": sha256_file(best_path),
        "wall_seconds": time.perf_counter() - overall_started,
        "software": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__, "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
    }
    atomic_write_text(completion, json.dumps(metadata, ensure_ascii=False, indent=2))
    atomic_write_text(
        run_dir / "run_state.json",
        json.dumps({"status": "complete", "fold": fold, "config": config_name, "seed": seed, "last_complete_epoch": epochs}, ensure_ascii=False, indent=2),
    )
    for redundant in (previous_path, best_path.with_name(best_path.stem + "_previous" + best_path.suffix)):
        if redundant.exists():
            redundant.unlink()
    print(json.dumps({"complete": str(completion), "fold": fold, "config": config_name, "seed": seed, "threshold": selected_threshold, "test": metadata["test"]}, ensure_ascii=False), flush=True)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 3407, 2026])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--benchmark-steps", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = [name for name in args.configs if name not in CONFIGS]
    if unknown:
        raise ValueError(f"Unknown configs: {unknown}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    atomic_write_text(
        ROOT / "experiment_status.json",
        json.dumps({"status": "running", "process_id": os.getpid(), "folds": args.folds, "configs": args.configs, "seeds": args.seeds}, ensure_ascii=False, indent=2),
    )
    print(f"device={device} folds={args.folds} configs={args.configs} seeds={args.seeds}", flush=True)
    for fold in args.folds:
        split = load_fold(fold)
        for seed in args.seeds:
            dataset = PatchDataset(split["train"], seed=seed)
            dataset.prepare_target_skeletons(device)
            print(f"DATA fold={fold} seed={seed} train={len(split['train'])} validation={len(split['validation'])} test={len(split['test'])} training_tiles={len(dataset)}", flush=True)
            for config_name in args.configs:
                run_one(config_name, seed, fold, split, args.epochs, args.validate_every, args.batch_size, dataset, device, args.benchmark_steps)
    atomic_write_text(ROOT / "experiment_status.json", json.dumps({"status": "complete", "completed_runs": 60}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
