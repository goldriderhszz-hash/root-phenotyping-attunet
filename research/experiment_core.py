from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from skimage.morphology import skeletonize
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("ROOTSCOPE_DATA_DIR", str(ROOT / "data"))).expanduser().resolve()
PATCH_SIZE = 256
TRAIN_STRIDE = 128


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Fixed seeds make the protocol repeatable. cuDNN benchmarking is retained for
    # practical runtime; exact bitwise identity can still vary across GPU/software.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read {path}")
    return image


def find_mask(name: str) -> Path:
    for candidate in (DATA_DIR / f"{name}_mask.png", DATA_DIR / f"{name}_mask.tif", DATA_DIR / f"{name}.png"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No mask for {name}")


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int, intermediate_channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(gate_channels, intermediate_channels, 1, bias=True),
            nn.BatchNorm2d(intermediate_channels),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(skip_channels, intermediate_channels, 1, bias=True),
            nn.BatchNorm2d(intermediate_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return skip * self.psi(F.relu(self.gate(gate) + self.skip(skip), inplace=True))


class UNet(nn.Module):
    def __init__(self, attention: bool):
        super().__init__()
        self.attention = attention
        self.pool = nn.MaxPool2d(2)
        self.d1 = DoubleConv(1, 32)
        self.d2 = DoubleConv(32, 64)
        self.d3 = DoubleConv(64, 128)
        self.d4 = DoubleConv(128, 256)
        self.bottleneck = DoubleConv(256, 512)
        if attention:
            self.a4 = AttentionGate(512, 256, 128)
            self.a3 = AttentionGate(256, 128, 64)
            self.a2 = AttentionGate(128, 64, 32)
            self.a1 = AttentionGate(64, 32, 16)
        self.u1 = DoubleConv(512 + 256, 256)
        self.u2 = DoubleConv(256 + 128, 128)
        self.u3 = DoubleConv(128 + 64, 64)
        self.u4 = DoubleConv(64 + 32, 32)
        self.out = nn.Conv2d(32, 1, 1)

    @staticmethod
    def up(x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))
        x4 = self.d4(self.pool(x3))
        b = self.bottleneck(self.pool(x4))
        z = self.up(b)
        s4 = self.a4(z, x4) if self.attention else x4
        z = self.u1(torch.cat([z, s4], dim=1))
        up3 = self.up(z)
        s3 = self.a3(up3, x3) if self.attention else x3
        z = self.u2(torch.cat([up3, s3], dim=1))
        up2 = self.up(z)
        s2 = self.a2(up2, x2) if self.attention else x2
        z = self.u3(torch.cat([up2, s2], dim=1))
        up1 = self.up(z)
        s1 = self.a1(up1, x1) if self.attention else x1
        z = self.u4(torch.cat([up1, s1], dim=1))
        return torch.sigmoid(self.out(z))


def dice_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = (prediction * target).sum()
    return 1.0 - (2.0 * intersection + 1.0) / (prediction.sum() + target.sum() + 1.0)


def soft_erode(image: torch.Tensor) -> torch.Tensor:
    vertical = -F.max_pool2d(-image, kernel_size=(3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-image, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def soft_dilate(image: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(image, kernel_size=3, stride=1, padding=1)


def soft_open(image: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(image))


def legacy_soft_skeleton(image: torch.Tensor, iterations: int = 12) -> torch.Tensor:
    work = image
    skeleton = torch.zeros_like(image)
    for _ in range(iterations):
        eroded = soft_erode(work)
        opened = soft_open(eroded)
        skeleton = torch.maximum(skeleton, eroded - opened)
        work = eroded
    return skeleton


def corrected_soft_skeleton(image: torch.Tensor, iterations: int = 12) -> torch.Tensor:
    # Original soft-skeleton recurrence used by the clDice reference implementation.
    opened = soft_open(image)
    skeleton = F.relu(image - opened)
    work = image
    for _ in range(iterations):
        work = soft_erode(work)
        opened = soft_open(work)
        delta = F.relu(work - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def corrected_cldice_loss(prediction: torch.Tensor, target: torch.Tensor, target_skeleton: torch.Tensor | None = None) -> torch.Tensor:
    smooth = 1e-5
    skeleton_prediction = corrected_soft_skeleton(prediction)
    skeleton_target = corrected_soft_skeleton(target) if target_skeleton is None else target_skeleton
    topology_precision = (skeleton_prediction * target).sum() / (skeleton_prediction.sum() + smooth)
    topology_sensitivity = (skeleton_target * prediction).sum() / (skeleton_target.sum() + smooth)
    return 1.0 - 2.0 * topology_precision * topology_sensitivity / (topology_precision + topology_sensitivity + smooth)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    attention: bool
    skeleton_weight: float
    bce_mode: str
    normalize_weights: bool = False
    transition_epoch: int = 25

    def weights(self, epoch: int) -> tuple[float, float, float]:
        if self.bce_mode == "fixed02":
            bce = 0.2
        elif self.bce_mode == "fixed04":
            bce = 0.4
        elif self.bce_mode == "schedule":
            bce = 0.2 if epoch <= self.transition_epoch else 0.4
        else:
            raise ValueError(self.bce_mode)
        weights = np.array([bce, 0.8, self.skeleton_weight], dtype=np.float64)
        if self.normalize_weights:
            weights /= weights.sum()
        return tuple(float(x) for x in weights)


CONFIGS = {
    "unet_pixel": ExperimentConfig("unet_pixel", False, 0.0, "fixed02"),
    "attunet_pixel": ExperimentConfig("attunet_pixel", True, 0.0, "fixed02"),
    "attunet_cl_fixed": ExperimentConfig("attunet_cl_fixed", True, 0.3, "fixed02"),
    "attunet_cl_schedule": ExperimentConfig("attunet_cl_schedule", True, 0.3, "schedule"),
}


def composite_loss(prediction: torch.Tensor, target: torch.Tensor, epoch: int, config: ExperimentConfig, target_skeleton: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    bce_value = F.binary_cross_entropy(prediction, target)
    dice_value = dice_loss(prediction, target)
    cl_value = corrected_cldice_loss(prediction, target, target_skeleton) if config.skeleton_weight else prediction.new_tensor(0.0)
    wbce, wdice, wcl = config.weights(epoch)
    total = wbce * bce_value + wdice * dice_value + wcl * cl_value
    return total, {
        "bce": float(bce_value.detach()), "dice_loss": float(dice_value.detach()),
        "cldice_loss": float(cl_value.detach()), "weight_bce": wbce,
        "weight_dice": wdice, "weight_cldice": wcl,
    }


class PatchDataset(Dataset):
    def __init__(self, names: list[str], seed: int):
        self.seed = seed
        self.epoch = 0
        self.images: dict[str, np.ndarray] = {}
        self.masks: dict[str, np.ndarray] = {}
        self.samples: list[tuple[str, int, int]] = []
        self.target_skeletons: np.ndarray | None = None
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        rng = random.Random(seed)
        for name in names:
            image = clahe.apply(read_gray(DATA_DIR / f"{name}.tif"))
            mask = (read_gray(find_mask(name)) > 127).astype(np.uint8)
            self.images[name] = image
            self.masks[name] = mask
            height, width = mask.shape
            for y in range(0, height - PATCH_SIZE + 1, TRAIN_STRIDE):
                for x in range(0, width - PATCH_SIZE + 1, TRAIN_STRIDE):
                    foreground = int(mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE].sum())
                    if foreground > 150 or (foreground > 30 and rng.random() > 0.5):
                        self.samples.append((name, y, x))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def prepare_target_skeletons(self, device: torch.device, batch_size: int = 64) -> None:
        cache_dir = ROOT / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        coordinate_digest = hashlib.sha256(json.dumps(self.samples, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
        cache_path = cache_dir / f"target_skeleton_seed{self.seed}_{coordinate_digest}.npy"
        if cache_path.exists():
            try:
                candidate = np.load(cache_path, mmap_mode="r")
                if candidate.shape == (len(self.samples), PATCH_SIZE, PATCH_SIZE):
                    self.target_skeletons = candidate
                    return
            except (OSError, ValueError):
                pass
        output = np.empty((len(self.samples), PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
        with torch.inference_mode():
            for start in range(0, len(self.samples), batch_size):
                subset = self.samples[start:start + batch_size]
                masks = np.stack([self.masks[name][y:y + PATCH_SIZE, x:x + PATCH_SIZE] for name, y, x in subset])[:, None]
                tensor = torch.from_numpy(masks.astype(np.float32)).to(device)
                skeletons = corrected_soft_skeleton(tensor).gt(0.5).byte().cpu().numpy()[:, 0]
                output[start:start + len(subset)] = skeletons
        temporary = cache_path.with_name(cache_path.stem + ".tmp.npy")
        np.save(temporary, output)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, cache_path)
        self.target_skeletons = np.load(cache_path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        name, y, x = self.samples[index]
        image = self.images[name][y:y + PATCH_SIZE, x:x + PATCH_SIZE].astype(np.float32) / 255.0
        mask = self.masks[name][y:y + PATCH_SIZE, x:x + PATCH_SIZE].astype(np.float32)
        if self.target_skeletons is None:
            raise RuntimeError("Call prepare_target_skeletons before constructing the DataLoader")
        target_skeleton = np.asarray(self.target_skeletons[index], dtype=np.float32)
        rng = random.Random((self.seed + 1) * 1_000_003 + self.epoch * 100_003 + index)
        rotation = rng.randrange(4)
        if rotation:
            image = np.rot90(image, rotation)
            mask = np.rot90(mask, rotation)
            target_skeleton = np.rot90(target_skeleton, rotation)
        if rng.random() > 0.5:
            image, mask = np.fliplr(image), np.fliplr(mask)
            target_skeleton = np.fliplr(target_skeleton)
        if rng.random() > 0.5:
            image, mask = np.flipud(image), np.flipud(mask)
            target_skeleton = np.flipud(target_skeleton)
        if rng.random() > 0.5:
            gamma = rng.uniform(0.8, 1.2)
            image = np.clip(np.power(image, gamma), 0.0, 1.0)
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)
        target_skeleton = np.ascontiguousarray(target_skeleton)
        return torch.from_numpy(image[None]).float(), torch.from_numpy(mask[None]).float(), torch.from_numpy(target_skeleton[None]).float()


def preprocess_full_image(name: str) -> tuple[np.ndarray, np.ndarray]:
    image = read_gray(DATA_DIR / f"{name}.tif")
    mask = read_gray(find_mask(name)) > 127
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    return clahe.apply(image).astype(np.float32) / 255.0, mask


def gaussian_window(size: int = PATCH_SIZE, sigma: float = 64.0) -> np.ndarray:
    axis = np.arange(size, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    centre = size // 2
    weight = np.exp(-((xx - centre) ** 2 + (yy - centre) ** 2) / (2.0 * sigma**2))
    return (0.1 + 0.9 * (weight - weight.min()) / (weight.max() - weight.min())).astype(np.float32)


@torch.inference_mode()
def sliding_predict(model: nn.Module, image: np.ndarray, device: torch.device, fusion: str = "gaussian", batch_size: int = 16) -> tuple[np.ndarray, float]:
    height, width = image.shape
    pad_h = (PATCH_SIZE - height % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - width % PATCH_SIZE) % PATCH_SIZE
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
    coordinates = [(y, x) for y in range(0, padded.shape[0] - PATCH_SIZE + 1, PATCH_SIZE // 2) for x in range(0, padded.shape[1] - PATCH_SIZE + 1, PATCH_SIZE // 2)]
    weight = gaussian_window() if fusion == "gaussian" else np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    total = np.zeros_like(padded, dtype=np.float32)
    weights = np.zeros_like(padded, dtype=np.float32)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for start in range(0, len(coordinates), batch_size):
        batch_coordinates = coordinates[start:start + batch_size]
        patches = np.stack([padded[y:y + PATCH_SIZE, x:x + PATCH_SIZE] for y, x in batch_coordinates])[:, None]
        tensor = torch.from_numpy(patches).to(device=device, dtype=torch.float32)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            prediction = model(tensor).float().cpu().numpy()[:, 0]
        for patch, (y, x) in zip(prediction, batch_coordinates):
            total[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += patch * weight
            weights[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += weight
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return (total / (weights + 1e-8))[:height, :width], elapsed


def confusion(prediction: np.ndarray, target: np.ndarray) -> tuple[int, int, int, int]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    tn = int(np.logical_and(~prediction, ~target).sum())
    return tp, fp, fn, tn


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def segmentation_metrics(probability: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, float | int]:
    binary = probability > threshold
    tp, fp, fn, tn = confusion(binary, target)
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    dice = ratio(2 * tp, 2 * tp + fp + fn)
    iou = ratio(tp, tp + fp + fn)
    skel_pred = skeletonize(binary)
    skel_target = skeletonize(target)
    tprec = ratio(np.logical_and(skel_pred, target).sum(), skel_pred.sum())
    tsens = ratio(np.logical_and(skel_target, binary).sum(), skel_target.sum())
    hard_cldice = ratio(2 * tprec * tsens, tprec + tsens)
    pred_components = int(label(binary, connectivity=2).max())
    target_components = int(label(target, connectivity=2).max())
    result: dict[str, float | int] = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision,
        "recall": recall, "dice": dice, "iou": iou, "hard_cldice": hard_cldice,
        "pred_components": pred_components, "target_components": target_components,
        "component_error": pred_components - target_components,
    }
    width = 2.0 * distance_transform_edt(target)
    bins = (("width_le3", 0.0, 3.0), ("width_3_6", 3.0, 6.0), ("width_6_12", 6.0, 12.0), ("width_gt12", 12.0, math.inf))
    for key, lower, upper in bins:
        support = np.logical_and(skel_target, np.logical_and(width > lower, width <= upper))
        denominator = int(support.sum())
        result[f"{key}_n"] = denominator
        result[f"{key}_recall"] = float(np.logical_and(support, binary).sum() / denominator) if denominator else float("nan")
    return result


def threshold_search(probabilities: dict[str, np.ndarray], targets: dict[str, np.ndarray], thresholds: Iterable[float]) -> tuple[float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        dice_values = []
        for name in sorted(probabilities):
            tp, fp, fn, _ = confusion(probabilities[name] > float(threshold), targets[name])
            dice_values.append(ratio(2 * tp, 2 * tp + fp + fn))
        rows.append({"threshold": float(threshold), "macro_dice": float(np.mean(dice_values))})
    # Prefer the lower threshold on exact ties, with no access to test results.
    best = max(rows, key=lambda row: (round(row["macro_dice"], 12), -row["threshold"]))
    return best["threshold"], rows


def count_model_complexity(model: nn.Module, input_shape: tuple[int, ...] = (1, 1, 256, 256)) -> dict[str, float | int]:
    macs = 0
    hooks = []

    def hook(module: nn.Conv2d, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        nonlocal macs
        batch, out_channels, out_h, out_w = output.shape
        kernel_h, kernel_w = module.kernel_size
        operations = batch * out_channels * out_h * out_w * (module.in_channels // module.groups) * kernel_h * kernel_w
        macs += int(operations)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(hook))
    model.eval()
    with torch.inference_mode():
        model(torch.zeros(input_shape))
    for item in hooks:
        item.remove()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return {"parameters": parameters, "model_size_mib_float32": parameters * 4 / 1024**2, "macs_per_256_tile": macs, "approx_flops_per_256_tile": 2 * macs}


def load_fold(fold: int) -> dict:
    payload = json.loads((ROOT / "fold_assignments.json").read_text(encoding="utf-8"))
    if fold not in range(5):
        raise ValueError(f"Fold must be 0..4, got {fold}")
    by_fold = {int(key): value for key, value in payload["folds"].items()}
    test = sorted(by_fold[fold])
    validation = sorted(by_fold[(fold + 1) % 5])
    train = sorted(name for key, values in by_fold.items() if key not in {fold, (fold + 1) % 5} for name in values)
    if not (len(train) == 30 and len(validation) == 10 and len(test) == 10):
        raise RuntimeError(f"Unexpected split sizes for fold {fold}: {len(train)}/{len(validation)}/{len(test)}")
    return {"fold": fold, "train": train, "validation": validation, "test": test}
