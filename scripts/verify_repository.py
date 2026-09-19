"""Verify the dataset layout, frozen split, checkpoint, and model contract."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(64).startswith(b"version https://git-lfs.github.com/spec/v1")


def main() -> None:
    images = {p.stem: p for p in (ROOT / "data/images").glob("*.tif")}
    masks = {p.stem.removesuffix("_mask"): p for p in (ROOT / "data/masks").glob("*_mask.png")}
    annotations = {p.stem: p for p in (ROOT / "data/annotations").glob("*.json")}

    if not (len(images) == len(masks) == len(annotations) == 50):
        raise SystemExit(
            f"Expected 50 triples, found images={len(images)}, masks={len(masks)}, annotations={len(annotations)}"
        )
    if not (images.keys() == masks.keys() == annotations.keys()):
        raise SystemExit("Image, mask, and annotation identifiers do not match")

    split = json.loads((ROOT / "data/splits/seed42.json").read_text(encoding="utf-8"))
    train, test = set(split["train"]), set(split["test"])
    if len(train) != 40 or len(test) != 10 or train & test or train | test != set(images):
        raise SystemExit("Frozen split does not form a disjoint 40/10 partition of the dataset")

    checkpoint = ROOT / "models/combined_attention_unet_cldice.pth"
    if not checkpoint.is_file():
        raise SystemExit("Archived checkpoint is missing")

    pointer_count = sum(_is_lfs_pointer(p) for p in [*images.values(), *masks.values(), checkpoint])
    if pointer_count:
        raise SystemExit(f"{pointer_count} Git LFS objects are not materialized; run `git lfs pull`")

    try:
        from root_phenotyping import AttentionUNet
    except ImportError as error:
        raise SystemExit("Install the package first with `python -m pip install -e .`") from error

    model = AttentionUNet()
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameters != 7_981_277:
        raise SystemExit(f"Unexpected trainable parameter count: {parameters}")

    print("Repository verification passed")
    print("Dataset triples: 50")
    print("Frozen split: 40 train / 10 test")
    print(f"Trainable parameters: {parameters}")
    print(f"Checkpoint: {checkpoint.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
