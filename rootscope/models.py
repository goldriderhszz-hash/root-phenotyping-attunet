"""Locked deployment model catalog and per-fold provenance."""
from __future__ import annotations

import json
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
CATALOG_PATH = MODEL_DIR / "catalog.json"
DEFAULT_MODEL_ID = "fold_0"


def catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def model_spec(model_id: str = DEFAULT_MODEL_ID) -> dict:
    for item in catalog()["models"]:
        if item["id"] == model_id:
            if not item.get("sha256") or item.get("threshold") is None:
                raise ValueError(f"Model {model_id} is not an export-complete release artifact")
            return item
    raise ValueError(f"Unknown model: {model_id}")


def model_path(model_id: str = DEFAULT_MODEL_ID) -> Path:
    return MODEL_DIR / model_spec(model_id)["filename"]
