"""Headless batch entry point for reproducible RootScope analyses."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rootscope.batch import IMAGE_SUFFIXES, run_batch
from rootscope.models import DEFAULT_MODEL_ID, catalog


def image_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for value in inputs:
        source = Path(value).expanduser()
        candidates = sorted(source.iterdir()) if source.is_dir() else [source]
        for path in candidates:
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                key = str(path.resolve()).casefold()
                if key not in seen:
                    seen.add(key)
                    paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RootScope native batch analysis")
    parser.add_argument("images", nargs="+", help="Image files or directories")
    parser.add_argument("--references", nargs="*", default=[], help="Reference masks or directories")
    parser.add_argument("--output", type=Path, required=True, help="Output parent directory")
    parser.add_argument("--model", choices=[item["id"] for item in catalog()["models"]], default=DEFAULT_MODEL_ID,
                        help="Locked seed-42 fold model; each has its own validation-selected threshold")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the selected model's validation threshold")
    parser.add_argument("--save-probability", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args(argv)
    try:
        images = image_paths(args.images)
        references = image_paths(args.references)
        result = run_batch(images, references, args.output, args.threshold,
                           args.save_probability, make_zip=not args.no_zip, model_id=args.model)
    except (OSError, ValueError) as error:
        parser.exit(2, f"RootScope: {error}\n")
    print(json.dumps({"folder": result["folder"], "archive": result["archive"],
                      "successful_images": len(result["results"]),
                      "failed_images": len(result["errors"])}, ensure_ascii=False))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
