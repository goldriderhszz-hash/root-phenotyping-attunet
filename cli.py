"""Headless batch entry point for reproducible RootScope analyses."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rootscope.batch import IMAGE_SUFFIXES, run_batch
from rootscope.models import DEFAULT_MODEL_ID, catalog
from rootscope.i18n import set_language


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
    parser.add_argument("--model", choices=["auto"] + [f"paper_fold_{i}" for i in range(5)] + [item["id"] for item in catalog()["models"]], default="auto",
                        help="auto selects a known study image by SHA-256; new images use fold 0 with a review flag")
    parser.add_argument("--pipeline", choices=["paper", "legacy"], default="paper", help="Paper primary reconstruction or historical skeleton descriptors")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the selected model's validation threshold")
    parser.add_argument("--save-probability", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument("--language", choices=['en', 'zh'], default='en', help="Language of progress and diagnostic messages; exported field names remain stable")
    args = parser.parse_args(argv)
    set_language(args.language)
    try:
        images = image_paths(args.images)
        references = image_paths(args.references)
        if args.pipeline == 'legacy' and args.model == 'auto':
            args.model = DEFAULT_MODEL_ID
        if args.pipeline == 'legacy' and args.model.startswith('paper_'):
            raise ValueError('Use fold_0 through fold_4 with the legacy pipeline')
        if args.pipeline == 'paper' and args.model != 'auto' and not args.model.startswith('paper_'):
            raise ValueError('Use auto or paper_fold_0 through paper_fold_4 with the paper pipeline')
        result = run_batch(images, references, args.output, args.threshold,
                           args.save_probability, make_zip=not args.no_zip, model_id=args.model, pipeline=args.pipeline)
    except (OSError, ValueError) as error:
        parser.exit(2, f"RootScope: {error}\n")
    print(json.dumps({"folder": result["folder"], "archive": result["archive"],
                      "successful_images": len(result["results"]),
                      "failed_images": len(result["errors"])}, ensure_ascii=False))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
