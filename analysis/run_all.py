"""Run the portable manuscript analyses without retraining the models."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def call(script: str, *args: str) -> None:
    print("Running:", script, *args, flush=True)
    subprocess.run(
        [sys.executable, "-B", str(HERE / script), *args],
        check=True,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-gpu-robustness",
        action="store_true",
        help="Rerun the 40-condition CUDA inference experiment.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    call("segmentation_eval.py", *( ["--force"] if args.force else [] ))
    call("phenotype_eval.py", *( ["--refresh"] if args.force else [] ))
    if args.include_gpu_robustness:
        call("model_robustness.py", *( ["--refresh"] if args.force else [] ))


if __name__ == "__main__":
    main()
