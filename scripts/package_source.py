"""Create a clean source archive suitable for a GitHub repository upload."""
from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parent / "RootScope-Desktop-Source.zip"
OMIT_DIRS = {".build-venv", ".venv", "_qa", "build", "dist", "release", "__pycache__", ".git",
             "validation_runs", "data", "cache", "oof_predictions"}
OMIT_FILES = {"build-log.txt", "final-build-log.txt", "RootScope.spec"}


def main() -> None:
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as package:
        for directory, folders, files in os.walk(ROOT):
            folders[:] = sorted(name for name in folders if name not in OMIT_DIRS)
            for name in sorted(files):
                if name in OMIT_FILES:
                    continue
                path = Path(directory) / name
                package.write(path, Path(ROOT.name) / path.relative_to(ROOT))
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    print(f"Source archive: {ARCHIVE}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
