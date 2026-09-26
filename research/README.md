# Research code and metadata

This directory retains the five-fold study implementation used to train the four manuscript configurations, including the fixed soft-clDice Attention U-Net exported for RootScope. It includes the 50-image fold assignment, per-image SHA-256 manifest, fixed-loss seed-42 reference test metrics, training and threshold-selection scripts, descriptor extraction, and aggregation code. The immutable release models are catalogued in [`model/catalog.json`](../model/catalog.json).

To prepare or rerun the study after the raw images and annotations are deposited, set `ROOTSCOPE_DATA_DIR` to a directory containing each TIFF and matching JSON/PNG reference file, then run the scripts from this directory. For example, in PowerShell:

Use a working copy of the data: `prepare_protocol.py` regenerates PNG masks from JSON annotations and writes them into `ROOTSCOPE_DATA_DIR` while retaining backups under `research/mask_cache_before_regeneration/`.

```powershell
$env:ROOTSCOPE_DATA_DIR = 'D:\path\to\study-data'
python research/prepare_protocol.py
python research/run_cross_validation.py --help
```

The code writes caches, checkpoints, and result tables under `research/`. These generated directories are excluded from Git. `run_config.json` records the experimental settings. The fold assignment rule uses foreground ratio stratification; the validation fold is `(test_fold + 1) mod 5`. Each model was trained on 30 images, selected on 10, and tested on 10. The original runs used three seeds for the manuscript comparisons; the released ONNX package contains the five seed-42 fixed-loss models only.

This repository currently provides metadata and software rather than the raw image collection. `dataset_manifest.csv` holds file names and hashes so a later dataset archive can be matched exactly. Published deployment and direct-comparison results are in [`validation/`](../validation/README.md) and [`benchmark/`](../benchmark/README.md).
