# Changelog

## 1.1.0 — Five-fold software validation release

- Bundled the five seed-42, fixed-loss Attention U-Net ONNX fold models with per-model SHA-256 and validation-selected thresholds.
- Added model selection to the native desktop interface and CLI, with model and threshold provenance in every batch.
- Added full-resolution out-of-fold deployment validation, locked input-integrity audit, and comparison with the original research pipeline.
- Added a controlled same-image comparison with the official RhizoVision Explorer 2.0.3 Windows application.
- Organized the repository around source, models, research methods, validation, benchmark, license, and build instructions; removed the previous root-level data and prototype files from the current branch.

## 1.0.0 — Initial native desktop build

- First Windows desktop application and CLI with one fold-0 ONNX model, image segmentation, skeleton-derived descriptors, batch export, and reference-mask scoring.
