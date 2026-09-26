# Research release

This repository has one public software release, identified by the immutable `research-release` tag and its Git commit. Its native Windows application includes five seed-42, fixed-loss Attention U-Net ONNX fold models, model-specific thresholds, image segmentation, skeleton-derived image descriptors, batch export, and reference-mask scoring.

The release adds full-resolution out-of-fold deployment validation on 50 held-out images, an input-integrity audit, original-research pipeline parity, and a packaged-EXE full-image check. An earlier same-image comparison with RhizoVision Explorer 2.0.3 has been withdrawn after a [polarity audit](benchmark/POLARITY_AUDIT.md); no comparative advantage is claimed pending a corrected run. Validation limits are stated in [`validation/`](validation/README.md).

The repository's current tree is organized around software, models, methods, verification, and build instructions. The previous root-level image files and prototype scripts were removed from the active branch. Historical Git commits may remain accessible; the research dataset is handled separately from this software release.

