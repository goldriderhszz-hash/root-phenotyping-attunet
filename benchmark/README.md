# Controlled direct comparison with RhizoVision Explorer

The comparison uses the official [RhizoVision Explorer 2.0.3 Windows archive](https://doi.org/10.5281/zenodo.3747697). RootScope and RhizoVision Explorer produced binary segmentations for the same ten original-size fold-0 held-out test images and were scored against identical reference masks with the same Dice and IoU formulas. RootScope used its locked `fold_0` model and validation-selected threshold 0.47.

RhizoVision Explorer was run in **Whole root** mode, **Keep largest component** enabled, with its other preprocessing controls at their defaults and no ROI. The rhizobag images contain dark roots; its documented high-contrast input assumption requires bright roots. Each image was converted to grayscale and intensity-inverted (`255 - grayscale`) without geometric changes. A pilot on one fold-1 validation image established the polarity and threshold range. The subsequent grid of 50, 60, and 70 was fixed before evaluating the full ten-image fold-1 validation set. The threshold with highest validation macro Dice was 60. The fold-0 test images were not used for this choice. RVE saved its actual binary segmentation PNGs; the official application was executed, rather than a reimplementation of its thresholding routine.

| Held-out fold 0, 10 full images | RootScope 1.1.0 | RhizoVision Explorer 2.0.3 |
| --- | ---: | ---: |
| Macro Dice | 0.59845 | 0.47626 |
| Macro IoU | 0.42776 | 0.31673 |

| Functional dimension | RootScope | RhizoVision Explorer 2.0.3 |
| --- | --- | --- |
| Tested workflow | Native desktop GUI and headless batch CLI | Official Windows GUI batch mode |
| Segmentation | Learned, tiled Attention U-Net with fold-specific ONNX model | Image threshold and morphological options |
| Saved outputs used here | Binary mask, overlay, skeleton, provenance, image descriptors | Binary mask and feature CSV |
| Measurement units | Pixel-based descriptors; no physical calibration | Pixel or user-calibrated physical units |
| Intended input conditions | This study's rhizobag photographs | Predominantly high-contrast scanned or camera root images |

The paired mean Dice difference was 0.12219 (10,000-resample image bootstrap 95% percentile interval 0.07519–0.18695; two-sided Wilcoxon signed-rank *p* = 0.00195). These are results for this rhizobag image domain. RhizoVision Explorer was designed primarily for high-contrast scans and excavated root systems, and both applications use different trait definitions. The table supports a segmentation comparison under this protocol, not a claim of general superiority or interchangeable biological phenotypes. The sample is only ten images from one held-out fold, so wider external comparisons remain desirable.

## Reproduce

1. Obtain RVE 2.0.3 from its official archive. Do not redistribute its GPL binary within RootScope.
2. Run `python benchmark/prepare_rve_inputs.py path/to/original-images path/to/rve-work`. This writes only contrast-inverted validation-fold-1 and test-fold-0 images and a SHA-256 input manifest.
3. In RVE, select Whole root mode and Keep largest component. Set threshold 50, 60, and 70 in turn; batch-process the `rve_validation_fold0` directory, saving segmented PNGs into `rve_validation_t50`, `rve_validation_t60`, and `rve_validation_t70` under `rve-work`.
4. Process `rve_test_fold0` at the selected validation threshold into `rve_test_t60`, with segmented PNG output enabled.
5. Run `python benchmark/evaluate_rve_comparison.py path/to/original-images path/to/rve-work`. The scorer verifies the 10 RootScope fold-0 rows in [`validation/`](../validation/README.md), selects the RVE threshold from validation only, and writes the three tables in this directory.

The released benchmark tables contain measurements and metadata, not the original images or RVE output bitmaps. The RVE archive, its source, and its GPL license remain with its authors.

`rve_input_manifest.csv` and `rve_output_hashes.csv` give SHA-256 values for all 20 corrected inputs and 40 segmented outputs. The archived Windows ZIP had SHA-256 `c4822980cdbfbb213f88cd1d738d4e7423f4064ce926ae880b5401d58dc896fd` during this run. These records make the comparison auditable without including or relicensing the other program's binary.
