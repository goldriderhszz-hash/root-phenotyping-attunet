# RhizoVision Explorer comparison status

**The published 12.22 percentage point Dice advantage is withdrawn pending a corrected RhizoVision Explorer run.** The original comparison used the official [2.0.3 Windows archive](https://doi.org/10.5281/zenodo.3747697), but its input polarity and exported-mask interpretation were inconsistent with the software's segmentation convention. The archived numerical tables were removed from the active branch; they remain available in Git history for audit. See [the polarity audit](POLARITY_AUDIT.md).

The underlying RootScope full-image validation in [`validation/`](../validation/README.md) is a separate result. It does not establish a direct advantage over RhizoVision Explorer.

## Corrected comparison protocol

1. Obtain the official RhizoVision Explorer 2.0.3 Windows archive. Use the same fold-1 validation images and fold-0 held-out test images as the original comparison.
2. Run `python benchmark/prepare_rve_inputs.py path/to/original-images path/to/rve-work`. The prepared inputs are grayscale with the original intensity polarity: dark roots on a lighter background. Retain the input manifest.
3. In RhizoVision Explorer, use **Whole root**, **Keep largest component**, no ROI, and **Invert images = false**. Save segmented PNGs and `metadata.csv` for every batch.
4. Fix a threshold grid using fold-1 validation images only. Run every threshold in separate `rve_validation_t<value>` folders, then select the highest validation macro Dice. Run fold 0 only once at the selected threshold in `rve_test_t<value>`.
5. Run `python benchmark/evaluate_rve_comparison.py path/to/original-images path/to/rve-work --thresholds <values>`. The scorer checks the input polarity and RhizoVision metadata before scoring. It treats **black (0) in RhizoVision's saved segmentation as root**, converts it to a Boolean foreground mask, and compares it with the white-root reference masks.
6. Inspect representative input and exported-mask overlays, the per-image measurements, and hashes before restoring any comparative claim to this README or the manuscript.

The old grid of 50, 60, and 70 was selected for the incorrectly inverted images. It must not be reused automatically for the corrected inputs. No corrected test-fold effect size or confidence interval is reported here.
