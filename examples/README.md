# Synthetic installation example

The 256 × 256 image and binary mask were constructed with OpenCV line primitives.
They contain no study data. They test installation, input/reference matching and exports;
their overlap is **not** evidence of biological accuracy. The model is the default fold-0
checkpoint at its locked 0.47 validation threshold.

```bash
python cli.py examples/synthetic_root.png --references examples/synthetic_root_mask.png --output results
```

Compare numerical fields with `expected_output/results.csv` and the prediction mask
with `expected_output/prediction_mask.png`. Generated time stamps, run folder names,
absolute output paths and elapsed times are expected to differ. The manifest records
the input, reference, model and expected mask SHA-256 values.
