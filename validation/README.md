# Software verification

## Application and executable checks

- `unit_tests.txt`: application behavior, five model artifacts, language parity and error paths.
- `windows_model_check.json`: packaged executable loading and predicting with five models.
- `binary_parity.json`: one full held-out fold-1 image and a synthetic installation example compared between source and executable; inputs, outputs and executable are identified by SHA-256.
- `gui_checks.json`: interface operation, idle language switching and minimum-window layout at simulated Tk scaling factors. These are not separate physical-display trials.
- `analysis_audits/`: completeness, uncertainty, width-stratum denominator and constructed-angle checks.
- `fresh_protocol_check.json`: directory preparation on 50 miniature synthetic image/annotation fixtures, separate from the biological dataset.
- `gaussian_definition_audit.json`: agreement of the research and deployment Gaussian windows with the mathematical definition.
- `cpu_timing/`: source ONNX CPU timing on 50 own-fold images, three passes; model loading, tile warm-up, reference evaluation and file export excluded.

## Full-image source assessment

The ONNX graphs and shared `rootscope.batch.run_batch` pipeline were evaluated on
50 full-resolution (3209 × 2311) rhizobag images. For fold k, its ten held-out
images were processed by `fold_k` with the validation-selected threshold. Each
image appears once in `deployment_seed42_oof.csv`. `deployment_summary.json`
reports macro and pooled scores and an image-level bootstrap interval.

| Five-fold out-of-fold source assessment | Result |
| --- | ---: |
| Macro Dice | 0.58098 (95% image-bootstrap interval 0.56626–0.59516) |
| Macro IoU | 0.41125 |
| Macro hard clDice | 0.67032 |
| Pooled Dice | 0.58591 |
| Maximum per-image Dice difference from research implementation | 0.00046 |

Mean absolute descriptor differences relative to the same extractor applied to
reference masks were 1272.04 pixels for dominant path length, 95.00 for retained
segments, 105.58 for candidate junction regions and 7.36 degrees for local acute
angle. These are agreement measures for image-space algorithms, not independently
validated anatomical traits. The input audit verifies the 50 TIFF hashes,
reference-mask pixel hashes and dimensions against the locked manifest.

The one-image executable comparison establishes packaging/runtime consistency for
that image. The 50-image source assessment estimates internal out-of-fold
performance for five separately selected models. Neither establishes external
accuracy, a final refit or ensemble accuracy, or the default model's performance
on all 50 images. Runtime rebuilds must be identified by their own executable hash.

After obtaining original images and reference masks:

```powershell
python scripts/validate_deployment.py research path\to\image-and-mask-directory
```

Original assets are absent from this repository. See [data availability](../DATA_AVAILABILITY.md).
