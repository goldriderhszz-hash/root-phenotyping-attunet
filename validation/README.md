# Full-image deployment validation

The released ONNX graphs and the same `rootscope.batch.run_batch` path used by the desktop GUI and CLI were run on 50 full-resolution (`3209 × 2311`) rhizobag images. For fold *k*, only that fold's ten held-out test images were processed by `fold_k`, using the threshold chosen on its validation images. Each image appears once in the out-of-fold table. The model-selection and image assignments were locked before this run.

`deployment_seed42_oof.csv` contains image-level metrics and descriptors, research-pipeline comparison values, selected model, threshold, and SHA-256. `deployment_summary.json` reports macro and pooled scores and a fixed-seed, image-level bootstrap interval. The deployment-to-research differences measure ONNX/runtime agreement and are not a second independent accuracy estimate.

| Five-fold out-of-fold deployment, 50 full images | Result |
| --- | ---: |
| Image-level macro Dice | 0.58098 (95% image-bootstrap interval 0.56626–0.59516) |
| Image-level macro IoU | 0.41125 |
| Image-level macro hard clDice | 0.67032 |
| Pooled Dice | 0.58591 |
| Maximum per-image Dice difference from research implementation | 0.00046 |

The package also reports mean absolute differences between descriptors extracted from predicted masks and the same descriptors extracted from reference masks: dominant path length 1272.04 pixels, retained segment count 95.00, candidate junction region count 105.58, and mean local acute angle 7.36 degrees (50 evaluable images for each). These are agreement measures for the defined image-space algorithms, not ground-truth botanical trait accuracy.

Before inference, `scripts/audit_validation_inputs.py` verified all 50 TIFF file hashes, binary reference-mask pixel hashes, and image dimensions against the locked `research/dataset_manifest.csv`. The compact audit result is in `input_integrity.json`.

The actual packaged Windows `RootScope.exe` was also run in batch mode on one full-size held-out fold-0 image with its reference mask. [`packaged_binary_full_image.json`](packaged_binary_full_image.json) records the EXE, ONNX, input, and reference SHA-256 values. Its Dice and IoU matched the source deployment pipeline exactly for that image. [`windows_binary_self_test.json`](windows_binary_self_test.json) records successful loading and inference with all five bundled models. This single-EXE check establishes packaging and runtime parity for one image; the 50-image validation below evaluates the released ONNX application pipeline from source.

To rerun after obtaining the original TIFF images and reference PNG masks:

```powershell
python scripts/validate_deployment.py research path\to\image-and-mask-directory
```

The included `research/` directory contains `fold_assignments.json`, the locked input manifest, and the fixed-loss seed-42 research-pipeline reference metrics for parity checking. The original study checkpoints are external to the software repository. Raw images, masks, and full-size prediction images are deliberately absent while data deposition is handled separately. The script writes the raw prediction artifacts to `validation_runs/`, which Git ignores.

This validation measures performance on the same experimental collection that supplied training and validation folds, with strict held-out testing within each fold. It does not measure a final refit, a five-model ensemble, independent external images, other species, or other imaging systems. The default `fold_0` model's own held-out estimate is its 10-image fold-0 result, not the aggregate 50-image score.
