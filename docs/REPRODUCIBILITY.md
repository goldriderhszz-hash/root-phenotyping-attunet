# Reproduce the manuscript

The study contains 50 images, four configurations, three seeds and five folds. Complete images were split before tiling; each rotation uses 30 training, 10 validation and 10 test images. Image-level splitting prevents tile leakage but does not establish plant-level independence.

## Regenerate summaries from included records

```powershell
python -m pip install -r requirements-analysis.txt
python analysis/reproduce_tables.py --output analysis/generated
python analysis/angle_geometry_audit.py
python analysis/make_descriptor_figures.py
python analysis/qa_analysis.py
python analysis/audit_uncertainty.py
python analysis/source_denominators.py
```

Inputs are `analysis/tables/`, `analysis/run_records.json` and `research/run_records/`. Outputs include Table 1 scores, seed-specific descriptor summaries, paired fold/seed effects, leave-one-fold-out summaries, Fig. 7 image-mean pairs/limits, geometric checks and figure PDFs/PNGs. All 600 segmentation and 2,400 descriptor rows are checked for matching and completeness.

Leave-one-fold-out omits existing test-fold records; no models are retrained. Descriptive intervals are conditional on these images, splits and fits. Folds share training images; three seeds do not create 150 independent plants. Difference limits use 50 per-image means across seeds and do not estimate agreement for one single checkpoint on new images.

## Verify deployment

**Resampling audit.** The recorded Dice intervals resample three seed indices and
50 image indices independently and average the paired Cartesian index product
(10,000 draws; RNG seed 20260920). Folds are held fixed in this calculation. Descriptor error differences average three seeds
within image, then sample ten images within each fixed fold (seed 20260925).
Deployment Dice samples 50 own-fold image scores (seed 20260926). Historical bootstrap
tail-fraction/Holm columns remain archival output, not calibrated p-values; no
significance claim is based on them. See `analysis/audit_uncertainty.py`.

`source_denominators.py` confirms 50 distinct images contribute reference points to
each width stratum. With `--data path\to\original-data`, it also reads recoverable
TIFF/annotation metadata. File display DPI is not botanical length calibration;
filename suffixes do not establish plant or acquisition-batch identities.

```powershell
python -m unittest discover -s tests -v
python desktop.py --self-test model-check.json
python scripts/validate_deployment.py research path\to\original-data --run-root validation_runs --report-root validation
python scripts/benchmark_cpu.py path\to\original-data
```

Five frozen seed-42 ONNX models and thresholds are catalogued in `model/`. Each study image must use its own held-out model. Aggregate deployment scores describe five separate models, not a default model or ensemble on new images. `validation/` identifies historical and version 1.1.0 executable checks. Synthetic examples test installation, not biological accuracy.

## Recompute analyses from masks

The research assets retained by the authors include 50 TIFFs, 50 annotation JSONs, 50 masks, 600 OOF prediction masks and 60 selected checkpoints. See [data availability](../DATA_AVAILABILITY.md) to request access. The commands below require those assets in the indicated directory structure.

```powershell
$env:ROOTSCOPE_DATA_DIR = 'path\to\archive\data'
$env:ROOTSCOPE_STUDY_DIR = 'path\to\archive'
$env:ROOTSCOPE_ANALYSIS_OUT = 'path\to\recomputed-analysis'
python analysis/compute_oof_descriptors.py --phase all
python analysis/paired_descriptor_effects.py
python analysis/compute_sensitivity.py --phase all
```

Inspect `--help` for phase choices. Each sensitivity comparison uses the same setting on prediction and reference. The 5/15/10 default is retained; test errors are not used to retune it. The optimized research extractor preserves frozen rules and ordering; the desktop uses the original extractor.

To re-export the five seed-42 fixed-loss checkpoints from the minimal archive while
preserving the locked ONNX files:

```powershell
python scripts/export_fold_models.py path\to\archive --checkpoint-root path\to\archive\checkpoints --record-root research\run_records --output results\reexported-models
```

Install `requirements-research.txt` for this optional conversion. Fresh exports may
have different graph bytes under another runtime; compare numerical predictions and
record their own hashes. The supplied ONNX models retain their recorded hashes.

## Retraining

Retained training source supports independent reproduction after obtaining the original data. Install `requirements-research.txt` and follow `research/README.md`. Use a disposable data copy: `prepare_protocol.py` regenerates masks and writes into the data directory. Original training used PyTorch 2.7.1 with CUDA 12.6; cuDNN benchmarking means seeds do not guarantee bitwise identity across hardware. Selected checkpoints support inference but do not contain every intermediate optimizer state.

| Manuscript item | Evidence |
| --- | --- |
| Table 1 and segmentation contrasts | all_oof_image_metrics.csv and reproduce_tables.py |
| Tables 2/3 and Fig. 7 | descriptor_comparison_long.csv and make_descriptor_figures.py |
| Fig. 8 | Complete 36-setting sensitivity records |
| Fig. 9 | Actual English interface, prespecified median-Dice image |
| Supplement fold robustness | leave_one_fold_out.csv and paired fold/seed effects |
| Supplement angle geometry | angle_geometry_audit.py and exploratory-angle records |
| Deployment and timing | validation/ and validation/timing scripts |

Retained original figure assets are mapped to their source records. Final diagrams without a generating script are not claimed to be fully regenerated from code.
