# Reproduce the reported records

Download from the [versioned research release](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/tag/paper-revision-20261020); check SHA256SUMS before extraction. Use an empty local directory. No private photographs are required for the first two tiers.

## 1. Frozen numerical evidence

Extract Additional_file_2_numerical.zip, install its pinned analysis requirements and run:

```text
python revision/replay_revision.py
python reconstruction/replay_reconstruction.py
```

This checks 600 segmentation and 2400 descriptor rows,60 run histories, the three distinct original interval procedures and 250 whole-root image/method rows. Tolerances remain1e-12for intervals and1e-10for descriptor summaries. Segmentation resamples 3 seed indices and 50 image indices independently (10000 draws,seed 20260920). Descriptor comparisons average seeds within an image and resample images within each of 5 fixed folds (seed 20260925). Deployment Dice resamples 50 own-fold image scores (seed 20260926). These are conditional descriptive intervals; historical p/Holm fields in old tables are not new significance evidence.

## 2. Legacy synthetic geometry

Extract Additional_file_3_legacy_geometry.zip and run `python replay_geometry.py`. It replays all 7 full masks and their horizontal reflections through the exact frozen extractor. Reproducing 54.16 degrees for a known 45 degree branch and 274.50/234.50 pixel returning-path outputs reproduces known failures; it does not certify the algorithm as correct. Do not substitute the later v16 code for this replay.

## 3. Images, figures and fixed-input reconstruction

Extract Study_sources_and_frozen_records_v20261020.zip into a new workspace. Extract the raw dataset archive into `研究资产/研究数据/` within that workspace, giving its `data/` subfolder. Optional D3/model weights extract into the same workspace. Original study-directory names preserve relative references. Instructions and SHA lists accompany each archive. Run each original study module from its own directory because its local `common.py` has a version-specific role.

The fixed D3 masks, protected paths, candidate pools, target matrices, selections and per-image results allow inspection of the exact reconstruction contrast. The 15 forests are 5 folds ×3 types; no all-image production model was fitted. All 50 images participated in development and the annotation-informed pool-feasibility oracle. Held-fold learner exclusion is not an independent full-pipeline or new-cohort test. The protected axis remains an input, not biological truth.

Figure programs in `论文新算法修订_20261019/work/` use frozen rows and actual image/path coordinates. EditablePPTX, vectorPDF/SVG and 600 dpi image masters are supplied separately with a figure hash manifest. The original controlled results remain separate from subsequent D3/reconstruction experiments. Numerical and synthetic replays were tested after extraction into a new directory; full retraining/inference was not rerun during manuscript preparation.

## 4. Historical application

The unchanged original RootScope Desktop code, executable and fiveONNXmodels remain at the rootscope-desktop software release and original commit `92e18aa37fece8f747548a38f73e382904814085`. Its interface and four result fields have not been replaced by the research prototype. Exact original image-to-model fold association and validation thresholds are necessary for its 50-image source comparison. Software test, model conversion, CPU timing, reconstruction and biological validity have different verification scopes.
