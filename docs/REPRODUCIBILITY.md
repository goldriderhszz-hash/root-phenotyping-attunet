# Reproducibility guide

## Scope

This repository supports three reproducibility levels:

1. Inspect the exact data, split, code, checkpoint, and machine-readable reported results.
2. Re-evaluate the archived predictions and checkpoint with the scripts in `analysis/`.
3. Retrain the combined model from the frozen 40-image training split and regenerate ten test predictions.

The manuscript's baseline configurations were archived as predictions and summary outputs, but complete baseline training specifications were not available. The repository therefore does not claim a fully controlled reconstruction of every comparator.

## Environment

The timing run used Windows 11, Python 3.12.13, PyTorch 2.7.1 with CUDA 12.6, and an NVIDIA GeForce RTX 4060 Laptop GPU with 8188 MiB memory. The CPU was reported as AMD64 Family 25 Model 116. Timing excludes startup, model loading, warm-up, reference loading, evaluation, human inspection, and file output.

Install Git LFS before cloning. Then install the package and verify the release:

```bash
git lfs pull
python -m pip install -e .
python scripts/verify_repository.py
```

## Training and prediction

```bash
python -m root_phenotyping.pipeline --data-dir data/images --mask-dir data/masks --output-dir runs/manuscript --epochs 50 --batch-size 8 --threshold 0.53
```

The command sorts source filenames, shuffles with seed 42, trains on the first 40, and predicts the remaining ten. The expected test names are recorded in `data/splits/seed42.json`. The final epoch is saved; the reported run did not use a separate validation loader.

## Descriptor extraction

```bash
python -m root_phenotyping.phenotypes --ground-truth-dir runs/manuscript/ground_truth --prediction-dir runs/manuscript/predictions --output-dir runs/manuscript/phenotypes
```

The same extractor is applied to prediction and annotation masks. Agreement therefore measures propagation of segmentation differences through fixed extraction rules; it is not independent biological validation.

## Manuscript analyses

The portable scripts under `analysis/` generate image-level segmentation metrics, paired bootstrap intervals, width-stratified recall, descriptor agreement, perturbation sensitivity, and timing outputs from the organized repository layout. Run `python analysis/run_all.py` for the CPU analyses. Add `--include-gpu-robustness` to rerun frozen-model inference on a CUDA GPU. Machine-readable outputs used in the manuscript are retained under `results/manuscript/`.

## Expected headline values

| Quantity | Expected value |
|---|---:|
| Combined pooled Dice | 75.06% |
| Combined pooled IoU | 60.07% |
| Combined mean image Dice | 74.83% |
| Mean Dice difference vs base U-Net | +0.46 percentage points |
| Exact centerline recall at width ≤3 px | 58.88% |
| Blur σ=1 px mean Dice change | -13.18 percentage points |
| Mean measured processing time | 8.103 s/image |

Small differences may occur across library versions, hardware, and nondeterministic CUDA kernels. The archived result tables are the record for the manuscript.
