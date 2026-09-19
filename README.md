# Root segmentation and skeleton measurements in rhizobag seedlings

This repository accompanies the Plant Methods manuscript **“Root segmentation and skeleton measurements in rhizobag seedlings using Attention U-Net and two-stage loss weighting.”** It contains the 50-image rhizobag dataset, LabelMe annotations, binary masks, the archived combined-model checkpoint, the training and inference implementation, the skeleton-based descriptor extractor, and the numerical outputs used in the manuscript.

The manuscript reports a pooled Dice score of 75.06% and IoU of 60.07% on the frozen ten-image test set. The four downstream outputs are image-derived descriptors, not independently validated biological traits: dominant-axis path index, retained segment count, junction-region count, and mean local acute angle.

## Repository layout

```text
data/
  images/          50 grayscale TIFF rhizobag images (Git LFS)
  masks/           50 binary root masks (Git LFS)
  annotations/     50 LabelMe JSON annotations
  splits/          frozen source-image split used by the manuscript
src/root_phenotyping/
  pipeline.py      Attention U-Net, two-stage loss, training, and inference
  phenotypes.py    skeletonization and four mask-derived descriptors
  gui.py           reconstructed PyQt interface for single-image analysis
  onnx_inference.py  ONNX Runtime sliding-window inference
software/           auditable desktop source, archived ONNX model, build recipe
analysis/           manuscript evaluation scripts
models/             archived combined-model checkpoint (Git LFS)
artifacts/ablation/  four archived checkpoints, predictions, and training logs
results/manuscript/ machine-readable manuscript result tables
docs/               data, model, and reproducibility documentation
```

## Installation

Python 3.12 was used for the reported timing experiment. Create an isolated environment and install the package:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

CUDA-enabled PyTorch must match the local CUDA driver. The reported run used PyTorch 2.7.1 with CUDA 12.6; see [`requirements-lock.txt`](requirements-lock.txt) for the recorded environment.

Clone with Git LFS so that images, masks, and the checkpoint are materialized:

```bash
git lfs install
git clone https://github.com/goldriderhszz-hash/root-phenotyping-attunet.git
cd root-phenotyping-attunet
git lfs pull
```

## Reproduce the manuscript workflow

Verify the dataset, frozen split, checkpoint, and architecture:

```bash
python scripts/verify_repository.py
```

Train the combined model and predict the frozen test set:

```bash
python -m root_phenotyping.pipeline \
  --data-dir data/images \
  --mask-dir data/masks \
  --output-dir runs/manuscript \
  --epochs 50 --batch-size 8 --threshold 0.53
```

Extract the four mask-derived descriptors from the generated predictions:

```bash
python -m root_phenotyping.phenotypes \
  --ground-truth-dir runs/manuscript/ground_truth \
  --prediction-dir runs/manuscript/predictions \
  --output-dir runs/manuscript/phenotypes
```

The exact split and principal hyperparameters are frozen in [`data/splits/seed42.json`](data/splits/seed42.json) and [`configs/manuscript.yaml`](configs/manuscript.yaml). Evaluation details and expected outputs are described in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

Recompute the CPU-based manuscript tables from the archived predictions:

```bash
python analysis/run_all.py
```

Add `--include-gpu-robustness` to rerun the CUDA perturbation and timing analysis.

## Data and interpretation boundaries

- The split is by source filename before tile extraction: 40 training images and 10 test images.
- Specimen-level independence was not established in the recorded metadata.
- One byte-identical training-image pair has discordant annotations (mask Dice 0.689). It is retained because it was part of the reported training history; see [`results/manuscript/duplicate_image_annotation_diagnostic.csv`](results/manuscript/duplicate_image_annotation_diagnostic.csv).
- The path multiplier 1.12 is an implementation constant, not a pixel-to-length calibration.
- Count and angle outputs are properties of two-dimensional masks. They do not resolve root identity, crossing versus branching, or physical units.
- The saved checkpoint represents one training run. The manuscript confidence intervals describe variation across the ten test images, not variation across retraining.

## Archived checkpoint

`models/combined_attention_unet_cldice.pth` is the checkpoint used by the frozen-model robustness analysis. Model assumptions, intended use, and limitations are documented in [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

## Desktop software

The authors' `AI_Phenotype_Tool.zip` has been reconstructed as reviewable source under [`software/`](software/README.md). The original opaque PyInstaller bundle is not duplicated in the submission tree; its ONNX model is retained byte-for-byte with provenance and checksums. Follow the software-specific installation instructions, then run `python -m root_phenotyping.gui`. A reproducible Windows build recipe is provided for a post-review GitHub Release.

## Citation

Use [`CITATION.cff`](CITATION.cff) to cite the software and dataset. Replace the manuscript-status fields with the final DOI and publication metadata after acceptance.

## License

No open-source or data license was recorded in the source repository. The copyright holders must select and add appropriate code and data licenses before archival publication or third-party reuse.
