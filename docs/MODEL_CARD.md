# Model card

## Model

The archived checkpoint is an Attention U-Net binary segmenter with four encoder stages (32, 64, 128, and 256 channels), a 512-channel bottleneck, attention-gated skip connections, bilinear decoder upsampling, and 7,981,277 trainable parameters.

`models/combined_attention_unet_cldice.pth` stores the state dictionary used by the frozen-model robustness analysis. It is tracked with Git LFS.

## Training data and objective

Forty of the 50 rhizobag images were used for training. Training used overlapping 256 × 256 tiles, foreground-dependent tile retention, right-angle rotations, horizontal and vertical flips, and gamma augmentation. The composite objective was:

- epochs 1–25: 0.2 BCE + 0.8 Dice loss + 0.3 implemented skeleton loss;
- epochs 26–50: 0.4 BCE + 0.8 Dice loss + 0.3 implemented skeleton loss.

The skeleton regularizer is clDice-inspired but differs from the original soft-clDice implementation: it omits the initial un-eroded skeleton layer and accumulates skeleton responses by element-wise maximum.

## Inference

Full images are reflection-padded and processed with CLAHE. Predictions from overlapping 256 × 256 windows at stride 128 are fused with a Gaussian window (σ = 64 pixels, minimum weight 0.1) and thresholded at 0.53.

## Reported performance

On the frozen ten-image test set, the combined configuration achieved pooled Dice 75.06% and pooled IoU 60.07%. Mean image-level Dice was 74.83%. Exact centerline recall was 58.88% for annotated structures no more than 3 pixels wide. A 3 × 3 Gaussian blur with σ = 1 pixel reduced mean Dice by 13.18 percentage points.

These values describe one saved training run and one dataset. They do not quantify variation across retraining, cameras, acquisition batches, genotypes, growth conditions, or independent sites.

## Intended use

The checkpoint is intended for research reproduction and exploratory segmentation of images acquired under conditions similar to the released rhizobag dataset.

## Out-of-scope use

Do not use the model as a calibrated physical measurement instrument, as proof of anatomical root identity, or as a validated biological trait assay. Do not infer real-world length from the path multiplier 1.12. Independent calibration and anatomical reference measurements are required.

## Limitations

- Recovery of very narrow roots is limited.
- Blur, focus errors, and acquisition changes can substantially reduce performance.
- False gaps can split skeletons; false bridges can merge structures and create junctions.
- Training data contain a known duplicate-image annotation inconsistency.
- The threshold-selection provenance is incomplete; the records do not establish that 0.53 was selected independently of the test set.
