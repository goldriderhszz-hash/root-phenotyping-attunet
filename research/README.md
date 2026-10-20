# Frozen study source and records

This directory contains the original five-fold protocol, data-hash manifest, fold assignments, training source and all 60 configuration/seed/fold histories and selected-checkpoint summaries. Numerical paper evidence is in analysis/tables/. Five deployable seed-42 fixed-loss ONNX graphs are catalogued in model/.

Training used PyTorch 2.7.1 with CUDA 12.6, three seeds, 50 epochs, four configurations, image-before-tile splitting and validation-only checkpoint/threshold selection. The test fold is k, validation is (k+1) mod 5 and the other folds train.

The study corpus comprises 50 image/annotation pairs identified by the frozen
hash manifest. `dataset_selection.json` summarizes the active corpus and confirmed
biological conditions. Filenames do not establish plant or batch identity.

The data provider confirmed on 30 September 2026 that all 50 images are Wm82,
untreated and approximately seven days after germination, as stated in the original
manuscript. These conditions are provider-confirmed rather than independently
recoverable from TIFF/annotation tags. Plant/batch grouping and physical scale remain
unavailable.

For independent retraining install requirements-research.txt and set ROOTSCOPE_DATA_DIR. Work from a disposable dataset copy: prepare_protocol.py regenerates masks from annotations and writes them into that directory. Keep frozen metadata/hashes separately because preparation regenerates metadata too. Inspect run_cross_validation.py --help to select runs.

The original 50 TIFF photographs, LabelMe annotations, merged masks and all 600 frozen out-of-fold prediction masks are public downloads in the [English research release](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/tag/submission-en-20261020), rather than files stored in this Git tree. The release also provides five selected later D3 checkpoints and 15 reconstruction forests. The 60 controlled-training PyTorch checkpoints and intermediate optimizer states are not bundled; their frozen histories and prediction records are provided. The original five deployable ONNX graphs remain in model/. See [data availability](../DATA_AVAILABILITY.md) for archive contents and reuse terms.
