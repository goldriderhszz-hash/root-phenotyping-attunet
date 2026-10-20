# Data and materials availability

The datasets supporting the manuscript are available in the [fixed research release `submission-en-20261020`](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/tag/submission-en-20261020). No login or contact request is required to download the public release. SHA256 checksums and dataset/figure manifests identify the exact files.

|Resource|Contents|
|---|---|
|Soybean_rhizobag_50_images_LabelMe_v20261020.zip|50 original TIFFs,50 LabelMe annotation JSONs and 50 merged masks;3290 whole-root polygons|
|Study_sources_and_frozen_records_v20261020.zip|Actual study-source snapshots, all 600 original prediction masks, frozen D3 masks, protected axes, candidates, features, targets and decoder records|
|D3_and_reconstruction_models_v20261020.zip|5 selected D3 PyTorch checkpoints and 15 whole-root forests (5 folds ×3 forest types)|
|Additional_file_2_numerical.zip|600 segmentation/2400 descriptor rows,60 training histories,150 CPU timing rows,250 reconstruction rows and numerical replays|
|Additional_file_3_legacy_geometry.zip|All 7 synthetic full masks, exact legacy extractor and original/reflection replay, including known failures|

The data provider's permission to publicly distribute the original photographs and annotations was confirmed on 20 October 2026. Data copyright and reuse terms are separate from MIT software licensing; see [DATA_LICENSE.md](DATA_LICENSE.md). The public image archive contains the same 50 study photographs, not the separate 49 unannotated images. Existing plant/bag/batch identities and physical scale were not recoverable; do not infer them from filenames or treat source images as verified independent plants. Polygons are whole-root regions, not exact manual centerlines or parent-child annotations.

Raw-image access, numerical replay and full training are distinct reproduction tiers. The selected D3 checkpoints and forests support the stated reconstruction version; the original 60 controlled-training checkpoints and intermediate D1 weights are not bundled as duplicate binaries. All 600 frozen controlled prediction masks and the corresponding training protocols/logs are provided. Replaying frozen measurements does not require retraining. The original desktop software/fiveONNXmodels are preserved under [their historical release](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/tag/rootscope-desktop). No new biological validation is implied by public availability.
