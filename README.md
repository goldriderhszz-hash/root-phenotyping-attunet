# Soybean rhizobag segmentation and contour-guided root reconstruction

This repository accompanies **Segmentation and contour-guided root reconstruction in soybean rhizobag images**. The [versioned research release](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/tag/submission-en-20261020) provides the editable English manuscript and supplementary information, eight figures and two pseudocode panels, numerical/synthetic replays, the original 50-image dataset, fixed intermediate records, source snapshots and selected models. The manuscript is an author-review research version, not an accepted journal article.

## Read and reproduce

- [English manuscript](manuscript/Manuscript_EN.md) · [Submission file index](SUBMISSION_FILES.md)
- [Manuscript/figure downloads](manuscript/README.md) · [Data and materials](DATA_AVAILABILITY.md)
- [Reproduction instructions](docs/REPRODUCIBILITY.md) · [Dataset manifest](data/dataset_manifest.csv)
- [Whole-root50-image paired records](analysis/reconstruction/whole_root_all250_image_method_rows.csv)
- [Version and evidence map](manuscript/VERSIONS.json) · [Research-data terms](DATA_LICENSE.md)

The original controlled comparison is 50 images ×4 configurations ×3 seeds (600 segmentation records,2400 descriptor records and 60 fits). Fixed skeleton loss increased Dice by 2.53 percentage points relative to pixel-loss Attention U-Net, but increased retained-segment and junction-region MAEs by 35.03 and 50.05. In a separate fixed-input reconstruction analysis,75%reciprocal-support one-to-one matches increased 586→682 of 3290 references and count MAE decreased 63.00→21.36. Mixed outputs decreased 674→562 but their output proportion increased 10.47%→16.82%. All 50 development images and regressions remain in the records. These are conditional internal results, not validated anatomical primary-root accuracy or external-cohort performance.

## Historical desktop application

The [RootScope Desktop release](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/tag/rootscope-desktop) remains the original grayscale segmentation/reference-descriptor implementation. Its software, five ONNX models, application interfaces and Git history are preserved. The newer whole-root research code is separate; publishing its records does not turn the old executable into an instance-reconstruction application. [Desktop guide](docs/USER_GUIDE.md), [model documentation](MODEL.md) and the original code remain available. Historical figure/table numbering is documented in the manuscript map, not silently reused for new figures.

Software retains MIT licensing. Original research images and annotations have provider-authorized public access; their separate terms are in DATA_LICENSE.md. The 50 images,3290 whole-root polygons and three training seeds are different statistical units; the three seeds do not create 150 independent plants. Length is in pixels, and root order, attachment identities and physical calibration are not independently established.
