# Soybean rhizobag segmentation and root reconstruction

Code, data and analysis for **Segmentation and contour-guided root reconstruction in soybean rhizobag images**.

The study combines an Attention U-Net segmentation comparison with contour-guided reconstruction of individual root paths in soybean rhizobag photographs. The dataset contains 50 images and 3290 whole-root polygon annotations. Segmentation quality and the resulting root measurements are evaluated separately.

[Manuscript](manuscript/README.md) · [Data and models](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/tag/v1.0.0) · [Reproduce the analysis](docs/REPRODUCIBILITY.md) · [Desktop application](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/tag/rootscope-desktop)

## Repository structure

|Directory|Contents|
|---|---|
|`manuscript/`|Manuscript, figure index and machine-readable tables|
|`data/`|Dataset description and image/annotation manifest|
|`research/`|Segmentation protocol, fold assignments, training source and run records|
|`analysis/`|Segmentation and descriptor analyses; root reconstruction records|
|`rootscope/`|Desktop segmentation and skeleton-measurement package|
|`model/`|Five ONNX segmentation models and their metadata|
|`validation/`|Application checks, source/model comparisons and recorded CPU timing|
|`docs/`|Installation, usage, analysis reproduction and file descriptions|

## Data and model downloads

Download the dataset, study sources and selected reconstruction models from [release v1.0.0](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/tag/v1.0.0). The [file index](docs/FILES.md) describes each archive; [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) gives the dataset and model contents. Check `SHA256SUMS.txt` after downloading.

The image annotations identify whole-root regions. Lengths are reported in pixels; the dataset does not supply physical calibration or exact manually traced centerlines. The manuscript defines the evaluation units and the scope of the reported measurements.

## Reproduce the analysis

Extract `Additional_file_2_numerical.zip` into an empty directory, install the included analysis requirements, and run:

```text
python revision/replay_revision.py
python reconstruction/replay_reconstruction.py
```

For the synthetic geometry audit, extract `Additional_file_3_geometry.zip` and run `python replay_geometry.py`. Full instructions, dependencies and the relationship between the segmentation and reconstruction experiments are in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Desktop application

[RootScope Desktop](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/tag/rootscope-desktop) provides a Windows application for segmentation and skeleton-derived measurements. Extract the complete Windows package and run `RootScope.exe`. Source installation and batch commands are in the [user guide](docs/USER_GUIDE.md). The desktop package and the contour-guided reconstruction study have distinct implementations, documented in [the version map](docs/VERSIONS.json).

## Citation and contact

See [CITATION.md](CITATION.md) for the manuscript and dataset citation, and `CITATION.cff` for the desktop software. Correspondence: Rongsheng Zhu, rshzhu@126.com. Software issues can be reported through [GitHub Issues](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/issues).

## License

Software is distributed under the [MIT license](LICENSE), with third-party notices retained. Research images and annotations have separate [data terms](DATA_LICENSE.md).
