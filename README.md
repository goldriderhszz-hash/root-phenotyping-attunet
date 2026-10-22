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
|`rootscope/`|Application and paper reconstruction runtime|
|`model/`|Historical and D3 ONNX models; numeric reconstruction forests and metadata|
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

## RootScope application

**RootScope** now runs the manuscript's primary contour-guided whole-root reconstruction: five D3 segmentation models, frozen quality/count forests, protected-axis continuation and edge-exclusive global decoding. [Install and use](docs/USER_GUIDE.md) · [Implementation alignment and limits](docs/PAPER_ALIGNMENT.md) · [Research figure/report templates](docs/JOURNAL_DESIGN.md).

```text
python -m pip install -r requirements.txt
python desktop.py
python cli.py images --output results
```

The UI and exports report pixel units, missing values, model hashes and review flags. The name has no version suffix; archived tags and commit IDs remain for reproducibility. The previous [desktop release](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/tag/rootscope-desktop) contains the historical skeleton pipeline, which remains available with `--pipeline legacy`. Current Windows builds are available from this repository's Actions artifacts.

The paper's primary reconstruction matched 682/3290 roots (20.73%) at the declared 75% reciprocal path-support threshold on reused development images. This is a research prototype requiring review, not independently validated anatomical phenotyping. Journal-oriented presentation does not establish acceptance by Plant Methods.

## Citation and contact

See [CITATION.md](CITATION.md) for the manuscript and dataset citation, and `CITATION.cff` for the desktop software. Correspondence: Rongsheng Zhu, rshzhu@126.com. Software issues can be reported through [GitHub Issues](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/issues).

## License

Software is distributed under the [MIT license](LICENSE), with third-party notices retained. Research images and annotations have separate [data terms](DATA_LICENSE.md).
