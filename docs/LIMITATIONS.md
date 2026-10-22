# Current RootScope application

The application now defaults to the paper reconstruction pipeline. See [USER_GUIDE.md](USER_GUIDE.md), [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md) and [JOURNAL_DESIGN.md](JOURNAL_DESIGN.md) for current behavior, validation limits, models and outputs. The historical records below retain their original scope.

# Measurement scope

This page describes the grayscale desktop measurements and controlled segmentation experiment. The manuscript also defines the separate contour-guided reconstruction experiment and its evaluation. Dataset contents and access terms are described in [DATA_AVAILABILITY.md](../DATA_AVAILABILITY.md).

The 50 untreated Wm82 seedling images describe an internal image collection. Plant identity, acquisition-batch grouping and physical scale are insufficient. External species/batch/developmental-stage transfer is untested.

Attention U-Net and skeleton supervision are established methods. The segmentation experiment compares loss configurations and measures error propagation through a common extractor. It does not establish superiority over all root analysis tools.

The same extractor processes prediction and annotation masks. Agreement therefore does not validate anatomy or shared extractor assumptions. Path length is not total root length; segments/junctions may include fragments or contacts. Local angle uses the first coordinate-ordered pixel within radius 8–15, or the final coordinate. A fixed-axis curved-geometry probe differed by approximately 15.88 degrees after coordinate reordering. Angles remain exploratory; the frozen rule was not retuned on test outcomes.

Fourteen of fifteen scheduled runs selected a checkpoint at or before epoch 25, before the coefficient increase at epoch 26, which also changed total nominal loss scale. No isolated late-stage effectiveness conclusion follows. Fold and seed repetition do not remove dependence or unknown biological independence.

The default is fold 0, with four other individually selectable models. Five-fold aggregate scores do not establish default-model accuracy on new images. No ensemble, anatomical calibration, mask editing or root tracking is implemented. The graphical interface and packaged executable were evaluated on Windows. Automated source checks run on Windows, Linux and macOS; these do not validate physical displays or native desktop packages on Linux/macOS. Minimum hardware requirements have not been measured.

