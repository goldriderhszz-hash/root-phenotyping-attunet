# Limitations

The 50 untreated Wm82 seedling images describe an internal image collection. Plant identity, acquisition-batch grouping and physical scale are insufficient. External species/batch/developmental-stage transfer is untested.

Attention U-Net and skeleton supervision are established methods. This study contributes a controlled loss comparison, held-out error propagation through a common extractor and a traceable deployment; it does not introduce a new architecture or establish superiority over all root tools.

The same extractor processes prediction and annotation masks. Agreement therefore does not validate anatomy or shared extractor assumptions. Path length is not total root length; segments/junctions may include fragments or contacts. Local angle uses the first coordinate-ordered pixel within radius 8–15, or the final coordinate. A fixed-axis curved-geometry probe differed by approximately 15.88 degrees after coordinate reordering. Angles remain exploratory; the frozen rule was not retuned on test outcomes.

Fourteen of fifteen scheduled runs selected a checkpoint at or before epoch 25, before the coefficient increase at epoch 26, which also changed total nominal loss scale. No isolated late-stage effectiveness conclusion follows. Fold and seed repetition do not remove dependence or unknown biological independence.

The default is fold 0, with four other individually selectable models. Five-fold aggregate scores do not establish default-model accuracy on new images. No ensemble, anatomical calibration, mask editing or root tracking is implemented. Candidate verification is on the available Windows computer; minimum hardware and new macOS/Linux results are not asserted.
