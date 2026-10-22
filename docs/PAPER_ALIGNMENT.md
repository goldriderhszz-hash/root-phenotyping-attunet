# RootScope and the accompanying manuscript

Checked 22 October 2026 against the author's current Plant Methods manuscript, including the six-concern analysis revision and subsequent figure/text/format revision. This is an implementation alignment record, not a claim of state-of-the-art performance or journal acceptance.

## Audit finding and change

The previous desktop release used five single-channel fixed-loss Attention U-Net ONNX models and skeleton-derived descriptors. It did not execute the manuscript's whole-root reconstruction. RootScope now defaults to the manuscript's declared primary arm E (`whole_RGB_joint`). Historical software and its research comparison remain reproducible through `--pipeline legacy` and the unchanged historical release.

| Component | Current implementation | Evidence |
| --- | --- | --- |
| Segmentation | Five frozen D3 three-channel Attention U-Net exports | SHA-256, source checkpoint hashes and conversion errors in `model/paper/catalog.json` |
| Input | Reflection-canonical CLAHE grayscale, Lab-b, absolute local contrast; 256 px tiles, 128 px stride, Gaussian sigma 64 | `rootscope/paper/preprocessing.py` and `runtime.py` |
| Protected axis | Frozen whole-route policy, weak-support terminal continuation with original axis points preserved | `axis.py`, `terminal.py`, `geometry.py` |
| Candidates | Single edges plus four unlearned pairing explanations; protected-axis conflicts excluded | `candidates.py` |
| Quality and count | Five RGB quality forests, five geometric control forests, five count forests; 128 trees each | Frozen weights exported to numeric NPZ, never refitted |
| Decoder | Whole-path edge-exclusive set packing, protected axis forced, predicted-count deviation penalty; 8 s MILP bound | `decoder.py`; solver status and gap exported |
| Four descriptors | Smoothed operational axis length, geometric Y attachments, degree-3 junctions, parent-child projected angle | `geometry.py`; pixel units, missing values retained |

The quality targets are reciprocal contour-derived path support, fragment severity and mixing severity. They are not calibrated probabilities of anatomical correctness. Model files use numeric arrays loaded with `allow_pickle=False`; arbitrary user-supplied pickle files are not executed.

## Validation

`validation/paper_alignment.json` records forest predictions for every candidate in all 50 saved image pools, exact primary selections for 50/50 images, and graph/axis/candidate/feature replay on one image from each fold. Maximum forest error is below 1e-12. This replay starts from saved D3 probabilities. Tests also load and run all five D3 ONNX models. Actual photograph inference and packaged binary checks are recorded separately; changes between historical GPU/FP16 and ONNX CPU inference can change mask pixels and downstream topology.

These checks establish an implementation connection to the paper. They do not create a new controlled neural experiment, independent validation, an all-data production model, or evidence of current field-wide superiority.

## Scientific limits retained

The primary reconstruction matched 682 of 3290 reference roots (20.73%) at the declared 75% reciprocal support threshold; count MAE was 21.36. The 50 images have been reused for development. Reference paths are derived from polygons, not exact manual centerline traces. Geometric learning improved the within-pool control; the extra image-feature increment over geometry alone remains uncertain. No post-hoc replacement of the declared primary arm was made.

Independent external root-tracing comparison, controlled end-to-end evaluation, plant/bag/batch independence records, and measured human correction rates remain incomplete. Preserving an axis does not establish biological primary-root identity. Projected angles have known geometric limitations. Journal-inspired presentation cannot resolve these evidence gaps.

## Fold selection

Automatic mode matches study images by SHA-256 to their recorded image fold and loads that fold's D3 threshold, axis policy and quality/count models. Unknown images use fold 0 and are flagged for review. A manual fold is an explicit operator choice. The segmentation and reconstruction stages were not trained as a fully nested pipeline; selecting the recorded fold does not remove upstream development reuse.

## Deployment comparison

On `10-1-2(3).tif`, the ONNX CPU run differed from the historical saved probabilities by up to 0.01810; 142 thresholded pixels differed. The current decoder selected 50 paths versus 52 in the historical record. The packaged executable and current source gave exactly the same descriptors and selected paths on this photograph. See `validation/paper_full_image.json` and `paper_binary.json`. The backend/precision explanation is plausible, but has not been isolated by a paired original-backend experiment; it is not presented as a demonstrated cause.

The source GUI constructor passed. The desktop helper did not expose a confirmed application screenshot, so visual acceptance remains unverified.
