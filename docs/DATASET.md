# Dataset documentation

## Contents

The dataset contains 50 matched triples:

- `data/images/<id>.tif`: grayscale rhizobag image.
- `data/masks/<id>_mask.png`: binary semantic root mask, with root pixels at 255 and background at 0.
- `data/annotations/<id>.json`: LabelMe annotation used to produce the mask.

Readable image-mask pairs contain 3209 × 2311 pixels. The manuscript reports a median annotated foreground fraction of 1.94% and a range of 0.43–8.32%.

## Frozen split

`data/splits/seed42.json` records the exact 40-image training and 10-image test allocation. Filenames were sorted and then shuffled with Python's `random` module using seed 42. Allocation occurred before 256 × 256 tile extraction, so overlapping tiles from one source image stay in one split.

The split unit is a source filename, not a confirmed biological specimen. Specimen-level independence was not recorded and must not be inferred from the filenames.

## Annotation semantics

Annotations mark visible foreground root pixels. Reflections, paper folds, and paper texture are excluded. Touching or overlapping roots share one semantic foreground class. The masks do not encode root identity, branching order, biological junction identity, or time-series correspondence.

## Known integrity issue

The manuscript audit found one pair of byte-identical training images under different filenames with discordant masks (mask Dice 0.689). The pair remains in this release because it was part of the reported training history. The diagnostic row is distributed in `results/manuscript/duplicate_image_annotation_diagnostic.csv`. Future dataset versions should resolve the annotation discrepancy and define independent specimen identifiers before creating a new split.

## Git LFS

TIFF images and PNG masks are stored through Git LFS. A regular source archive downloaded without LFS objects may contain only pointer files. Use `git lfs pull` and run `python scripts/verify_repository.py` before analysis.

## Reuse limits

The source repository did not record an explicit data license. Public access does not itself grant reuse rights. The copyright holders must add a data license before third-party reuse or deposition in a long-term repository.
