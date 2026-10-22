# RootScope analysis report

## Study and input

Record the scientific question, species, acquisition conditions, plant/bag/batch identifiers, biological unit, image selection, annotations and actual physical calibration. Mark unknown facts as unknown.

## Implementation

Use `provenance.json` for the exact software commit, model hashes, folds, thresholds, preprocessing, solver status and dependencies. Name the analysis pipeline and its connection to the manuscript.

## Results

Use `results.csv` for per-image observations and `results.json` for complete geometry. Report missingness and failed images. Length is in pixels unless an independently verified calibration is available. Retained paths, Y attachments and graph junctions are operational quantities. Define the sampling unit and uncertainty method before reporting intervals.

## Figure legend

Provide a concise title and explain every panel, colour, unit, image-selection rule, scale and uncertainty symbol. Orange: protected operational axis; blue: other retained paths. Report actual sample sizes. Contour-derived reference compatibility is not anatomical ground truth.

## Availability and requirements

Project: RootScope. Home: https://github.com/goldriderhszz-hash/root-phenotyping-attunet . Record the actual archived commit/release identifier. Windows desktop; Python 3.12 source; dependencies in requirements.txt. MIT software; data permissions remain separate. Do not invent an archive DOI.

## Limitations and review

Describe internal development reuse, unvalidated biological identity and external-test gaps. Record actual operator corrections if measured. Automatic flags do not measure correction rate.
