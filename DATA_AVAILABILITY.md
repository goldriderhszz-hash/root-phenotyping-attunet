# Data and materials availability

The versioned [v1.1.0 release](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/tag/v1.1.0) provides the source, Windows application
and a separate numerical-record archive. The repository includes the evidence
needed to regenerate the reported numerical summaries without training.

| Material | Location and access |
| --- | --- |
| Application source, MIT license, pinned dependencies | This repository and source archive |
| Five seed-42 fixed-loss ONNX models, thresholds and SHA-256 | `model/`; also in the Windows application |
| Image hash manifest and locked folds | `research/dataset_manifest.csv`, `research/fold_assignments.json` |
| All 60 fit histories and selected-checkpoint summaries | `research/run_records/` |
| 600 segmentation and 2,400 descriptor records | `analysis/tables/` and numerical-record archive |
| Complete sensitivity records and analysis programs | `analysis/` and numerical-record archive |
| Deployment, CPU timing, executable and geometry checks | `validation/` |
| Manuscript figure assets and caption/hash mapping | `manuscript/` |
| 50 original TIFFs, annotations and reference masks | Retained by the authors; absent from this public repository |
| 600 out-of-fold prediction masks and 60 selected PyTorch checkpoints | Retained by the authors; absent from this public repository |

For access to original images, annotations, masks or research checkpoints, contact
the corresponding author at **rshzhu@126.com**. Access requests require a separate
data-owner decision; the repository does not grant image-data permissions.
The software MIT license covers software and associated documentation and does
not automatically license original research images. No dataset DOI or permanent
third-party archive identifier is claimed.

Included records support arithmetic replay, model inspection and installation
checks. Repeating image-level evaluations, annotation processing or training
requires the original assets. Filenames and hashes identify images; they do not
establish plant-level independence, acquisition batches or physical length scale.
See [reproducibility](docs/REPRODUCIBILITY.md) for inputs and limits at each level.
