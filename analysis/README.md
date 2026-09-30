# Manuscript analysis

Frozen numerical evidence is in `tables/`. `reproduce_tables.py` checks all
pairings/counts and regenerates statistics, fold robustness and difference limits.
`angle_geometry_audit.py` executes the frozen angle rule on constructed geometry.
`make_descriptor_figures.py` exports figures with source records.

`audit_uncertainty.py` replays the recorded seed-and-image segmentation resampling.
Recorded tail-fraction/Holm columns are retained as computational output;
they are not calibrated hypothesis-test p-values and support no significance claim.
`source_denominators.py` audits width-stratum contributions and optionally reads
existing image metadata. Verification records are in `validation/analysis_audits/`.

Mask-analysis scripts use `ROOTSCOPE_DATA_DIR`, `ROOTSCOPE_STUDY_DIR` and
`ROOTSCOPE_ANALYSIS_OUT` for portable execution. External masks/predictions are
needed for recomputation; included-record summaries run without them.
See [reproducibility](../docs/REPRODUCIBILITY.md).
