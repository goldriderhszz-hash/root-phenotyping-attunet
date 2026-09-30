# Output schema

Field names are identical in both interface languages. `results.json` stores nested descriptors and optional reference evaluation; CSV flattens scientific measurements. Non-finite values become missing. Failed images appear in `errors.csv` and are excluded from successful results, so report their denominators.

| Fields | Role |
| --- | --- |
| image_name, image_sha256, width, height | Input identity, hash and pixel dimensions |
| foreground_fraction, threshold | Foreground share and actual probability cutoff |
| dominant_path_length | Raw-skeleton path length in pixels |
| retained_segment_count, junction_region_count | Operational counts under the frozen extractor |
| mean_local_acute_angle_deg | Exploratory local acute angle in degrees |
| valid_angle_n, angle_failure_reason, path_status | Measurement availability and reasons |
| scale_status, length_unit, closing_kernel, pruning_iterations, min_component_pixels | Scale interpretation and extractor settings |
| qc_flags | Machine-readable review flags |
| eval_*, reference_*, error_* | Optional metrics, reference descriptors and prediction-minus-reference errors |

QC flags are `empty_prediction`, `very_low_foreground` (<0.001), `high_foreground` (>0.2), `touches_image_border`, `path_unavailable` and `angle_unavailable`. These are review heuristics, not calibrated error probabilities.

Provenance identifies the version, timestamps, model/fold/seed, ONNX/checkpoint hashes, default/used thresholds, input/reference hashes, processing settings, runtime provider and library versions. Per-image IDs, paths and timestamps can differ between equivalent runs. Private `_` keys are omitted from public result JSON. CSV uses UTF-8 BOM.
