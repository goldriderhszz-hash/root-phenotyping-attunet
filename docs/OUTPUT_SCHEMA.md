# RootScope outputs

The paper pipeline exports `axis_curve_length_px`, `reconstructed_path_count`, `first_order_lateral_count` (geometric Y attachments), `branch_junction_count` (degree 3), `ambiguous_contact_count`, `mean_parent_child_angle_deg`, `valid_angle_n`, and `missing_angle_n`. Missing values are JSON null / blank CSV cells; reports display NA. Do not interpret blank as zero.

Results identify image hash, fold, threshold, ONNX hash, assignment method and review flags. Nested reconstruction records include selected candidate IDs, paths/curves in y,x pixel coordinates, graph edges, model error-target predictions, extension metadata, solver status and gap. Status 0 indicates a solver optimum; feasible time-limited incumbents are explicitly flagged. Boolean identity-validation fields remain false.

`provenance.json` includes the complete model catalog and software dependencies. `Report.md` gives a human-readable summary; `reconstruction.svg` embeds the source photograph and editable vector curves. PNG dpi metadata does not change source resolution. No physical calibration is inferred.

With `--pipeline legacy`, the historical fields retain their original definitions: dominant raw-skeleton path, processed segments, junction regions and local acute angle. These historical quantities must not be pooled with the paper fields merely because their labels are similar.
