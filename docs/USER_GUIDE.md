# RootScope Desktop user guide

This guide describes RootScope Desktop from image import to traceable exports.

## Import and configure

Use **Select images**, **Import folder** or drag-and-drop. Supported formats are TIFF, PNG, JPEG and BMP. Folder import includes files at the selected level, not nested directories. Duplicate file paths are ignored. **Clear** clears inputs while idle.

If source images and reference-mask PNGs share a directory, select only the source
files: whole-folder import includes every supported image, including mask PNGs.
Reference matching uses stems; provide an unambiguous matching mask for each source.

Optional reference masks match the source stem or `_mask`/`-mask` suffixes. Supply white root foreground, black background and identical dimensions. The reader binarizes references at intensity >127. Reference masks do not enter prediction.

Select a fold model. Selection restores its validation threshold; manual changes are recorded in provenance. For the study use `research/fold_assignments.json` to choose each image's held-out model. A checkpoint selected for unrelated images does not carry an external accuracy guarantee. Selecting a threshold from test-mask scores invalidates held-out evaluation.

For new images, fold 0 and its threshold 0.47 provide a declared, reproducible starting configuration. Inspect masks and QC outputs; no fold has been established as biologically superior on new images. Do not choose among folds by scoring the same labels later reported as held-out evaluation. The fixed-loss models were retained as the highest-Dice case for the error audit, although the study's pixel-loss Attention U-Net had smaller segment/junction errors. These five models do not establish reliable biological counting.

Import upright root images. The dominant path selects the component with greatest vertical span and does not estimate anatomical orientation; rotating an image can change the selected path. Eight-neighbor adjacency can produce candidate junctions at rasterized bends as well as contacts and crossings.

Choose a writable output directory and optionally save float32 probability maps. Each task writes a new subdirectory.

## Analyze and inspect

Choose **Run analysis**. Language/model selection is disabled during processing. **Cancel** stops after the current image completes, retaining completed outputs and a cancelled report. Closing during analysis asks whether to exit after that image.

Select a result row. Source/overlay/mask/skeleton views support inspection of missed thin roots, extra foreground, crossings, false connections and border contact. The overlay highlights predicted foreground and the dominant path. The skeleton view combines the pruned skeleton with the raw dominant path. Individual-root identity is not traced.

The table reports a path in pixels, retained segments, candidate junction regions and an exploratory angle in degrees. **No flags** indicates only that implemented checks raised no flag; **Review required** indicates at least one flag. Neither is an anatomical accuracy verdict.

Settings and results have vertical scrollbars; the table has a horizontal scrollbar. Scroll small windows to inspect all settings and preview content.

## Language and export

Choose **English** or **中文** while idle. First use defaults to English; later runs retain the selected language where profile writing is available. Inputs, references, output setting, model, threshold, probability option, selected layer and current results are preserved. OS file dialogs retain their OS language.

Scientific columns, model identifiers, units and QC codes remain stable. Diagnostic text may differ by language. Numeric results do not.

Choose **Open results** and retain CSV, JSON, provenance and masks together. Original images/references must be preserved separately. CSV uses UTF-8 BOM. Missing numbers are empty in CSV and `null` in JSON. Read `path_status`, `angle_failure_reason`, `valid_angle_n`, `qc_flags` and `errors.csv`; do not replace missing values with zero.

## Batch use

```powershell
python cli.py images --references masks --output results --model fold_2 --save-probability --no-zip
python cli.py image.tif --output results --language zh
RootScope.exe --batch image.tif --output results --model fold_2 --no-zip
```

The command/GUI share the numerical pipeline. CLI exit codes are 0 for completion without image errors, 1 for logged image errors and 2 for argument/input validation errors. The windowed executable has no terminal progress; wait for its process to end.
