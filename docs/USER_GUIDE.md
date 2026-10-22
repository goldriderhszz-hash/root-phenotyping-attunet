# RootScope user guide

RootScope performs D3 segmentation and contour-guided whole-root reconstruction for rhizobag photographs. Results require review. See [paper alignment](PAPER_ALIGNMENT.md) for validation and scientific limitations.

## Windows

Extract the complete `RootScope-Windows-x64.zip`. Open `RootScope.exe` and keep `_internal` beside it. Select images or a folder, keep model `auto`, choose an output directory and run analysis. Known study images use their own recorded fold by image hash; new images use fold 0 with a review flag. Automatic mode uses each fold's validation-selected threshold. Manual fold/threshold choices are recorded.

Select a result to view the source, overlay, mask and retained paths. The table shows operational axis length (px), retained paths, geometric Y attachments, degree-3 junctions and parent-child projected angle (deg). Missing measurements are NA. Orange is the protected axis; blue is other retained paths. **Review required** is a request to inspect results, not a measurement of human correction burden.

The language selector supports English and Chinese. Inputs and completed results are preserved when switching while idle. Large images and complex graphs can take appreciably longer than small examples; cancellation takes effect between images.

## Source and command line

```text
python -m pip install -r requirements.txt
python desktop.py
python cli.py images --output results
python cli.py image.tif --output results --model paper_fold_2 --save-probability
python cli.py images --output historical-results --pipeline legacy --model fold_0
```

Python 3.12 with Tk is needed for the GUI. Source tests and Windows packaging are in GitHub Actions. The paper runtime loads frozen numeric forests without requiring scikit-learn/joblib. No ensemble or all-data production model is implied. A grayscale file is accepted as a photograph with replicated channels, but it differs from the colour acquisition used in D3 and requires separate validation.

## Outputs

Each run creates `Report.md`, `results.csv`, `results.json`, `errors.csv`, `provenance.json`, masks, PNG overlays and editable `reconstruction.svg`. The optional float32 probability archive supports debugging. A matching reference mask supplies segmentation scores; it cannot supply whole-root identities or anatomical counts. Whole-root reference contours are never passed to reconstruction at inference.

The historical release remains available at the `rootscope-desktop` tag. It uses the old skeleton pipeline; the new application is named RootScope. Commit IDs, algorithm identifiers and archived tags are retained for reproducibility.

## Build notes

Run `scripts/build_windows.ps1 -Python python` on Windows. Standard Python distributions provide Tcl/Tk dependencies. A relocated/custom Tcl runtime may require `-TclZlib PATH_TO_ZLIB1_DLL`; this explicit option bundles that runtime dependency. Run `RootScope.exe --self-test REPORT.json` to verify all ten ONNX models and twenty reconstruction artifacts. A visible GUI still requires an interactive Windows desktop.

The SVG embeds a JPEG presentation photograph; `source_decoded.png` preserves the decoded image losslessly. Analytical calculations always use the input photograph, not the figure preview.
