# AI Phenotype Tool

This directory is the auditable reconstruction of the authors' Windows root-image application. It uses the archived ONNX model from `AI_Phenotype_Tool.zip`, the manuscript preprocessing and threshold (`CLAHE`, 256-pixel Gaussian sliding windows, 50% overlap, threshold 0.53), and the same descriptor implementation exposed by `root_phenotyping.phenotypes`.

The previous 169 MB ZIP was a PyInstaller runtime bundle containing `main.exe`, third-party libraries, and the ONNX model but no editable application source. It is intentionally not retained in the submission tree. The two ONNX files below are byte-identical copies extracted from that archive; their checksums are recorded in `PROVENANCE.md`.

## Run from source

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r software/requirements.txt
python -m pip install -e . --no-deps
python -m root_phenotyping.gui
```

The interface accepts TIFF, PNG, JPEG, and BMP images and reports four two-dimensional image-derived descriptors. The dominant-axis path value is not calibrated to a physical unit.

## Build the Windows application

```powershell
powershell -ExecutionPolicy Bypass -File software/build_windows.ps1
```

The build appears under `dist/AI_Phenotype_Tool/`. Archive that directory for a GitHub Release; do not commit the generated runtime bundle to the source tree.

## Relationship to the manuscript

The GUI is a convenience interface. The paper's reported numbers remain reproducible through the command-line workflow in the repository root. This reconstruction replaces ambiguous labels such as “primary root length” with the interpretation limits used in the revised manuscript and repository documentation.
