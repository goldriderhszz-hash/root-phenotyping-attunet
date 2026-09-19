# Reconstruction provenance

The original `AI_Phenotype_Tool.zip` was retrieved from Git LFS object `96e7ea9cdce9535bd7186ced71fb568536c8433ef896e8a3ea5dae202a63bd04` and inspected without executing the bundled application.

| Archived object | Size (bytes) | SHA-256 |
|---|---:|---|
| `main.exe` | 7,426,443 | `f93605f3b48c99584c6d1489e491642ce42ae34871798333bb552428015750eb` |
| `precision_boost_attention_unet.onnx` | 125,275 | `382585956579edf3d3c3b897dcd84dfbc65805cbf41d09f9f46c63da868f08d8` |
| `precision_boost_attention_unet.onnx.data` | 31,916,032 | `e3611828ead827c69724f045a89c18871b614144dcb5a688350fcad0f76f8d68` |

Static inspection identified a Python 3.10/PyQt5 application with ONNX Runtime inference, 256-pixel Gaussian-weighted windows, 50% overlap, CLAHE preprocessing, a 0.53 threshold, skeleton visualization, and mask-derived measurements. No authoring source files were present in the ZIP.

The files in this directory are therefore a clean-room reconstruction from the observable application structure and the public manuscript implementation, not a claim that the original source text was recovered verbatim. The archived ONNX graph and external tensor-data file are retained unchanged.

The Windows directory build was smoke-tested on 19 September 2026 with Python 3.12.14, PyInstaller 6.22.3, and ONNX Runtime 1.23.2. The application process remained live after loading the bundled model, and both model-file hashes in the build matched the table above.
