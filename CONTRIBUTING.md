# Contributing

Issues and pull requests are welcome. Please describe the input image type, expected behavior, actual behavior, operating system, and RootScope version. Do not post private or unpublished plant images without permission.

For code changes, create a Python 3.12 environment, install `requirements-build.txt`, and run `python -m unittest discover -s tests -v`. Keep model behavior reproducible: changes to preprocessing, inference, descriptor definitions, or a fold's default threshold should update the method documentation, `model/catalog.json`, and provenance fields. Changes to any ONNX artifact require a new SHA-256 and conversion report. Keep user-facing statements limited to validated image descriptors and evaluation measures. Add real-image validation whenever a change could alter segmentation or descriptor outputs.
