# Troubleshooting

| Symptom | Action |
| --- | --- |
| EXE does not launch | Extract the entire ZIP and keep _internal beside it |
| Model hash mismatch | Check catalog SHA-256; restore the original ONNX file |
| Corrupt image | Verify supported format and complete file contents |
| Reference not matched | Use the same stem or _mask/-mask suffix |
| Reference size mismatch | Supply the original dimensions; resizing changes evaluation |
| No successful image | Inspect errors.csv and resolve its recorded causes |
| Missing path/angle | Inspect masks and status fields; do not substitute zero |
| Slow image | Wait; cancellation is checked after the current image |
| Small window | Use settings/results vertical and table horizontal scrolling |
| Output writing fails | Choose a writable directory and check disk space |
| Language not retained | Read-only profiles still allow an in-session switch |

Issue reports should identify the source commit or executable hash, OS, dimensions, fold, threshold, error and optional reference. Use a minimal permitted example instead of uploading an unpublished collection.
