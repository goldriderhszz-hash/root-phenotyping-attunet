# Released model catalog

Five frozen Attention U-Net models with the fixed soft-clDice training loss are distributed as ONNX graphs. They are independent fold checkpoints from seed 42, not an ensemble and not a refit on all 50 images. The source checkpoint, epoch, and threshold for each fold were selected without looking at that fold's test images. The machine-readable record is [`model/catalog.json`](model/catalog.json).

| Model | Validation-selected threshold | ONNX SHA-256 | Source checkpoint SHA-256 |
| --- | ---: | --- | --- |
| `fold_0` | 0.47 | `f81770d5c8e2e710da1b30f3354fef4e69d4641079b291fd7e857d4c647b8892` | `be9605d801cda3f92fbd10a3bb7fa28392bc93fc048f9c4cefad0a29853ccb50` |
| `fold_1` | 0.10 | `abcfbccab4d37b4696bb27c6474eed9e30bf0eaf722223990b77dcb78b6dc039` | `485a70d52d484f296e38c1eb90b414fbf0419e9d93f0d97fe273c6bb7401b35f` |
| `fold_2` | 0.82 | `a070c19c20783b5e6dce7d437ec7e973ff625726f9021638e34276ca9123cc59` | `877ff0a7bb1c581f2a7f74a171b8a70bd9fe74e399d9425f8c14d195e333913b` |
| `fold_3` | 0.10 | `7b087ce9197ca5463dcfc873e2f0fae935a64317fd25ce0013df44a365bb01fd` | `7667596688e702abbd6394f29b80a3bd29c1e179f5851521b9c08283bf8a96b9` |
| `fold_4` | 0.89 | `85fd5f8fbad1b31dde9c3f97855caa8089256c283f852d061579865d734f9603` | `5d42a481d6628dddc07d9f2b3ea7565b72526070e0934354b1116e68aa92f3e2` |

Each graph uses ONNX opset 17 and was checked against its PyTorch model on a deterministic 256 × 256 tile. The maximum absolute probability differences were at most `1.79e-7` in these conversion tests. This tile test verifies conversion only; the full-image deployment test is reported separately in [`validation/`](validation/README.md).

The desktop selects `fold_0` by default and checks the SHA-256 of the selected graph before processing. The GUI and CLI permit explicit choice of any fold and record the choice and threshold in `provenance.json`. Selecting a fold for a new image does not confer cross-validation independence or imply a known accuracy on that image. The 50-image out-of-fold result evaluates the fixed pipeline with each image processed by its own held-out fold model. External acquisition conditions need separate testing.

The PyTorch training checkpoints are not included in this inference package. To reproduce the exports from those original artifacts, supply the completed study directory to `scripts/export_fold_models.py`; the script checks each checkpoint against the study record before conversion. The original fold-0 training and export reports are included for provenance. All five ONNX graphs are included for immediate offline use.
