# Archived ablation artifacts

Each subdirectory contains the final checkpoint, ten frozen test predictions, the recorded 50-epoch training log, and the original pooled metric summary for one saved configuration.

| Directory | Configuration |
|---|---|
| `base_unet` | Base U-Net |
| `attention_unet` | Attention U-Net without the skeleton term |
| `unet_skeleton` | Base U-Net with the implemented skeleton term |
| `combined` | Attention U-Net with the implemented skeleton term and two-stage BCE weighting |

The combined checkpoint is also exposed at `models/combined_attention_unet_cldice.pth` for a stable user-facing path. Git LFS deduplicates the object in remote storage.

Complete baseline training specifications were not preserved. These artifacts support exact re-evaluation of the saved configurations, but they do not establish a fully controlled attribution of performance differences to individual components.
