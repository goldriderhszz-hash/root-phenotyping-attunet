# Manuscript to repository mapping

| Manuscript component | Repository location |
|---|---|
| Section 2.3 image split and tiles | `data/splits/seed42.json`, `src/root_phenotyping/pipeline.py` |
| Section 2.4 Attention U-Net | `src/root_phenotyping/pipeline.py` |
| Section 2.4.2 two-stage loss | `src/root_phenotyping/pipeline.py`, `configs/manuscript.yaml` |
| Section 2.3.2 Gaussian fusion | `src/root_phenotyping/pipeline.py` |
| Section 2.5 skeleton descriptors | `src/root_phenotyping/phenotypes.py` |
| Sections 2.6 and 3 evaluation | `analysis/`, `results/manuscript/` |
| Frozen combined checkpoint | `models/combined_attention_unet_cldice.pth` |
| Source images | `data/images/` |
| Binary reference masks | `data/masks/` |
| LabelMe annotations | `data/annotations/` |

The repository commit cited in the submitted manuscript should be updated from the original baseline commit `85876d62dc5907cd9f0b1e158a07ba111bf267c6` to the immutable release or archival DOI created from this submission branch.
