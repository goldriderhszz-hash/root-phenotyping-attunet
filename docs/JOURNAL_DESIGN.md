# Journal-oriented interface and output design

Sources checked 22 October 2026:

- [Plant Methods general submission guidelines](https://link.springer.com/journal/13007/submission-guidelines).
- [Plant Methods software article criteria](https://link.springer.com/journal/13007/submission-guidelines/software).
- [Nature figure specifications](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/).

There is no official Plant Methods desktop-UI theme or GitHub directory template. The interface uses readable sans-serif typography, a restrained palette, explicit units and review states. Colour is supported by text labels. The application/project/executable name is **RootScope**, without a version suffix. Algorithm identities, commit IDs and archival release tags remain in provenance, because removing them would harm reproducibility.

## Export templates

Every paper run creates a descriptive `Report.md`, machine-readable CSV/JSON and model provenance. Per-image `reconstruction.svg` embeds the original photograph and retains editable vector curves in separate elements. Orange identifies the protected operational axis; blue identifies other retained paths. Output includes a 300-dpi-tagged PNG preview. This metadata does not upsample the source or establish effective print resolution. SVG size fits within 180 mm width and 170 mm height while preserving aspect ratio. A different journal may require a different final size.

`templates/PlantMethods_report.md` is an analysis-report scaffold, not an official manuscript template. `templates/figure_profiles.json` records export targets and source URLs. Put titles and legends in the manuscript when preparing Plant Methods figure files; do not bake them into the image. Keep source images, legible keys, units, real scale calibration and figure-specific captions. Do not invent scale bars or anatomical labels.

## Submission boundary

Plant Methods expects validated software and comparison with related tools. It requires software availability/requirements and data availability information. Those evidence requirements are not certified by a polished UI. Its general guidance requests line/page numbering and double spacing; its software article guidance requests a structured abstract of at most 350 words. The author's existing one-paragraph/no-line-number manuscript exceptions are not official journal rules. This software update does not edit or submit the manuscript.

The embedded photograph in SVG uses JPEG quality 95 for presentation. `source_decoded.png` preserves the lossless decoded 8-bit photograph; inference always reads the original input. Vector coordinates remain in original pixel space. The verified full-image SVG was 2.67 MB, below the Plant Methods 10 MB figure-file limit. Inspect effective resolution at final print size before submission.

The source GUI constructor was checked for the RootScope title, automatic model selection and seven measurement columns. Native screenshot inspection could not be confirmed through the desktop helper in this environment; no visual acceptance claim is made.
