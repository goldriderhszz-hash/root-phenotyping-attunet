# Source ONNX CPU timing

All 50 own-fold images were processed three times (150 actual image-runs) on
AMD Ryzen 7 7840H with Radeon 780M Graphics, 8 physical cores and
16 logical processors, Windows 11,
CPUExecutionProvider, eight intra-op threads and default inter-op settings.
No timings were copied or extrapolated from other images/repeats.

Total median: **52.26 s/image**; IQR **49.05–57.41 s**;
P90 **66.55 s**. CLAHE/inference/fusion median: 37.41 s;
descriptor median: 14.33 s. Stage medians need not sum to total median.

The clock includes image decoding, CLAHE, tiled ONNX inference/Gaussian fusion,
thresholding and the frozen descriptor extractor. It excludes model loading, a
256 × 256 warm-up tile, reference evaluation and disk output. These source numerical
pipeline timings are distinct from full interactive/packaged-app wall time.
Original GPU timing used another implementation/scope and is not a controlled
hardware comparison. Repeats are runtime measurements on the same 50 images, not
150 independent plants. All 150 descriptor rows match the existing own-fold source
deployment rows within 1e-8; no model or parameter was selected using timing.

cpu_image_runs.csv contains every measured stage; summary.json identifies hardware,
models, thresholds, runtime versions, source hashes and descriptive quantiles.
