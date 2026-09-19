# Manuscript analyses

`segmentation_eval.py` evaluates the four archived prediction sets on all ten frozen test images. It calculates pooled and image-level overlap metrics, hard-skeleton clDice, paired complete-image bootstrap intervals, the 2-pixel tolerance diagnostic, and width-stratified exact centerline recall.

`phenotype_eval.py` applies the unchanged extractor to the combined-model predictions and reference masks, then calculates MAE, RMSE, MAPE, squared Pearson correlation, Lin's concordance correlation coefficient, bootstrap bias intervals, and Bland-Altman summaries.

`model_robustness.py` reruns the archived combined checkpoint under the original, gamma 0.8, gamma 1.2, and Gaussian-blur conditions. It requires a CUDA GPU and intentionally does not substitute CPU timing.

Run the CPU analyses with:

```bash
python analysis/run_all.py
```

Include the frozen-model perturbation and timing experiment with:

```bash
python analysis/run_all.py --include-gpu-robustness
```

Generated files are written under `runs/evaluation/`. The submitted numerical records remain under `results/manuscript/`.
