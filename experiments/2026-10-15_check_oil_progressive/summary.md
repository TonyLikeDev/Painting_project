# Ablation summary

45 runs: 5 images x 3 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| original | 20.30 +- 0.09 | 0.578 +- 0.002 | 0.553 +- 0.001 | - | - |
| ours | 20.95 +- 0.05 | 0.584 +- 0.003 | 0.549 +- 0.002 | 25.20 +- 0.12 | 19.82 +- 0.56 |
| ours_linear | 20.25 +- 0.04 | 0.577 +- 0.001 | 0.551 +- 0.004 | 22.79 +- 0.14 | 19.78 +- 0.68 |

## Paired against `original` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| ours | PSNR (dB) | +0.646 | [+0.209, +1.084] | 100% of 5 | 0.0625 |
| ours | SSIM | +0.006 | [-0.005, +0.016] | 80% of 5 | 0.188 |
| ours | LPIPS | -0.003 | [-0.017, +0.010] | 40% of 5 | 0.812 |
| ours_linear | PSNR (dB) | -0.049 | [-0.145, +0.046] | 40% of 5 | 0.312 |
| ours_linear | SSIM | -0.000 | [-0.006, +0.005] | 60% of 5 | 1 |
| ours_linear | LPIPS | -0.002 | [-0.013, +0.009] | 60% of 5 | 0.625 |
