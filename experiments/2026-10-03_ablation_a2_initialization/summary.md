# Ablation summary

120 runs: 10 images x 4 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pixel | 19.81 +- 0.06 | 0.556 +- 0.002 | 0.565 +- 0.001 | 24.89 +- 0.10 | 12.96 +- 0.26 | 348 (30/30) | 400 (19/30) |
| ot | 19.75 +- 0.06 | 0.550 +- 0.002 | 0.573 +- 0.003 | 24.63 +- 0.01 | 33.68 +- 0.86 | 378 (30/30) | 422 (16/30) |
| pixel_flat | 18.94 +- 0.06 | 0.533 +- 0.002 | 0.597 +- 0.001 | 21.54 +- 0.17 | 12.72 +- 0.22 | 380 (9/30) | 487 (5/30) |
| ot_flat | 18.68 +- 0.12 | 0.526 +- 0.002 | 0.601 +- 0.002 | 21.04 +- 0.15 | 29.90 +- 0.85 | 380 (7/30) | 448 (4/30) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `pixel` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| ot | PSNR (dB) | -0.056 | [-0.177, +0.064] | 50% of 10 | 0.322 |
| ot | SSIM | -0.006 | [-0.009, -0.003] | 10% of 10 | 0.00391 |
| ot | LPIPS | +0.008 | [+0.003, +0.014] | 10% of 10 | 0.00391 |
| pixel_flat | PSNR (dB) | -0.866 | [-1.249, -0.482] | 0% of 10 | 0.00195 |
| pixel_flat | SSIM | -0.024 | [-0.037, -0.010] | 10% of 10 | 0.00391 |
| pixel_flat | LPIPS | +0.033 | [+0.010, +0.056] | 0% of 10 | 0.00195 |
| ot_flat | PSNR (dB) | -1.130 | [-1.608, -0.652] | 0% of 10 | 0.00195 |
| ot_flat | SSIM | -0.031 | [-0.043, -0.018] | 0% of 10 | 0.00195 |
| ot_flat | LPIPS | +0.037 | [+0.016, +0.058] | 0% of 10 | 0.00195 |
