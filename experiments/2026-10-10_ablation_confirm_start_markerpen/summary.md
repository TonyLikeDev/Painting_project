# Ablation summary

60 runs: 10 images x 2 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c10 | 17.93 +- 0.01 | 0.561 +- 0.002 | 0.556 +- 0.001 | 23.86 +- 0.07 | 6.34 +- 0.06 | 297 (25/30) | 262 (9/30) |
| c40 | 19.13 +- 0.08 | 0.601 +- 0.004 | 0.522 +- 0.003 | 24.56 +- 0.15 | 6.36 +- 0.03 | 319 (29/30) | 223 (10/30) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `c10` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| c40 | PSNR (dB) | +1.199 | [+0.969, +1.428] | 100% of 10 | 0.00195 |
| c40 | SSIM | +0.040 | [+0.020, +0.060] | 90% of 10 | 0.00391 |
| c40 | LPIPS | -0.033 | [-0.048, -0.019] | 100% of 10 | 0.00195 |
