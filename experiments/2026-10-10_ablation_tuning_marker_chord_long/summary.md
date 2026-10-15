# Ablation summary

36 runs: 3 images x 4 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c30 | 18.00 +- 0.14 | 0.637 +- 0.006 | 0.524 +- 0.009 | 23.97 +- 0.11 | 5.69 +- 0.11 | 363 (9/9) | 321 (3/9) |
| c40 | 18.29 +- 0.13 | 0.639 +- 0.003 | 0.518 +- 0.007 | 23.32 +- 0.19 | 5.61 +- 0.00 | 437 (8/9) | 370 (3/9) |
| c50 | 18.19 +- 0.05 | 0.634 +- 0.001 | 0.524 +- 0.004 | 22.58 +- 0.15 | 5.61 +- 0.00 | 294 (4/9) | 466 (3/9) |
| c60 | 18.00 +- 0.19 | 0.621 +- 0.006 | 0.536 +- 0.003 | 21.80 +- 0.13 | 5.60 +- 0.01 | 345 (3/9) | - (0/9) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `c30` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| c40 | PSNR (dB) | +0.288 | [+0.013, +0.563] | 100% of 3 | 0.25 |
| c40 | SSIM | +0.002 | [-0.014, +0.018] | 67% of 3 | 0.75 |
| c40 | LPIPS | -0.006 | [-0.015, +0.002] | 100% of 3 | 0.25 |
| c50 | PSNR (dB) | +0.189 | [-0.275, +0.653] | 67% of 3 | 0.5 |
| c50 | SSIM | -0.003 | [-0.037, +0.032] | 67% of 3 | 1 |
| c50 | LPIPS | -0.000 | [-0.038, +0.037] | 67% of 3 | 1 |
| c60 | PSNR (dB) | +0.004 | [-0.372, +0.379] | 67% of 3 | 1 |
| c60 | SSIM | -0.016 | [-0.047, +0.015] | 0% of 3 | 0.25 |
| c60 | LPIPS | +0.012 | [-0.051, +0.075] | 33% of 3 | 0.75 |
