# Ablation summary

27 runs: 3 images x 3 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| orig | 19.37 +- 0.09 | 0.606 +- 0.003 | 0.525 +- 0.002 | 24.03 +- 0.11 | 6.51 +- 0.31 | 395 (9/9) | 366 (3/9) |
| small | 18.74 +- 0.04 | 0.607 +- 0.001 | 0.530 +- 0.002 | 22.21 +- 0.17 | 6.99 +- 0.20 | 438 (6/9) | 496 (1/9) |
| large | 19.72 +- 0.06 | 0.602 +- 0.000 | 0.535 +- 0.003 | 25.19 +- 0.03 | 6.95 +- 0.05 | 284 (9/9) | 446 (9/9) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `orig` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| small | PSNR (dB) | -0.631 | [-1.382, +0.121] | 0% of 3 | 0.25 |
| small | SSIM | +0.001 | [-0.027, +0.029] | 33% of 3 | 1 |
| small | LPIPS | +0.005 | [-0.029, +0.039] | 33% of 3 | 0.5 |
| large | PSNR (dB) | +0.352 | [-0.093, +0.798] | 100% of 3 | 0.25 |
| large | SSIM | -0.005 | [-0.040, +0.031] | 67% of 3 | 1 |
| large | LPIPS | +0.010 | [-0.037, +0.057] | 33% of 3 | 0.75 |
