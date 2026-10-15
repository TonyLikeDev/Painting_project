# Ablation summary

27 runs: 3 images x 3 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| orig | 20.00 +- 0.04 | 0.674 +- 0.001 | 0.467 +- 0.003 | 24.01 +- 0.03 | 7.06 +- 0.18 | 413 (9/9) | 333 (3/9) |
| opaque | 20.03 +- 0.03 | 0.675 +- 0.001 | 0.464 +- 0.002 | 24.13 +- 0.05 | 7.05 +- 0.02 | 413 (9/9) | 336 (3/9) |
| low | 19.93 +- 0.06 | 0.657 +- 0.002 | 0.478 +- 0.003 | 23.90 +- 0.08 | 7.05 +- 0.01 | 408 (9/9) | 349 (3/9) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `orig` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| opaque | PSNR (dB) | +0.029 | [-0.278, +0.336] | 67% of 3 | 0.75 |
| opaque | SSIM | +0.000 | [-0.008, +0.008] | 33% of 3 | 1 |
| opaque | LPIPS | -0.003 | [-0.031, +0.026] | 67% of 3 | 0.75 |
| low | PSNR (dB) | -0.067 | [-0.288, +0.154] | 33% of 3 | 0.5 |
| low | SSIM | -0.017 | [-0.063, +0.029] | 33% of 3 | 0.5 |
| low | LPIPS | +0.011 | [-0.043, +0.064] | 33% of 3 | 0.75 |
