# Ablation summary

27 runs: 3 images x 3 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| orig | 18.69 +- 0.04 | 0.638 +- 0.004 | 0.473 +- 0.005 | 23.03 +- 0.09 | 6.67 +- 0.16 | 380 (6/9) | 432 (3/9) |
| small | 18.58 +- 0.08 | 0.638 +- 0.003 | 0.470 +- 0.007 | 22.61 +- 0.15 | 6.62 +- 0.18 | 432 (6/9) | 468 (3/9) |
| large | 19.04 +- 0.04 | 0.646 +- 0.002 | 0.474 +- 0.002 | 23.87 +- 0.11 | 6.55 +- 0.09 | 400 (9/9) | 355 (3/9) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `orig` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| small | PSNR (dB) | -0.111 | [-0.549, +0.328] | 33% of 3 | 0.5 |
| small | SSIM | -0.000 | [-0.009, +0.008] | 33% of 3 | 1 |
| small | LPIPS | -0.003 | [-0.020, +0.014] | 67% of 3 | 0.75 |
| large | PSNR (dB) | +0.349 | [+0.251, +0.448] | 100% of 3 | 0.25 |
| large | SSIM | +0.007 | [-0.018, +0.032] | 67% of 3 | 0.5 |
| large | LPIPS | +0.001 | [-0.067, +0.069] | 67% of 3 | 1 |
