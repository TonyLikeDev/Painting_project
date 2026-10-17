# Ablation summary

72 runs: 3 images x 8 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB | optimizer steps | peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prog | 20.01 +- 0.09 | 0.603 +- 0.004 | 0.533 +- 0.005 | 23.38 +- 0.11 | 21.35 +- 1.22 | - (0/9) | - (0/9) | 2475 +- 0 | 0.177 +- 0.007 |
| prog_alloc | 20.22 +- 0.12 | 0.617 +- 0.002 | 0.507 +- 0.004 | 24.25 +- 0.13 | 28.83 +- 2.22 | - (0/9) | - (0/9) | 2465 +- 3 | 0.324 +- 0.006 |
| prog_cos | 19.89 +- 0.04 | 0.602 +- 0.001 | 0.535 +- 0.002 | 23.15 +- 0.05 | 21.16 +- 0.95 | - (0/9) | - (0/9) | 2475 +- 0 | 0.172 +- 0.000 |
| prog_cos4 | 20.13 +- 0.03 | 0.605 +- 0.002 | 0.528 +- 0.003 | 23.86 +- 0.05 | 20.74 +- 0.25 | - (0/9) | - (0/9) | 2475 +- 0 | 0.172 +- 0.000 |
| prog_alloc_cos4 | 20.22 +- 0.05 | 0.621 +- 0.007 | 0.506 +- 0.007 | 24.57 +- 0.14 | 30.19 +- 5.04 | - (0/9) | - (0/9) | 2465 +- 1 | 0.331 +- 0.006 |
| fixed | 19.72 +- 0.06 | 0.602 +- 0.000 | 0.535 +- 0.003 | 25.19 +- 0.03 | 14.27 +- 0.53 | 284 (9/9) | 446 (9/9) | 500 +- 0 | 0.385 +- 0.007 |
| fixed_alloc | 19.83 +- 0.05 | 0.627 +- 0.001 | 0.489 +- 0.002 | 25.62 +- 0.06 | 21.93 +- 1.11 | 172 (9/9) | 254 (8/9) | 489 +- 0 | 0.625 +- 0.007 |
| fixed_cos4 | 19.73 +- 0.07 | 0.604 +- 0.004 | 0.528 +- 0.004 | 25.69 +- 0.06 | 13.92 +- 0.47 | 259 (9/9) | 393 (9/9) | 500 +- 0 | 0.381 +- 0.000 |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `prog` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| prog_alloc | PSNR (dB) | +0.211 | [-0.082, +0.504] | 100% of 3 | 0.25 |
| prog_alloc | SSIM | +0.014 | [-0.031, +0.059] | 67% of 3 | 0.5 |
| prog_alloc | LPIPS | -0.027 | [-0.090, +0.037] | 100% of 3 | 0.25 |
| prog_cos | PSNR (dB) | -0.113 | [-0.497, +0.270] | 33% of 3 | 0.5 |
| prog_cos | SSIM | -0.001 | [-0.003, +0.001] | 0% of 3 | 0.25 |
| prog_cos | LPIPS | +0.001 | [-0.018, +0.020] | 67% of 3 | 1 |
| prog_cos4 | PSNR (dB) | +0.124 | [-0.142, +0.390] | 100% of 3 | 0.25 |
| prog_cos4 | SSIM | +0.002 | [-0.020, +0.023] | 67% of 3 | 0.75 |
| prog_cos4 | LPIPS | -0.005 | [-0.010, +0.000] | 100% of 3 | 0.25 |
| prog_alloc_cos4 | PSNR (dB) | +0.212 | [-0.031, +0.456] | 100% of 3 | 0.25 |
| prog_alloc_cos4 | SSIM | +0.018 | [-0.025, +0.060] | 100% of 3 | 0.25 |
| prog_alloc_cos4 | LPIPS | -0.027 | [-0.100, +0.046] | 100% of 3 | 0.25 |
| fixed | PSNR (dB) | -0.289 | [-0.610, +0.031] | 0% of 3 | 0.25 |
| fixed | SSIM | -0.002 | [-0.037, +0.034] | 33% of 3 | 0.75 |
| fixed | LPIPS | +0.001 | [-0.047, +0.049] | 67% of 3 | 1 |
| fixed_alloc | PSNR (dB) | -0.182 | [-1.168, +0.805] | 33% of 3 | 0.5 |
| fixed_alloc | SSIM | +0.023 | [-0.106, +0.153] | 67% of 3 | 0.75 |
| fixed_alloc | LPIPS | -0.044 | [-0.256, +0.167] | 67% of 3 | 0.75 |
| fixed_cos4 | PSNR (dB) | -0.273 | [-0.551, +0.006] | 0% of 3 | 0.25 |
| fixed_cos4 | SSIM | +0.001 | [-0.040, +0.041] | 33% of 3 | 1 |
| fixed_cos4 | LPIPS | -0.005 | [-0.050, +0.039] | 67% of 3 | 0.75 |
