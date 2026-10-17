# Ablation summary

54 runs: 3 images x 6 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB | optimizer steps | peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prog | 20.01 +- 0.09 | 0.603 +- 0.004 | 0.533 +- 0.005 | 23.38 +- 0.11 | 22.98 +- 1.64 | - (0/9) | - (0/9) | 2475 +- 0 | 0.181 +- 0.007 |
| prog_i5 | 19.52 +- 0.09 | 0.588 +- 0.002 | 0.554 +- 0.003 | 22.14 +- 0.10 | 2.58 +- 0.27 | - (0/9) | - (0/9) | 225 +- 0 | 0.172 +- 0.000 |
| prog_i11 | 19.72 +- 0.10 | 0.593 +- 0.004 | 0.548 +- 0.004 | 22.76 +- 0.06 | 4.41 +- 0.18 | - (0/9) | - (0/9) | 495 +- 0 | 0.172 +- 0.000 |
| prog_i22 | 19.93 +- 0.04 | 0.598 +- 0.002 | 0.543 +- 0.004 | 23.02 +- 0.04 | 8.72 +- 0.68 | - (0/9) | - (0/9) | 990 +- 0 | 0.172 +- 0.000 |
| prog_i11_alloc | 20.12 +- 0.06 | 0.614 +- 0.003 | 0.520 +- 0.001 | 23.99 +- 0.08 | 10.94 +- 0.86 | - (0/9) | - (0/9) | 758 +- 8 | 0.326 +- 0.010 |
| fixed | 19.72 +- 0.06 | 0.602 +- 0.000 | 0.535 +- 0.003 | 25.19 +- 0.03 | 12.62 +- 0.83 | 284 (9/9) | 446 (9/9) | 500 +- 0 | 0.380 +- 0.000 |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `prog` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| prog_i5 | PSNR (dB) | -0.489 | [-0.879, -0.099] | 0% of 3 | 0.25 |
| prog_i5 | SSIM | -0.016 | [-0.030, -0.002] | 0% of 3 | 0.25 |
| prog_i5 | LPIPS | +0.021 | [+0.001, +0.040] | 0% of 3 | 0.25 |
| prog_i11 | PSNR (dB) | -0.285 | [-0.856, +0.286] | 0% of 3 | 0.25 |
| prog_i11 | SSIM | -0.011 | [-0.020, -0.001] | 0% of 3 | 0.25 |
| prog_i11 | LPIPS | +0.015 | [+0.012, +0.017] | 0% of 3 | 0.25 |
| prog_i22 | PSNR (dB) | -0.078 | [-0.239, +0.083] | 0% of 3 | 0.25 |
| prog_i22 | SSIM | -0.005 | [-0.009, -0.001] | 0% of 3 | 0.25 |
| prog_i22 | LPIPS | +0.009 | [-0.001, +0.020] | 0% of 3 | 0.25 |
| prog_i11_alloc | PSNR (dB) | +0.108 | [-0.170, +0.387] | 100% of 3 | 0.25 |
| prog_i11_alloc | SSIM | +0.010 | [-0.040, +0.061] | 33% of 3 | 1 |
| prog_i11_alloc | LPIPS | -0.014 | [-0.106, +0.079] | 33% of 3 | 1 |
| fixed | PSNR (dB) | -0.289 | [-0.610, +0.031] | 0% of 3 | 0.25 |
| fixed | SSIM | -0.002 | [-0.037, +0.034] | 33% of 3 | 0.75 |
| fixed | LPIPS | +0.001 | [-0.047, +0.049] | 67% of 3 | 1 |
