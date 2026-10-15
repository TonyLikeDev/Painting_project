# Ablation summary

27 runs: 3 images x 3 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| orig | 19.04 +- 0.04 | 0.646 +- 0.002 | 0.474 +- 0.002 | 23.87 +- 0.11 | 7.10 +- 0.09 | 400 (9/9) | 355 (3/9) |
| mid | 19.15 +- 0.08 | 0.654 +- 0.002 | 0.464 +- 0.006 | 23.89 +- 0.05 | 7.03 +- 0.03 | 387 (9/9) | 344 (3/9) |
| low | 19.40 +- 0.15 | 0.662 +- 0.002 | 0.444 +- 0.002 | 23.86 +- 0.07 | 6.94 +- 0.10 | 384 (9/9) | 343 (3/9) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `orig` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| mid | PSNR (dB) | +0.111 | [+0.024, +0.197] | 100% of 3 | 0.25 |
| mid | SSIM | +0.008 | [-0.006, +0.022] | 100% of 3 | 0.25 |
| mid | LPIPS | -0.009 | [-0.042, +0.023] | 67% of 3 | 0.5 |
| low | PSNR (dB) | +0.355 | [-0.059, +0.769] | 100% of 3 | 0.25 |
| low | SSIM | +0.016 | [-0.039, +0.072] | 100% of 3 | 0.25 |
| low | LPIPS | -0.030 | [-0.152, +0.092] | 100% of 3 | 0.25 |
