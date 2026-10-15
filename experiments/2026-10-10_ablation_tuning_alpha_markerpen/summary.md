# Ablation summary

27 runs: 3 images x 3 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| orig | 18.08 +- 0.14 | 0.631 +- 0.004 | 0.530 +- 0.001 | 24.19 +- 0.24 | 7.08 +- 0.10 | 365 (9/9) | 318 (3/9) |
| opaque | 18.22 +- 0.08 | 0.635 +- 0.004 | 0.528 +- 0.002 | 24.08 +- 0.11 | 6.97 +- 0.08 | 390 (9/9) | 298 (3/9) |
| low | 18.28 +- 0.23 | 0.627 +- 0.007 | 0.531 +- 0.007 | 24.32 +- 0.11 | 6.97 +- 0.02 | 360 (9/9) | 313 (3/9) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `orig` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| opaque | PSNR (dB) | +0.135 | [-0.355, +0.626] | 67% of 3 | 0.5 |
| opaque | SSIM | +0.005 | [-0.013, +0.022] | 67% of 3 | 0.5 |
| opaque | LPIPS | -0.003 | [-0.040, +0.034] | 67% of 3 | 0.75 |
| low | PSNR (dB) | +0.202 | [-0.182, +0.586] | 100% of 3 | 0.25 |
| low | SSIM | -0.004 | [-0.024, +0.016] | 33% of 3 | 0.5 |
| low | LPIPS | +0.000 | [-0.026, +0.026] | 33% of 3 | 1 |
