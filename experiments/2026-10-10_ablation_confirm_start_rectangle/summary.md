# Ablation summary

60 runs: 10 images x 2 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| orig | 20.71 +- 0.02 | 0.612 +- 0.000 | 0.483 +- 0.001 | 25.26 +- 0.03 | 6.47 +- 0.02 | 342 (30/30) | 422 (20/30) |
| large | 21.07 +- 0.02 | 0.634 +- 0.003 | 0.466 +- 0.003 | 25.92 +- 0.10 | 6.47 +- 0.01 | 298 (30/30) | 392 (27/30) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `orig` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| large | PSNR (dB) | +0.360 | [+0.142, +0.578] | 100% of 10 | 0.00195 |
| large | SSIM | +0.022 | [+0.015, +0.030] | 100% of 10 | 0.00195 |
| large | LPIPS | -0.017 | [-0.027, -0.006] | 100% of 10 | 0.00195 |
