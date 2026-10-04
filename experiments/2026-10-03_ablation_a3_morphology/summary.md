# Ablation summary

60 runs: 10 images x 2 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| With dilate/erode (default) | 19.84 +- 0.04 | 0.556 +- 0.001 | 24.93 +- 0.12 | 15.43 +- 0.81 | 353 (30/30) | 428 (21/30) |
| Without | 17.13 +- 0.02 | 0.534 +- 0.002 | 23.18 +- 0.03 | 14.48 +- 0.32 | 287 (21/30) | 425 (9/30) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `With dilate/erode (default)` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| Without | PSNR (dB) | -2.707 | [-3.945, -1.470] | 0% of 10 | 0.00195 |
| Without | SSIM | -0.022 | [-0.068, +0.023] | 30% of 10 | 0.275 |
