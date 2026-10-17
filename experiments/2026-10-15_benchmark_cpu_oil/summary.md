# Ablation summary

6 runs: 3 images x 2 configs x 1 seeds [0]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB | optimizer steps | peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| progressive | 22.73 +- 0.00 | 0.592 +- 0.000 | 26.68 +- 0.00 | 130.39 +- 0.00 | - (0/3) | - (0/3) | 2475 +- 0 | 0.127 +- 0.000 |
| fixed | 22.31 +- 0.00 | 0.577 +- 0.000 | 28.24 +- 0.00 | 107.61 +- 0.00 | 216 (3/3) | 282 (3/3) | 500 +- 0 | 0.185 +- 0.000 |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `fixed` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| progressive | PSNR (dB) | +0.424 | [+0.136, +0.712] | 100% of 3 | 0.25 |
| progressive | SSIM | +0.015 | [-0.007, +0.037] | 100% of 3 | 0.25 |
