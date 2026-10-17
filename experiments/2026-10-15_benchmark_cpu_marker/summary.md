# Ablation summary

6 runs: 3 images x 2 configs x 1 seeds [0]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB | optimizer steps | peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| progressive | 23.44 +- 0.00 | 0.722 +- 0.000 | 27.10 +- 0.00 | 130.66 +- 0.00 | - (0/3) | - (0/3) | 2475 +- 0 | 0.101 +- 0.000 |
| fixed | 19.46 +- 0.00 | 0.663 +- 0.000 | 27.10 +- 0.00 | 109.29 +- 0.00 | 210 (3/3) | 264 (2/3) | 500 +- 0 | 0.181 +- 0.000 |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `fixed` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| progressive | PSNR (dB) | +3.980 | [-3.252, +11.212] | 100% of 3 | 0.25 |
| progressive | SSIM | +0.059 | [+0.026, +0.093] | 100% of 3 | 0.25 |
