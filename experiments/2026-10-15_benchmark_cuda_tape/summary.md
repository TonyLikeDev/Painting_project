# Ablation summary

6 runs: 3 images x 2 configs x 1 seeds [0]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB | optimizer steps | peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| progressive | 24.70 +- 0.00 | 0.736 +- 0.000 | 27.94 +- 0.00 | 12.89 +- 0.00 | - (0/3) | - (0/3) | 2475 +- 0 | 0.179 +- 0.000 |
| fixed | 22.80 +- 0.00 | 0.686 +- 0.000 | 27.13 +- 0.00 | 5.69 +- 0.00 | 267 (3/3) | 288 (2/3) | 500 +- 0 | 0.380 +- 0.000 |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `fixed` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| progressive | PSNR (dB) | +1.902 | [-1.521, +5.325] | 100% of 3 | 0.25 |
| progressive | SSIM | +0.050 | [+0.001, +0.099] | 100% of 3 | 0.25 |
