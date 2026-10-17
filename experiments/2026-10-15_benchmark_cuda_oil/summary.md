# Ablation summary

6 runs: 3 images x 2 configs x 1 seeds [0]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB | optimizer steps | peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| progressive | 22.74 +- 0.00 | 0.592 +- 0.000 | 26.76 +- 0.00 | 12.01 +- 0.00 | - (0/3) | - (0/3) | 2475 +- 0 | 0.185 +- 0.000 |
| fixed | 22.31 +- 0.00 | 0.578 +- 0.000 | 28.09 +- 0.00 | 5.63 +- 0.00 | 227 (3/3) | 312 (3/3) | 500 +- 0 | 0.380 +- 0.000 |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `fixed` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| progressive | PSNR (dB) | +0.434 | [-0.049, +0.918] | 100% of 3 | 0.25 |
| progressive | SSIM | +0.014 | [-0.012, +0.041] | 100% of 3 | 0.25 |
