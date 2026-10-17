# Ablation summary

6 runs: 3 images x 2 configs x 1 seeds [0]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB | optimizer steps | peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| progressive | 24.78 +- 0.00 | 0.736 +- 0.000 | 27.98 +- 0.00 | 130.78 +- 0.00 | - (0/3) | - (0/3) | 2475 +- 0 | 0.094 +- 0.000 |
| fixed | 22.96 +- 0.00 | 0.688 +- 0.000 | 27.11 +- 0.00 | 109.05 +- 0.00 | 268 (3/3) | 295 (2/3) | 500 +- 0 | 0.189 +- 0.000 |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `fixed` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| progressive | PSNR (dB) | +1.825 | [-1.166, +4.816] | 100% of 3 | 0.25 |
| progressive | SSIM | +0.048 | [+0.008, +0.088] | 100% of 3 | 0.25 |
