# Ablation summary

6 runs: 3 images x 2 configs x 1 seeds [0]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB | optimizer steps | peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| progressive | 23.32 +- 0.00 | 0.720 +- 0.000 | 27.27 +- 0.00 | 12.43 +- 0.00 | - (0/3) | - (0/3) | 2475 +- 0 | 0.179 +- 0.000 |
| fixed | 19.11 +- 0.00 | 0.653 +- 0.000 | 26.98 +- 0.00 | 5.68 +- 0.00 | 210 (3/3) | 250 (2/3) | 500 +- 0 | 0.380 +- 0.000 |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `fixed` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| progressive | PSNR (dB) | +4.218 | [-2.305, +10.740] | 100% of 3 | 0.25 |
| progressive | SSIM | +0.068 | [+0.018, +0.118] | 100% of 3 | 0.25 |
