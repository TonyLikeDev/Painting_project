# Ablation summary

6 runs: 3 images x 2 configs x 1 seeds [0]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB | optimizer steps | peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| progressive | 23.81 +- 0.00 | 0.723 +- 0.000 | 27.86 +- 0.00 | 130.33 +- 0.00 | - (0/3) | - (0/3) | 2475 +- 0 | 0.107 +- 0.000 |
| fixed | 20.32 +- 0.00 | 0.641 +- 0.000 | 27.00 +- 0.00 | 108.49 +- 0.00 | 234 (3/3) | 328 (3/3) | 500 +- 0 | 0.224 +- 0.000 |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `fixed` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| progressive | PSNR (dB) | +3.487 | [-1.988, +8.962] | 100% of 3 | 0.25 |
| progressive | SSIM | +0.082 | [-0.021, +0.184] | 100% of 3 | 0.25 |
