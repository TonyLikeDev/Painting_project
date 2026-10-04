# Ablation summary

120 runs: 10 images x 4 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Error map, pixel | 19.81 +- 0.06 | 0.556 +- 0.002 | 24.89 +- 0.10 | 12.96 +- 0.26 | 348 (30/30) | 400 (19/30) |
| Error map, + Sinkhorn 100 | 19.75 +- 0.06 | 0.550 +- 0.002 | 24.63 +- 0.01 | 33.68 +- 0.86 | 378 (30/30) | 422 (16/30) |
| Uniform, pixel | 18.94 +- 0.06 | 0.533 +- 0.002 | 21.54 +- 0.17 | 12.72 +- 0.22 | 380 (9/30) | 487 (5/30) |
| Uniform, + Sinkhorn 100 | 18.68 +- 0.12 | 0.526 +- 0.002 | 21.04 +- 0.15 | 29.90 +- 0.85 | 380 (7/30) | 448 (4/30) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `Uniform, pixel` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| Error map, pixel | PSNR (dB) | +0.866 | [+0.482, +1.249] | 100% of 10 | 0.00195 |
| Error map, pixel | SSIM | +0.024 | [+0.010, +0.037] | 90% of 10 | 0.00391 |
| Error map, + Sinkhorn 100 | PSNR (dB) | +0.809 | [+0.417, +1.201] | 100% of 10 | 0.00195 |
| Error map, + Sinkhorn 100 | SSIM | +0.018 | [+0.003, +0.032] | 90% of 10 | 0.00586 |
| Uniform, + Sinkhorn 100 | PSNR (dB) | -0.264 | [-0.485, -0.044] | 20% of 10 | 0.0273 |
| Uniform, + Sinkhorn 100 | SSIM | -0.007 | [-0.011, -0.003] | 0% of 10 | 0.00195 |
