# Ablation summary

150 runs: 10 images x 5 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pixel only | 19.84 +- 0.06 | 0.556 +- 0.001 | 24.91 +- 0.11 | 11.78 +- 0.29 | 349 (30/30) | 409 (19/30) |
| Sinkhorn 0.1 | 19.83 +- 0.01 | 0.557 +- 0.001 | 24.94 +- 0.03 | 30.91 +- 0.53 | 340 (30/30) | 414 (20/30) |
| Sinkhorn 1 | 19.85 +- 0.02 | 0.557 +- 0.001 | 24.94 +- 0.07 | 30.19 +- 0.65 | 345 (30/30) | 412 (20/30) |
| Sinkhorn 10 | 19.84 +- 0.03 | 0.556 +- 0.000 | 25.01 +- 0.03 | 32.82 +- 1.11 | 338 (30/30) | 406 (19/30) |
| Sinkhorn 100 | 19.73 +- 0.02 | 0.552 +- 0.002 | 24.69 +- 0.06 | 29.12 +- 0.67 | 369 (30/30) | 419 (15/30) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `Pixel only` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| Sinkhorn 0.1 | PSNR (dB) | -0.011 | [-0.076, +0.054] | 50% of 10 | 0.846 |
| Sinkhorn 0.1 | SSIM | +0.000 | [-0.002, +0.003] | 50% of 10 | 1 |
| Sinkhorn 1 | PSNR (dB) | +0.016 | [-0.087, +0.118] | 60% of 10 | 1 |
| Sinkhorn 1 | SSIM | +0.001 | [-0.002, +0.004] | 60% of 10 | 0.77 |
| Sinkhorn 10 | PSNR (dB) | +0.004 | [-0.096, +0.105] | 40% of 10 | 1 |
| Sinkhorn 10 | SSIM | -0.000 | [-0.004, +0.003] | 50% of 10 | 0.77 |
| Sinkhorn 100 | PSNR (dB) | -0.104 | [-0.178, -0.030] | 20% of 10 | 0.0195 |
| Sinkhorn 100 | SSIM | -0.004 | [-0.008, -0.001] | 20% of 10 | 0.0195 |
