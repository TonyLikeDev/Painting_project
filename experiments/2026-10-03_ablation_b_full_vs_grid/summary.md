# Ablation summary

60 runs: 10 images x 2 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full image (1 x 1) | 14.08 +- 0.11 | 0.435 +- 0.002 | 27.60 +- 0.70 | 76.39 +- 2.34 | 100 (30/30) | 150 (30/30) |
| 5 x 5 grid | 19.78 +- 0.03 | 0.556 +- 0.001 | 24.96 +- 0.05 | 14.22 +- 0.31 | 342 (30/30) | 407 (19/30) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `5 x 5 grid` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| Full image (1 x 1) | PSNR (dB) | -5.701 | [-7.015, -4.388] | 0% of 10 | 0.00195 |
| Full image (1 x 1) | SSIM | -0.121 | [-0.165, -0.078] | 0% of 10 | 0.00195 |
