# Ablation summary

45 runs: 3 images x 5 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| orig | 4.20 +- 0.00 | 0.008 +- 0.000 | 0.927 +- 0.012 | 4.18 +- 0.00 | 7.78 +- 0.26 | - (0/9) | - (0/9) |
| c10 | 17.11 +- 0.07 | 0.619 +- 0.003 | 0.548 +- 0.009 | 22.90 +- 0.17 | 7.70 +- 0.12 | 340 (6/9) | 372 (3/9) |
| c15 | 17.23 +- 0.23 | 0.622 +- 0.010 | 0.546 +- 0.010 | 23.56 +- 0.21 | 7.70 +- 0.07 | 407 (9/9) | 306 (3/9) |
| c20 | 17.52 +- 0.16 | 0.632 +- 0.006 | 0.538 +- 0.006 | 23.83 +- 0.27 | 7.84 +- 0.19 | 388 (9/9) | 298 (3/9) |
| c30 | 18.00 +- 0.14 | 0.637 +- 0.006 | 0.524 +- 0.009 | 23.97 +- 0.11 | 7.61 +- 0.18 | 363 (9/9) | 321 (3/9) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `orig` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| c10 | PSNR (dB) | +12.911 | [+7.035, +18.788] | 100% of 3 | 0.25 |
| c10 | SSIM | +0.611 | [+0.280, +0.943] | 100% of 3 | 0.25 |
| c10 | LPIPS | -0.379 | [-0.594, -0.164] | 100% of 3 | 0.25 |
| c15 | PSNR (dB) | +13.033 | [+7.485, +18.581] | 100% of 3 | 0.25 |
| c15 | SSIM | +0.614 | [+0.290, +0.938] | 100% of 3 | 0.25 |
| c15 | LPIPS | -0.381 | [-0.597, -0.166] | 100% of 3 | 0.25 |
| c20 | PSNR (dB) | +13.321 | [+7.858, +18.785] | 100% of 3 | 0.25 |
| c20 | SSIM | +0.624 | [+0.299, +0.949] | 100% of 3 | 0.25 |
| c20 | LPIPS | -0.389 | [-0.599, -0.179] | 100% of 3 | 0.25 |
| c30 | PSNR (dB) | +13.800 | [+8.539, +19.061] | 100% of 3 | 0.25 |
| c30 | SSIM | +0.629 | [+0.332, +0.927] | 100% of 3 | 0.25 |
| c30 | LPIPS | -0.403 | [-0.612, -0.193] | 100% of 3 | 0.25 |
