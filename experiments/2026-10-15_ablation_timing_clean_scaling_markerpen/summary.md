# Ablation summary

12 runs: 3 images x 4 configs x 1 seeds [0]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | optimizer steps | peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s100 | 17.08 +- 0.00 | 0.505 +- 0.000 | 0.615 +- 0.000 | 17.85 +- 0.00 | 8.92 +- 0.00 | 2500 +- 0 | 0.037 +- 0.000 |
| s300 | 20.94 +- 0.00 | 0.575 +- 0.000 | 0.546 +- 0.000 | 23.43 +- 0.00 | 10.37 +- 0.00 | 2500 +- 0 | 0.097 +- 0.000 |
| s500 | 21.69 +- 0.00 | 0.594 +- 0.000 | 0.513 +- 0.000 | 25.06 +- 0.00 | 12.36 +- 0.00 | 2475 +- 0 | 0.173 +- 0.000 |
| s1000 | 21.93 +- 0.00 | 0.604 +- 0.000 | 0.485 +- 0.000 | 26.77 +- 0.00 | 17.14 +- 0.00 | 2430 +- 0 | 0.343 +- 0.000 |

## Paired against `s500` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| s100 | PSNR (dB) | -4.607 | [-6.045, -3.168] | 0% of 3 | 0.25 |
| s100 | SSIM | -0.089 | [-0.262, +0.084] | 0% of 3 | 0.25 |
| s100 | LPIPS | +0.102 | [+0.089, +0.115] | 0% of 3 | 0.25 |
| s300 | PSNR (dB) | -0.751 | [-2.005, +0.504] | 0% of 3 | 0.25 |
| s300 | SSIM | -0.018 | [-0.041, +0.005] | 0% of 3 | 0.25 |
| s300 | LPIPS | +0.034 | [+0.002, +0.066] | 0% of 3 | 0.25 |
| s1000 | PSNR (dB) | +0.236 | [-0.386, +0.859] | 67% of 3 | 0.5 |
| s1000 | SSIM | +0.010 | [-0.009, +0.029] | 100% of 3 | 0.25 |
| s1000 | LPIPS | -0.027 | [-0.029, -0.025] | 100% of 3 | 0.25 |
