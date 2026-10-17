# Ablation summary

12 runs: 3 images x 4 configs x 1 seeds [0]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | optimizer steps | peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s100 | 17.08 +- 0.00 | 0.536 +- 0.000 | 0.611 +- 0.000 | 17.46 +- 0.00 | 9.07 +- 0.00 | 2500 +- 0 | 0.037 +- 0.000 |
| s300 | 22.09 +- 0.00 | 0.606 +- 0.000 | 0.510 +- 0.000 | 24.31 +- 0.00 | 10.36 +- 0.00 | 2500 +- 0 | 0.097 +- 0.000 |
| s500 | 22.94 +- 0.00 | 0.615 +- 0.000 | 0.484 +- 0.000 | 26.03 +- 0.00 | 12.46 +- 0.00 | 2475 +- 0 | 0.173 +- 0.000 |
| s1000 | 23.34 +- 0.00 | 0.632 +- 0.000 | 0.433 +- 0.000 | 27.96 +- 0.00 | 17.56 +- 0.00 | 2430 +- 0 | 0.343 +- 0.000 |

## Paired against `s500` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| s100 | PSNR (dB) | -5.864 | [-8.159, -3.569] | 0% of 3 | 0.25 |
| s100 | SSIM | -0.079 | [-0.142, -0.017] | 0% of 3 | 0.25 |
| s100 | LPIPS | +0.127 | [+0.104, +0.151] | 0% of 3 | 0.25 |
| s300 | PSNR (dB) | -0.852 | [-1.221, -0.483] | 0% of 3 | 0.25 |
| s300 | SSIM | -0.009 | [-0.015, -0.003] | 0% of 3 | 0.25 |
| s300 | LPIPS | +0.027 | [+0.024, +0.030] | 0% of 3 | 0.25 |
| s1000 | PSNR (dB) | +0.397 | [-0.338, +1.132] | 100% of 3 | 0.25 |
| s1000 | SSIM | +0.017 | [-0.006, +0.041] | 100% of 3 | 0.25 |
| s1000 | LPIPS | -0.050 | [-0.054, -0.047] | 100% of 3 | 0.25 |
