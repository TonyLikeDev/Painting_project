# Ablation summary

12 runs: 3 images x 4 configs x 1 seeds [0]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | optimizer steps | peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s100 | 14.48 +- 0.00 | 0.411 +- 0.000 | 0.660 +- 0.000 | 14.84 +- 0.00 | 9.23 +- 0.00 | 2500 +- 0 | 0.037 +- 0.000 |
| s300 | 21.96 +- 0.00 | 0.598 +- 0.000 | 0.519 +- 0.000 | 23.84 +- 0.00 | 10.85 +- 0.00 | 2500 +- 0 | 0.097 +- 0.000 |
| s500 | 23.06 +- 0.00 | 0.623 +- 0.000 | 0.479 +- 0.000 | 25.74 +- 0.00 | 12.67 +- 0.00 | 2475 +- 0 | 0.173 +- 0.000 |
| s1000 | 23.88 +- 0.00 | 0.646 +- 0.000 | 0.431 +- 0.000 | 28.00 +- 0.00 | 17.66 +- 0.00 | 2430 +- 0 | 0.343 +- 0.000 |

## Paired against `s500` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| s100 | PSNR (dB) | -8.580 | [-14.288, -2.871] | 0% of 3 | 0.25 |
| s100 | SSIM | -0.212 | [-0.220, -0.204] | 0% of 3 | 0.25 |
| s100 | LPIPS | +0.182 | [+0.085, +0.278] | 0% of 3 | 0.25 |
| s300 | PSNR (dB) | -1.097 | [-1.357, -0.837] | 0% of 3 | 0.25 |
| s300 | SSIM | -0.025 | [-0.054, +0.004] | 0% of 3 | 0.25 |
| s300 | LPIPS | +0.040 | [+0.018, +0.061] | 0% of 3 | 0.25 |
| s1000 | PSNR (dB) | +0.820 | [+0.255, +1.384] | 100% of 3 | 0.25 |
| s1000 | SSIM | +0.023 | [+0.006, +0.039] | 100% of 3 | 0.25 |
| s1000 | LPIPS | -0.048 | [-0.064, -0.032] | 100% of 3 | 0.25 |
