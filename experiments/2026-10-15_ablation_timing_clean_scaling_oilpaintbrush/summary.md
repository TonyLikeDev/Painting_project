# Ablation summary

12 runs: 3 images x 4 configs x 1 seeds [0]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | optimizer steps | peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s100 | 16.70 +- 0.00 | 0.465 +- 0.000 | 0.643 +- 0.000 | 17.33 +- 0.00 | 8.59 +- 0.00 | 2500 +- 0 | 0.037 +- 0.000 |
| s300 | 20.73 +- 0.00 | 0.531 +- 0.000 | 0.555 +- 0.000 | 23.48 +- 0.00 | 9.84 +- 0.00 | 2500 +- 0 | 0.097 +- 0.000 |
| s500 | 21.40 +- 0.00 | 0.536 +- 0.000 | 0.540 +- 0.000 | 25.21 +- 0.00 | 11.72 +- 0.00 | 2475 +- 0 | 0.173 +- 0.000 |
| s1000 | 21.74 +- 0.00 | 0.542 +- 0.000 | 0.520 +- 0.000 | 27.32 +- 0.00 | 16.87 +- 0.00 | 2430 +- 0 | 0.343 +- 0.000 |

## Paired against `s500` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| s100 | PSNR (dB) | -4.704 | [-6.501, -2.907] | 0% of 3 | 0.25 |
| s100 | SSIM | -0.070 | [-0.166, +0.025] | 0% of 3 | 0.25 |
| s100 | LPIPS | +0.104 | [+0.035, +0.172] | 0% of 3 | 0.25 |
| s300 | PSNR (dB) | -0.674 | [-1.080, -0.269] | 0% of 3 | 0.25 |
| s300 | SSIM | -0.004 | [-0.013, +0.004] | 0% of 3 | 0.25 |
| s300 | LPIPS | +0.015 | [+0.002, +0.028] | 0% of 3 | 0.25 |
| s1000 | PSNR (dB) | +0.338 | [-0.344, +1.020] | 100% of 3 | 0.25 |
| s1000 | SSIM | +0.007 | [+0.004, +0.009] | 100% of 3 | 0.25 |
| s1000 | LPIPS | -0.020 | [-0.035, -0.005] | 100% of 3 | 0.25 |
