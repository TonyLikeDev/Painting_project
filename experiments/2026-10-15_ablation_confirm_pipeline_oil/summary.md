# Ablation summary

150 runs: 10 images x 5 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | optimizer steps | peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prog | 20.56 +- 0.06 | 0.576 +- 0.003 | 0.521 +- 0.002 | 24.54 +- 0.10 | 21.14 +- 1.21 | 2475 +- 0 | 0.174 +- 0.002 |
| prog_alloc | 20.64 +- 0.04 | 0.580 +- 0.002 | 0.514 +- 0.003 | 25.20 +- 0.02 | 28.19 +- 0.84 | 2466 +- 1 | 0.327 +- 0.004 |
| prog_cos4 | 20.63 +- 0.03 | 0.580 +- 0.001 | 0.517 +- 0.002 | 24.85 +- 0.14 | 20.84 +- 0.62 | 2475 +- 0 | 0.176 +- 0.006 |
| prog_alloc_cos4 | 20.67 +- 0.04 | 0.581 +- 0.000 | 0.511 +- 0.003 | 25.59 +- 0.06 | 29.09 +- 0.38 | 2467 +- 2 | 0.331 +- 0.005 |
| prog_i11_alloc | 20.55 +- 0.04 | 0.570 +- 0.003 | 0.529 +- 0.002 | 24.73 +- 0.05 | 10.31 +- 0.21 | 751 +- 9 | 0.323 +- 0.003 |

## Paired against `prog` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| prog_alloc | PSNR (dB) | +0.082 | [-0.004, +0.167] | 80% of 10 | 0.0645 |
| prog_alloc | SSIM | +0.004 | [-0.000, +0.008] | 80% of 10 | 0.0371 |
| prog_alloc | LPIPS | -0.007 | [-0.014, +0.000] | 90% of 10 | 0.0137 |
| prog_cos4 | PSNR (dB) | +0.070 | [+0.026, +0.114] | 80% of 10 | 0.00977 |
| prog_cos4 | SSIM | +0.005 | [+0.002, +0.007] | 80% of 10 | 0.00977 |
| prog_cos4 | LPIPS | -0.004 | [-0.008, +0.001] | 90% of 10 | 0.0645 |
| prog_alloc_cos4 | PSNR (dB) | +0.106 | [+0.034, +0.178] | 90% of 10 | 0.0137 |
| prog_alloc_cos4 | SSIM | +0.006 | [+0.002, +0.009] | 100% of 10 | 0.00195 |
| prog_alloc_cos4 | LPIPS | -0.010 | [-0.015, -0.005] | 90% of 10 | 0.00391 |
| prog_i11_alloc | PSNR (dB) | -0.007 | [-0.085, +0.070] | 40% of 10 | 0.77 |
| prog_i11_alloc | SSIM | -0.006 | [-0.011, -0.002] | 20% of 10 | 0.0195 |
| prog_i11_alloc | LPIPS | +0.008 | [+0.003, +0.013] | 30% of 10 | 0.0273 |
