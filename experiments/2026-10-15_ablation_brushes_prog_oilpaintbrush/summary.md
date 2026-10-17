# Ablation summary

90 runs: 10 images x 3 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB | optimizer steps | peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 20.30 +- 0.02 | 0.544 +- 0.001 | 0.528 +- 0.002 | 26.47 +- 0.02 | 11.50 +- 0.17 | 264 (30/30) | 374 (30/30) | 500 +- 0 | 0.382 +- 0.002 |
| prog | 20.56 +- 0.06 | 0.576 +- 0.003 | 0.521 +- 0.002 | 24.54 +- 0.10 | 20.02 +- 0.59 | - (0/30) | - (0/30) | 2475 +- 0 | 0.172 +- 0.000 |
| prog_cos4 | 20.63 +- 0.03 | 0.580 +- 0.001 | 0.517 +- 0.002 | 24.85 +- 0.14 | 20.11 +- 0.33 | - (0/30) | - (0/30) | 2475 +- 0 | 0.176 +- 0.000 |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `fixed` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| prog | PSNR (dB) | +0.262 | [+0.096, +0.428] | 90% of 10 | 0.00977 |
| prog | SSIM | +0.032 | [+0.005, +0.059] | 90% of 10 | 0.0137 |
| prog | LPIPS | -0.008 | [-0.029, +0.013] | 60% of 10 | 0.492 |
| prog_cos4 | PSNR (dB) | +0.332 | [+0.181, +0.484] | 100% of 10 | 0.00195 |
| prog_cos4 | SSIM | +0.036 | [+0.009, +0.064] | 90% of 10 | 0.00977 |
| prog_cos4 | LPIPS | -0.012 | [-0.029, +0.005] | 70% of 10 | 0.193 |
