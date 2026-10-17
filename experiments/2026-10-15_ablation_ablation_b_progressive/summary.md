# Ablation summary

180 runs: 10 images x 6 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB | optimizer steps | peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 20.31 +- 0.02 | 0.558 +- 0.001 | 0.564 +- 0.000 | 26.37 +- 0.04 | 11.08 +- 0.56 | 268 (30/30) | 356 (27/30) | 500 +- 0 | 0.382 +- 0.002 |
| full | 13.82 +- 0.25 | 0.444 +- 0.002 | 0.681 +- 0.002 | 22.65 +- 1.24 | 56.45 +- 0.78 | 137 (24/30) | 32 (11/30) | 500 +- 0 | 0.380 +- 0.000 |
| prog | 20.55 +- 0.04 | 0.570 +- 0.002 | 0.558 +- 0.004 | 24.46 +- 0.13 | 20.46 +- 0.36 | - (0/30) | - (0/30) | 2475 +- 0 | 0.179 +- 0.008 |
| prog_c | 20.56 +- 0.05 | 0.571 +- 0.000 | 0.558 +- 0.002 | 24.42 +- 0.06 | 20.48 +- 0.26 | - (0/30) | - (0/30) | 2475 +- 0 | 0.175 +- 0.004 |
| prog_short | 20.38 +- 0.02 | 0.559 +- 0.002 | 0.578 +- 0.001 | 23.67 +- 0.02 | 4.63 +- 0.17 | - (0/30) | - (0/30) | 495 +- 0 | 0.173 +- 0.002 |
| fixed_long | 20.21 +- 0.03 | 0.556 +- 0.001 | 0.556 +- 0.002 | 27.11 +- 0.06 | 55.86 +- 0.95 | 971 (30/30) | 1436 (30/30) | 2500 +- 0 | 0.385 +- 0.002 |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `fixed` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| full | PSNR (dB) | -6.485 | [-7.853, -5.116] | 0% of 10 | 0.00195 |
| full | SSIM | -0.114 | [-0.161, -0.067] | 0% of 10 | 0.00195 |
| full | LPIPS | +0.117 | [+0.073, +0.162] | 10% of 10 | 0.00391 |
| prog | PSNR (dB) | +0.246 | [+0.090, +0.402] | 90% of 10 | 0.00977 |
| prog | SSIM | +0.012 | [-0.007, +0.031] | 70% of 10 | 0.193 |
| prog | LPIPS | -0.006 | [-0.027, +0.015] | 50% of 10 | 0.922 |
| prog_c | PSNR (dB) | +0.256 | [+0.115, +0.398] | 100% of 10 | 0.00195 |
| prog_c | SSIM | +0.013 | [-0.005, +0.031] | 80% of 10 | 0.105 |
| prog_c | LPIPS | -0.006 | [-0.023, +0.011] | 50% of 10 | 0.77 |
| prog_short | PSNR (dB) | +0.069 | [-0.143, +0.282] | 60% of 10 | 0.557 |
| prog_short | SSIM | +0.001 | [-0.017, +0.020] | 40% of 10 | 0.695 |
| prog_short | LPIPS | +0.014 | [-0.003, +0.031] | 20% of 10 | 0.105 |
| fixed_long | PSNR (dB) | -0.100 | [-0.187, -0.013] | 30% of 10 | 0.0273 |
| fixed_long | SSIM | -0.002 | [-0.004, +0.000] | 30% of 10 | 0.084 |
| fixed_long | LPIPS | -0.008 | [-0.013, -0.004] | 90% of 10 | 0.00391 |
