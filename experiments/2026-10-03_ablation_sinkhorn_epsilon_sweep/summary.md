# Ablation summary

75 runs: 5 images x 5 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pixel | 20.04 +- 0.15 | 0.559 +- 0.002 | 0.565 +- 0.001 | 25.37 +- 0.16 | 14.60 +- 0.09 | 303 (15/15) | 409 (12/15) |
| eps0.003 | 20.13 +- 0.11 | 0.563 +- 0.000 | 0.569 +- 0.001 | 25.52 +- 0.21 | 35.89 +- 1.42 | 305 (15/15) | 409 (12/15) |
| eps0.01 | 19.90 +- 0.08 | 0.554 +- 0.001 | 0.574 +- 0.002 | 24.93 +- 0.10 | 33.95 +- 1.69 | 379 (15/15) | 413 (9/15) |
| eps0.03 | 19.56 +- 0.09 | 0.532 +- 0.001 | 0.587 +- 0.001 | 23.46 +- 0.21 | 34.87 +- 1.87 | 362 (9/15) | 369 (6/15) |
| eps0.1 | 16.57 +- 0.05 | 0.400 +- 0.002 | 0.642 +- 0.002 | 15.46 +- 0.04 | 34.93 +- 2.15 | - (0/15) | - (0/15) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `pixel` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| eps0.003 | PSNR (dB) | +0.093 | [-0.015, +0.201] | 80% of 5 | 0.125 |
| eps0.003 | SSIM | +0.005 | [-0.007, +0.016] | 60% of 5 | 0.438 |
| eps0.003 | LPIPS | +0.004 | [-0.001, +0.010] | 0% of 5 | 0.0625 |
| eps0.01 | PSNR (dB) | -0.141 | [-0.468, +0.186] | 20% of 5 | 0.312 |
| eps0.01 | SSIM | -0.005 | [-0.010, +0.000] | 0% of 5 | 0.0625 |
| eps0.01 | LPIPS | +0.009 | [-0.004, +0.021] | 20% of 5 | 0.125 |
| eps0.03 | PSNR (dB) | -0.481 | [-0.882, -0.079] | 0% of 5 | 0.0625 |
| eps0.03 | SSIM | -0.027 | [-0.049, -0.004] | 0% of 5 | 0.0625 |
| eps0.03 | LPIPS | +0.022 | [-0.001, +0.045] | 0% of 5 | 0.0625 |
| eps0.1 | PSNR (dB) | -3.474 | [-4.347, -2.600] | 0% of 5 | 0.0625 |
| eps0.1 | SSIM | -0.159 | [-0.297, -0.020] | 0% of 5 | 0.0625 |
| eps0.1 | LPIPS | +0.077 | [+0.004, +0.149] | 0% of 5 | 0.0625 |
