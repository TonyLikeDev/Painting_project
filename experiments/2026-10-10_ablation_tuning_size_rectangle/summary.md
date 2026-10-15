# Ablation summary

27 runs: 3 images x 3 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| orig | 19.74 +- 0.04 | 0.660 +- 0.003 | 0.486 +- 0.002 | 23.25 +- 0.09 | 6.16 +- 0.21 | 449 (7/9) | 404 (3/9) |
| small | 19.50 +- 0.05 | 0.647 +- 0.002 | 0.499 +- 0.004 | 22.64 +- 0.09 | 6.02 +- 0.08 | 353 (4/9) | 437 (3/9) |
| large | 20.00 +- 0.04 | 0.674 +- 0.001 | 0.467 +- 0.003 | 24.01 +- 0.03 | 5.70 +- 0.03 | 413 (9/9) | 333 (3/9) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `orig` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| small | PSNR (dB) | -0.244 | [-0.366, -0.122] | 0% of 3 | 0.25 |
| small | SSIM | -0.014 | [-0.029, +0.002] | 0% of 3 | 0.25 |
| small | LPIPS | +0.013 | [+0.007, +0.019] | 0% of 3 | 0.25 |
| large | PSNR (dB) | +0.256 | [+0.094, +0.417] | 100% of 3 | 0.25 |
| large | SSIM | +0.014 | [-0.008, +0.036] | 100% of 3 | 0.25 |
| large | LPIPS | -0.019 | [-0.035, -0.003] | 100% of 3 | 0.25 |
