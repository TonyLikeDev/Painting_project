# Ablation summary

60 runs: 10 images x 2 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| orig | 19.91 +- 0.06 | 0.539 +- 0.001 | 0.531 +- 0.001 | 25.29 +- 0.06 | 6.83 +- 0.01 | 319 (30/30) | 417 (24/30) |
| large | 20.30 +- 0.02 | 0.544 +- 0.001 | 0.528 +- 0.002 | 26.47 +- 0.02 | 6.90 +- 0.02 | 264 (30/30) | 374 (30/30) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `orig` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| large | PSNR (dB) | +0.389 | [+0.135, +0.643] | 100% of 10 | 0.00195 |
| large | SSIM | +0.005 | [+0.001, +0.008] | 80% of 10 | 0.0195 |
| large | LPIPS | -0.003 | [-0.009, +0.004] | 70% of 10 | 0.275 |
