# Ablation summary

60 runs: 10 images x 2 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c10 | 17.91 +- 0.08 | 0.560 +- 0.003 | 0.559 +- 0.001 | 24.06 +- 0.07 | 6.99 +- 0.06 | 297 (24/30) | 250 (11/30) |
| c40 | 19.09 +- 0.05 | 0.602 +- 0.003 | 0.523 +- 0.003 | 25.43 +- 0.03 | 6.98 +- 0.03 | 270 (30/30) | 396 (23/30) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `c10` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| c40 | PSNR (dB) | +1.183 | [+0.951, +1.415] | 100% of 10 | 0.00195 |
| c40 | SSIM | +0.042 | [+0.020, +0.063] | 100% of 10 | 0.00195 |
| c40 | LPIPS | -0.037 | [-0.044, -0.029] | 100% of 10 | 0.00195 |
