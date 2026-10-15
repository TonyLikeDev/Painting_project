# Ablation summary

60 runs: 10 images x 2 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pixel | 19.26 +- 0.02 | 0.585 +- 0.001 | 0.539 +- 0.001 | 25.01 +- 0.05 | 6.94 +- 0.07 | 321 (30/30) | 293 (16/30) |
| ot | 19.26 +- 0.02 | 0.585 +- 0.001 | 0.538 +- 0.002 | 24.92 +- 0.06 | 17.99 +- 0.01 | 330 (30/30) | 279 (13/30) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `pixel` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| ot | PSNR (dB) | -0.001 | [-0.045, +0.043] | 50% of 10 | 0.77 |
| ot | SSIM | -0.000 | [-0.003, +0.002] | 60% of 10 | 0.922 |
| ot | LPIPS | -0.001 | [-0.005, +0.003] | 80% of 10 | 0.232 |
