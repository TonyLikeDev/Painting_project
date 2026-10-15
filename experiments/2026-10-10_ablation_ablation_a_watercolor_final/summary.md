# Ablation summary

60 runs: 10 images x 2 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pixel | 19.59 +- 0.01 | 0.595 +- 0.001 | 0.524 +- 0.002 | 25.21 +- 0.05 | 14.90 +- 0.19 | 305 (30/30) | 246 (16/30) |
| ot | 19.63 +- 0.06 | 0.596 +- 0.000 | 0.524 +- 0.001 | 25.17 +- 0.03 | 40.33 +- 0.59 | 306 (30/30) | 253 (17/30) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `pixel` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| ot | PSNR (dB) | +0.038 | [-0.055, +0.131] | 60% of 10 | 0.492 |
| ot | SSIM | +0.001 | [-0.002, +0.004] | 60% of 10 | 0.322 |
| ot | LPIPS | -0.000 | [-0.004, +0.004] | 60% of 10 | 0.625 |
