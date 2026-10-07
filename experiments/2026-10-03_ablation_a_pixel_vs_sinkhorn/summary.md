# Ablation summary

180 runs: 30 images x 2 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pixel | 18.67 +- 0.03 | 0.490 +- 0.001 | 0.536 +- 0.000 | 23.65 +- 0.05 | 6.17 +- 0.03 | 335 (75/90) | 416 (42/90) |
| ot | 18.69 +- 0.01 | 0.490 +- 0.001 | 0.536 +- 0.001 | 23.69 +- 0.05 | 15.99 +- 0.05 | 335 (75/90) | 400 (41/90) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `pixel` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| ot | PSNR (dB) | +0.011 | [-0.027, +0.049] | 47% of 30 | 0.984 |
| ot | SSIM | +0.001 | [-0.000, +0.002] | 60% of 30 | 0.164 |
| ot | LPIPS | -0.000 | [-0.002, +0.002] | 40% of 30 | 0.903 |
