# Ablation summary

60 runs: 10 images x 2 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pixel | 18.82 +- 0.08 | 0.601 +- 0.003 | 0.560 +- 0.002 | 25.28 +- 0.13 | 7.02 +- 0.02 | 282 (30/30) | 409 (20/30) |
| ot | 18.80 +- 0.08 | 0.601 +- 0.001 | 0.563 +- 0.002 | 25.32 +- 0.10 | 18.19 +- 0.13 | 273 (30/30) | 386 (20/30) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `pixel` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| ot | PSNR (dB) | -0.016 | [-0.121, +0.088] | 40% of 10 | 0.846 |
| ot | SSIM | +0.000 | [-0.003, +0.003] | 60% of 10 | 0.275 |
| ot | LPIPS | +0.002 | [-0.001, +0.006] | 40% of 10 | 0.275 |
