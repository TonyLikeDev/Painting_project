# Ablation summary

9 runs: 3 images x 1 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| paint | 19.40 +- 0.09 | 0.537 +- 0.001 | 0.567 +- 0.002 | 24.28 +- 0.33 | 7.02 +- 0.10 | 341 (9/9) | 380 (4/9) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `paint` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
