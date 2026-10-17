# Ablation summary

3 runs: 1 images x 3 configs x 1 seeds [0]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB | optimizer steps | peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prog500 | 23.73 +- 0.00 | 0.563 +- 0.000 | 0.518 +- 0.000 | 24.96 +- 0.00 | 45.52 +- 0.00 | - (0/1) | - (0/1) | 2475 +- 0 | 1.630 +- 0.000 |
| prog1000 | 24.42 +- 0.00 | 0.592 +- 0.000 | 0.483 +- 0.000 | 26.35 +- 0.00 | 118.83 +- 0.00 | - (0/1) | - (0/1) | 2430 +- 0 | 3.076 +- 0.000 |
| fixed500 | 22.06 +- 0.00 | 0.519 +- 0.000 | 0.542 +- 0.000 | 25.41 +- 0.00 | 109.03 +- 0.00 | 234 (1/1) | 365 (1/1) | 500 +- 0 | 3.410 +- 0.000 |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `prog500` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
