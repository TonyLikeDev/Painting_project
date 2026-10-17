# Ablation summary

4 runs: 1 images x 4 configs x 1 seeds [0]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB | optimizer steps | peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed500 | 22.06 +- 0.00 | 0.519 +- 0.000 | 0.542 +- 0.000 | 25.41 +- 0.00 | 80.25 +- 0.00 | 234 (1/1) | 365 (1/1) | 500 +- 0 | 3.495 +- 0.000 |
| fixed1000 | 22.65 +- 0.00 | 0.548 +- 0.000 | 0.516 +- 0.000 | 26.68 +- 0.00 | 368.86 +- 0.00 | 148 (1/1) | 222 (1/1) | 480 +- 0 | 6.811 +- 0.000 |
| prog500 | 23.73 +- 0.00 | 0.563 +- 0.000 | 0.518 +- 0.000 | 24.96 +- 0.00 | 252.49 +- 0.00 | - (0/1) | - (0/1) | 2475 +- 0 | 1.545 +- 0.000 |
| prog1000 | 24.42 +- 0.00 | 0.592 +- 0.000 | 0.483 +- 0.000 | 26.35 +- 0.00 | 144.93 +- 0.00 | - (0/1) | - (0/1) | 2430 +- 0 | 3.160 +- 0.000 |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `fixed500` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
