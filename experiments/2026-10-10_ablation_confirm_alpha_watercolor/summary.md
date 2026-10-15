# Ablation summary

60 runs: 10 images x 2 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| orig | 19.45 +- 0.04 | 0.566 +- 0.002 | 0.502 +- 0.001 | 25.14 +- 0.05 | 13.08 +- 0.26 | 306 (30/30) | 417 (19/30) |
| low | 19.76 +- 0.04 | 0.572 +- 0.001 | 0.494 +- 0.002 | 25.29 +- 0.01 | 14.01 +- 0.27 | 290 (30/30) | 437 (21/30) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `orig` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| low | PSNR (dB) | +0.303 | [+0.202, +0.404] | 100% of 10 | 0.00195 |
| low | SSIM | +0.006 | [+0.002, +0.010] | 90% of 10 | 0.00391 |
| low | LPIPS | -0.009 | [-0.019, +0.002] | 90% of 10 | 0.0645 |
