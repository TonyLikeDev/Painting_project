# Ablation summary

27 runs: 3 images x 3 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low | 19.40 +- 0.15 | 0.662 +- 0.002 | 0.444 +- 0.002 | 23.86 +- 0.07 | 7.02 +- 0.03 | 384 (9/9) | 343 (3/9) |
| low2 | 19.41 +- 0.02 | 0.663 +- 0.002 | 0.436 +- 0.004 | 23.84 +- 0.06 | 7.03 +- 0.01 | 368 (9/9) | 353 (3/9) |
| low3 | 19.45 +- 0.02 | 0.658 +- 0.001 | 0.432 +- 0.002 | 23.84 +- 0.04 | 6.95 +- 0.15 | 354 (9/9) | 363 (3/9) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `low` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| low2 | PSNR (dB) | +0.011 | [-0.330, +0.352] | 67% of 3 | 0.75 |
| low2 | SSIM | +0.001 | [-0.030, +0.032] | 33% of 3 | 1 |
| low2 | LPIPS | -0.008 | [-0.076, +0.059] | 33% of 3 | 1 |
| low3 | PSNR (dB) | +0.053 | [-0.544, +0.650] | 33% of 3 | 1 |
| low3 | SSIM | -0.005 | [-0.060, +0.051] | 33% of 3 | 0.75 |
| low3 | LPIPS | -0.012 | [-0.117, +0.093] | 33% of 3 | 1 |
