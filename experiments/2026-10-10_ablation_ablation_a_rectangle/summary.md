# Ablation summary

60 runs: 10 images x 2 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pixel | 20.88 +- 0.01 | 0.634 +- 0.001 | 0.500 +- 0.001 | 25.38 +- 0.06 | 7.06 +- 0.02 | 318 (30/30) | 407 (21/30) |
| ot | 20.92 +- 0.04 | 0.635 +- 0.002 | 0.503 +- 0.002 | 25.38 +- 0.00 | 18.30 +- 0.05 | 312 (30/30) | 399 (21/30) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `pixel` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| ot | PSNR (dB) | +0.047 | [-0.001, +0.094] | 80% of 10 | 0.084 |
| ot | SSIM | +0.001 | [-0.002, +0.003] | 50% of 10 | 1 |
| ot | LPIPS | +0.002 | [-0.003, +0.008] | 40% of 10 | 0.322 |
