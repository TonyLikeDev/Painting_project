# Ablation summary

60 runs: 10 images x 2 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| orig | 19.09 +- 0.05 | 0.602 +- 0.003 | 0.523 +- 0.003 | 25.43 +- 0.03 | 10.31 +- 0.32 | 270 (30/30) | 396 (23/30) |
| low | 19.18 +- 0.07 | 0.597 +- 0.002 | 0.536 +- 0.002 | 25.51 +- 0.03 | 10.09 +- 0.02 | 270 (30/30) | 396 (24/30) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `orig` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| low | PSNR (dB) | +0.089 | [-0.207, +0.385] | 70% of 10 | 0.193 |
| low | SSIM | -0.005 | [-0.016, +0.005] | 60% of 10 | 0.846 |
| low | LPIPS | +0.013 | [-0.007, +0.033] | 10% of 10 | 0.0488 |
