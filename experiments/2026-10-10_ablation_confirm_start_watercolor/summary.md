# Ablation summary

60 runs: 10 images x 2 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| orig | 19.10 +- 0.08 | 0.553 +- 0.001 | 0.515 +- 0.003 | 24.40 +- 0.06 | 6.45 +- 0.05 | 362 (30/30) | 358 (10/30) |
| large | 19.45 +- 0.04 | 0.566 +- 0.002 | 0.502 +- 0.001 | 25.14 +- 0.05 | 6.44 +- 0.01 | 306 (30/30) | 417 (19/30) |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `orig` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| large | PSNR (dB) | +0.353 | [+0.227, +0.478] | 100% of 10 | 0.00195 |
| large | SSIM | +0.013 | [+0.000, +0.026] | 90% of 10 | 0.00586 |
| large | LPIPS | -0.012 | [-0.021, -0.004] | 90% of 10 | 0.0195 |
