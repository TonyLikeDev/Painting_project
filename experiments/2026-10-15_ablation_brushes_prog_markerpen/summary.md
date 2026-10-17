# Ablation summary

90 runs: 10 images x 3 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB | optimizer steps | peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 19.09 +- 0.05 | 0.602 +- 0.003 | 0.523 +- 0.003 | 25.43 +- 0.03 | 10.92 +- 0.40 | 270 (30/30) | 396 (23/30) | 500 +- 0 | 0.383 +- 0.002 |
| prog | 20.83 +- 0.06 | 0.646 +- 0.001 | 0.461 +- 0.003 | 24.36 +- 0.09 | 20.05 +- 0.32 | - (0/30) | - (0/30) | 2475 +- 0 | 0.173 +- 0.000 |
| prog_cos4 | 21.16 +- 0.05 | 0.655 +- 0.002 | 0.444 +- 0.002 | 24.70 +- 0.05 | 19.78 +- 0.42 | - (0/30) | - (0/30) | 2475 +- 0 | 0.175 +- 0.002 |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `fixed` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| prog | PSNR (dB) | +1.744 | [+0.666, +2.822] | 100% of 10 | 0.00195 |
| prog | SSIM | +0.044 | [+0.034, +0.053] | 100% of 10 | 0.00195 |
| prog | LPIPS | -0.062 | [-0.108, -0.016] | 100% of 10 | 0.00195 |
| prog_cos4 | PSNR (dB) | +2.068 | [+0.747, +3.389] | 100% of 10 | 0.00195 |
| prog_cos4 | SSIM | +0.053 | [+0.042, +0.063] | 100% of 10 | 0.00195 |
| prog_cos4 | LPIPS | -0.079 | [-0.134, -0.023] | 100% of 10 | 0.00195 |
