# Ablation summary

90 runs: 10 images x 3 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB | optimizer steps | peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 19.76 +- 0.04 | 0.572 +- 0.001 | 0.494 +- 0.002 | 25.29 +- 0.01 | 9.55 +- 0.70 | 290 (30/30) | 437 (21/30) | 500 +- 0 | 0.382 +- 0.002 |
| prog | 21.82 +- 0.01 | 0.656 +- 0.001 | 0.457 +- 0.002 | 25.15 +- 0.03 | 16.39 +- 2.09 | - (0/30) | - (0/30) | 2475 +- 0 | 0.174 +- 0.002 |
| prog_cos4 | 22.06 +- 0.05 | 0.667 +- 0.001 | 0.441 +- 0.002 | 25.60 +- 0.05 | 17.89 +- 0.86 | - (0/30) | - (0/30) | 2475 +- 0 | 0.174 +- 0.002 |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `fixed` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| prog | PSNR (dB) | +2.060 | [+1.175, +2.945] | 100% of 10 | 0.00195 |
| prog | SSIM | +0.083 | [+0.046, +0.121] | 100% of 10 | 0.00195 |
| prog | LPIPS | -0.037 | [-0.065, -0.008] | 80% of 10 | 0.00977 |
| prog_cos4 | PSNR (dB) | +2.304 | [+1.245, +3.363] | 100% of 10 | 0.00195 |
| prog_cos4 | SSIM | +0.095 | [+0.057, +0.133] | 100% of 10 | 0.00195 |
| prog_cos4 | LPIPS | -0.053 | [-0.090, -0.016] | 100% of 10 | 0.00195 |
