# Ablation summary

90 runs: 10 images x 3 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | steps to 22 dB | steps to 24 dB | optimizer steps | peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 21.07 +- 0.02 | 0.634 +- 0.003 | 0.466 +- 0.003 | 25.92 +- 0.10 | 11.56 +- 0.15 | 298 (30/30) | 392 (27/30) | 500 +- 0 | 0.382 +- 0.002 |
| prog | 22.22 +- 0.02 | 0.670 +- 0.001 | 0.426 +- 0.002 | 25.27 +- 0.03 | 20.94 +- 0.27 | - (0/30) | - (0/30) | 2475 +- 0 | 0.173 +- 0.000 |
| prog_cos4 | 22.37 +- 0.04 | 0.676 +- 0.001 | 0.416 +- 0.002 | 25.55 +- 0.03 | 20.56 +- 0.03 | - (0/30) | - (0/30) | 2475 +- 0 | 0.176 +- 0.000 |

Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs).

## Paired against `fixed` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| prog | PSNR (dB) | +1.143 | [+0.676, +1.611] | 100% of 10 | 0.00195 |
| prog | SSIM | +0.036 | [+0.026, +0.046] | 100% of 10 | 0.00195 |
| prog | LPIPS | -0.040 | [-0.080, -0.001] | 90% of 10 | 0.00586 |
| prog_cos4 | PSNR (dB) | +1.297 | [+0.763, +1.832] | 100% of 10 | 0.00195 |
| prog_cos4 | SSIM | +0.042 | [+0.031, +0.053] | 100% of 10 | 0.00195 |
| prog_cos4 | LPIPS | -0.051 | [-0.093, -0.008] | 100% of 10 | 0.00195 |
