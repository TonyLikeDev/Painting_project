# Ablation summary

60 runs: 10 images x 2 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | optimizer steps | peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prog_cos4 | 22.06 +- 0.05 | 0.667 +- 0.001 | 0.441 +- 0.002 | 25.60 +- 0.05 | 20.78 +- 0.27 | 2475 +- 0 | 0.174 +- 0.002 |
| prog_cos4_bf16 | 21.93 +- 0.03 | 0.666 +- 0.001 | 0.440 +- 0.002 | 25.51 +- 0.04 | 17.65 +- 0.18 | 2475 +- 0 | 0.104 +- 0.002 |

## Paired against `prog_cos4` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| prog_cos4_bf16 | PSNR (dB) | -0.125 | [-0.208, -0.042] | 10% of 10 | 0.00391 |
| prog_cos4_bf16 | SSIM | -0.001 | [-0.005, +0.003] | 60% of 10 | 1 |
| prog_cos4_bf16 | LPIPS | -0.001 | [-0.005, +0.004] | 50% of 10 | 0.922 |
