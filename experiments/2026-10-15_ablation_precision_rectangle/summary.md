# Ablation summary

60 runs: 10 images x 2 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | optimizer steps | peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prog_cos4 | 22.37 +- 0.04 | 0.676 +- 0.001 | 0.416 +- 0.002 | 25.55 +- 0.03 | 20.54 +- 0.33 | 2475 +- 0 | 0.177 +- 0.006 |
| prog_cos4_bf16 | 22.35 +- 0.04 | 0.676 +- 0.001 | 0.415 +- 0.002 | 25.51 +- 0.04 | 17.90 +- 0.64 | 2475 +- 0 | 0.102 +- 0.000 |

## Paired against `prog_cos4` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| prog_cos4_bf16 | PSNR (dB) | -0.020 | [-0.103, +0.063] | 50% of 10 | 0.77 |
| prog_cos4_bf16 | SSIM | -0.000 | [-0.003, +0.002] | 50% of 10 | 0.922 |
| prog_cos4_bf16 | LPIPS | -0.001 | [-0.004, +0.003] | 70% of 10 | 0.557 |
