# Ablation summary

60 runs: 10 images x 2 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | optimizer steps | peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prog_cos4 | 20.63 +- 0.03 | 0.580 +- 0.001 | 0.517 +- 0.002 | 24.85 +- 0.14 | 20.37 +- 0.26 | 2475 +- 0 | 0.175 +- 0.004 |
| prog_cos4_bf16 | 20.67 +- 0.03 | 0.579 +- 0.001 | 0.518 +- 0.003 | 24.87 +- 0.02 | 17.90 +- 0.20 | 2475 +- 0 | 0.099 +- 0.002 |

## Paired against `prog_cos4` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| prog_cos4_bf16 | PSNR (dB) | +0.039 | [-0.013, +0.091] | 70% of 10 | 0.275 |
| prog_cos4_bf16 | SSIM | -0.002 | [-0.003, -0.000] | 30% of 10 | 0.0488 |
| prog_cos4_bf16 | LPIPS | +0.001 | [-0.002, +0.004] | 50% of 10 | 0.625 |
