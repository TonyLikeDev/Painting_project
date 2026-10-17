# Ablation summary

60 runs: 10 images x 2 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) | optimizer steps | peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prog_cos4 | 21.16 +- 0.05 | 0.655 +- 0.002 | 0.444 +- 0.002 | 24.70 +- 0.05 | 20.31 +- 0.20 | 2475 +- 0 | 0.175 +- 0.004 |
| prog_cos4_bf16 | 21.24 +- 0.04 | 0.652 +- 0.000 | 0.448 +- 0.003 | 24.63 +- 0.03 | 17.62 +- 0.09 | 2475 +- 0 | 0.103 +- 0.002 |

## Paired against `prog_cos4` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| prog_cos4_bf16 | PSNR (dB) | +0.080 | [-0.042, +0.202] | 80% of 10 | 0.131 |
| prog_cos4_bf16 | SSIM | -0.002 | [-0.006, +0.001] | 30% of 10 | 0.275 |
| prog_cos4_bf16 | LPIPS | +0.004 | [-0.001, +0.009] | 20% of 10 | 0.084 |
