# Ablation summary

45 runs: 5 images x 3 configs x 3 seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.

| Config | PSNR (dB) | SSIM | LPIPS | neural-canvas PSNR (dB) | optimize (s) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| original | 19.75 +- 0.14 | 0.552 +- 0.001 | 0.570 +- 0.005 | - | - |
| ours | 20.04 +- 0.04 | 0.558 +- 0.001 | 0.566 +- 0.004 | 25.39 +- 0.11 | 17.07 +- 0.35 |
| ours_linear | 19.73 +- 0.13 | 0.555 +- 0.001 | 0.567 +- 0.001 | 23.75 +- 0.17 | 13.34 +- 0.99 |

## Paired against `ours_linear` (one difference per image, seeds averaged)

| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |
| :--- | :--- | ---: | :--- | ---: | ---: |
| original | PSNR (dB) | +0.021 | [-0.173, +0.215] | 60% of 5 | 0.812 |
| original | SSIM | -0.003 | [-0.009, +0.002] | 40% of 5 | 0.312 |
| original | LPIPS | +0.003 | [-0.009, +0.015] | 40% of 5 | 0.625 |
| ours | PSNR (dB) | +0.318 | [+0.021, +0.616] | 100% of 5 | 0.0625 |
| ours | SSIM | +0.003 | [+0.001, +0.004] | 100% of 5 | 0.0625 |
| ours | LPIPS | -0.001 | [-0.007, +0.004] | 40% of 5 | 0.812 |
