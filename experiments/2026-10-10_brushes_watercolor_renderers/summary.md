# Brush materials on the same images

3 images (51e3d4424d53b10ff3d8992cc12c30771037dbf85254784b7d297ed7934b_640, 54e2d3424c54a414f1dc8460962e33791c3ad6e04e507749712e79d29244c3_640, pexels-beratorer-30650522), seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds. Times are seconds per painting.

| Brush | Folder (config) | Strokes | PSNR (dB) | SSIM | LPIPS | optimize (s) | render (s) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original renderer | 2026-10-10_ablation_confirm_alpha_watercolor (low) | 500 | 20.41 +- 0.08 | 0.606 +- 0.002 | 0.477 +- 0.004 | 10.0 +- 0.1 | 1.58 +- 0.01 |
| Our renderer | 2026-10-10_ablation_heldout_ours_watercolor (paint) | 500 | 22.53 +- 0.04 | 0.645 +- 0.003 | 0.426 +- 0.001 | 6.6 +- 0.1 | 1.64 +- 0.05 |

## PSNR (dB) per image, seeds averaged

| Image | Original renderer | Our renderer |
| :--- | ---: | ---: |
| 51e3d4424d53b10ff3d8992cc12c30771037dbf85254784b7d297ed7934b_640 | 22.02 | 27.77 |
| 54e2d3424c54a414f1dc8460962e33791c3ad6e04e507749712e79d29244c3_640 | 18.97 | 19.66 |
| pexels-beratorer-30650522 | 20.25 | 20.15 |
