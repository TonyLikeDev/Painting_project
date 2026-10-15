# Brush materials on the same images

3 images (apple, iceland, jay), seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds. Times are seconds per painting.

| Brush | Folder (config) | Strokes | PSNR (dB) | SSIM | LPIPS | optimize (s) | render (s) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oil paint | 2026-10-10_ablation_brushes_oilpaintbrush (paint) | 500 | 20.86 +- 0.06 | 0.505 +- 0.002 | 0.548 +- 0.006 | 6.9 +- 0.0 | 1.46 +- 0.01 |
| Watercolour | 2026-10-10_ablation_brushes_watercolor (paint) | 500 | 20.03 +- 0.07 | 0.530 +- 0.003 | 0.526 +- 0.002 | 7.1 +- 0.1 | 1.59 +- 0.01 |
| Marker pen | 2026-10-10_ablation_brushes_markerpen (paint) | 500 | 19.40 +- 0.09 | 0.537 +- 0.001 | 0.567 +- 0.002 | 7.0 +- 0.1 | 1.64 +- 0.04 |
| Coloured tape | 2026-10-10_ablation_brushes_rectangle (paint) | 500 | 21.53 +- 0.10 | 0.581 +- 0.002 | 0.504 +- 0.003 | 7.1 +- 0.1 | 1.04 +- 0.00 |

## PSNR (dB) per image, seeds averaged

| Image | Oil paint | Watercolour | Marker pen | Coloured tape |
| :--- | ---: | ---: | ---: | ---: |
| apple | 20.70 | 19.96 | 19.14 | 21.01 |
| iceland | 20.40 | 19.29 | 18.57 | 20.83 |
| jay | 21.48 | 20.85 | 20.48 | 22.77 |
