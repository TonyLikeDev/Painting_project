# Brush materials on the same images

3 images (apple, iceland, jay), seeds [0, 1, 2]. Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds. Times are seconds per painting.

| Brush | Folder (config) | Strokes | PSNR (dB) | SSIM | LPIPS | optimize (s) | render (s) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oil paint | 2026-10-10_ablation_brushes_oilpaintbrush (paint) | 500 | 20.86 +- 0.06 | 0.505 +- 0.002 | 0.548 +- 0.006 | 6.9 +- 0.0 | 1.46 +- 0.01 |
| Watercolour | 2026-10-10_ablation_brushes_watercolor_final (paint) | 500 | 20.36 +- 0.10 | 0.535 +- 0.002 | 0.517 +- 0.001 | 8.6 +- 1.8 | 1.56 +- 0.04 |
| Marker pen | 2026-10-10_ablation_brushes_markerpen_fixed (paint) | 500 | 19.41 +- 0.27 | 0.539 +- 0.002 | 0.565 +- 0.003 | 7.1 +- 0.1 | 1.61 +- 0.03 |
| Coloured tape | 2026-10-10_ablation_brushes_rectangle (paint) | 500 | 21.53 +- 0.10 | 0.581 +- 0.002 | 0.504 +- 0.003 | 7.1 +- 0.1 | 1.04 +- 0.00 |

## PSNR (dB) per image, seeds averaged

| Image | Oil paint | Watercolour | Marker pen | Coloured tape |
| :--- | ---: | ---: | ---: | ---: |
| apple | 20.70 | 20.16 | 19.14 | 21.01 |
| iceland | 20.40 | 19.73 | 18.55 | 20.83 |
| jay | 21.48 | 21.18 | 20.53 | 22.77 |
