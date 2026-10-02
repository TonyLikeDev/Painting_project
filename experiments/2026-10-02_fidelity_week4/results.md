# Renderer fidelity (1000 held-out strokes per brush and size)

Device for the timing columns: `cuda`. PSNR is computed from the MSE over the whole set; the SSIM is the mean per stroke; "Reported" is the validation accuracy the authors stored in their checkpoint (mean of foreground and alpha PSNR on their own set).

| Renderer | Brush | Size | Params (M) | Epochs | Reported (dB) | PSNR fg | PSNR alpha | PSNR mean | SSIM fg | SSIM alpha | Fwd (ms) | Fwd+bwd (ms) | Peak mem (GB) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original full | oilpaintbrush | 128 | 18.09 | 43 | 25.26 | 27.85 | 22.56 | 25.20 | 0.942 | 0.934 | 7.6 | 15.6 | 0.50 |
| original light | oilpaintbrush | 32 | 5.37 | 400 | 24.31 | 27.38 | 21.38 | 24.38 | 0.944 | 0.920 | 1.2 | 2.6 | 0.09 |
| original full | watercolor | 128 | 18.12 | 484 | 30.83 | 30.87 | 30.49 | 30.68 | 0.973 | 0.970 | 7.7 | 15.9 | 0.50 |
| original light | watercolor | 32 | 5.39 | 400 | 25.84 | 25.80 | 25.72 | 25.76 | 0.925 | 0.917 | 1.2 | 2.6 | 0.09 |
| original full | markerpen | 128 | 18.09 | 198 | 28.75 | 28.84 | 28.58 | 28.71 | 0.962 | 0.953 | 7.7 | 15.8 | 0.50 |
| original light | markerpen | 32 | 5.37 | 400 | 24.83 | 24.82 | 24.36 | 24.59 | 0.914 | 0.903 | 1.2 | 2.5 | 0.09 |
| original full | rectangle | 128 | 18.07 | 412 | 33.61 | 33.59 | 33.46 | 33.53 | 0.987 | 0.985 | 7.7 | 15.8 | 0.50 |
| original light | rectangle | 32 | 5.34 | 400 | 28.23 | 28.39 | 28.27 | 28.33 | 0.959 | 0.957 | 1.2 | 3.1 | 0.09 |
| ours oil light | oilpaintbrush | 32 | 5.37 | 100 | - | 27.17 | 21.15 | 24.16 | 0.944 | 0.929 | 1.2 | 2.8 | 0.09 |
