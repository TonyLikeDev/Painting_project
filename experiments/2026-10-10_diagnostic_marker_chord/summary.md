# Marker-pen strokes of a given chord length: are they drawn?

512 straight strokes per chord length (block units), 16 directions over 180 degrees, centres in [0.25, 0.75]. visible = share with largest opacity > 0.5; grad = mean norm of d(sum of opacity) / d(shape parameters).

| Chord | Renderer | Size | Visible | Mean largest opacity | Gradient |
| ---: | :--- | ---: | ---: | ---: | ---: |
| 0 | rasterizer | 32 | 0.00 | 0.000 | - |
| 0 | rasterizer | 128 | 0.00 | 0.000 | - |
| 0 | neural light | 32 | 0.00 | 0.001 | 0.685 |
| 0 | neural full | 128 | 0.83 | 0.687 | 7.62e+03 |
| 0.02 | rasterizer | 32 | 0.00 | 0.000 | - |
| 0.02 | rasterizer | 128 | 0.33 | 0.290 | - |
| 0.02 | neural light | 32 | 0.00 | 0.003 | 5.98 |
| 0.02 | neural full | 128 | 1.00 | 0.874 | 2.48e+04 |
| 0.05 | rasterizer | 32 | 0.00 | 0.000 | - |
| 0.05 | rasterizer | 128 | 1.00 | 0.887 | - |
| 0.05 | neural light | 32 | 0.00 | 0.021 | 162 |
| 0.05 | neural full | 128 | 1.00 | 0.899 | 1.26e+04 |
| 0.075 | rasterizer | 32 | 0.21 | 0.186 | - |
| 0.075 | rasterizer | 128 | 1.00 | 0.887 | - |
| 0.075 | neural light | 32 | 0.26 | 0.333 | 3.75e+03 |
| 0.075 | neural full | 128 | 1.00 | 0.905 | 1.1e+04 |
| 0.1 | rasterizer | 32 | 0.71 | 0.631 | - |
| 0.1 | rasterizer | 128 | 1.00 | 0.887 | - |
| 0.1 | neural light | 32 | 1.00 | 0.859 | 5.89e+03 |
| 0.1 | neural full | 128 | 1.00 | 0.912 | 1.08e+04 |
| 0.125 | rasterizer | 32 | 0.98 | 0.868 | - |
| 0.125 | rasterizer | 128 | 1.00 | 0.887 | - |
| 0.125 | neural light | 32 | 1.00 | 0.911 | 1.69e+03 |
| 0.125 | neural full | 128 | 1.00 | 0.919 | 1.11e+04 |
| 0.15 | rasterizer | 32 | 1.00 | 0.887 | - |
| 0.15 | rasterizer | 128 | 1.00 | 0.887 | - |
| 0.15 | neural light | 32 | 1.00 | 0.929 | 1.55e+03 |
| 0.15 | neural full | 128 | 1.00 | 0.925 | 1.13e+04 |
| 0.2 | rasterizer | 32 | 1.00 | 0.887 | - |
| 0.2 | rasterizer | 128 | 1.00 | 0.887 | - |
| 0.2 | neural light | 32 | 1.00 | 0.948 | 1.53e+03 |
| 0.2 | neural full | 128 | 1.00 | 0.936 | 1.2e+04 |
