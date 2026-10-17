# Painting step profile (cuda)

| Variant | Optimize (s), mean of reps | PSNR (dB) |
| :--- | ---: | ---: |
| loop composite (as built) | 5.82 | 20.73 |
| closed-form composite | 5.23 | 20.48 |
| autocast float16 | 4.36 | 20.86 |
| autocast bfloat16 | 4.36 | 20.63 |

Profile of one painting: 143571 kernel launches; 7.25 s under the profiler, of which the GPU ran kernels for 4.72 s.

| CPU step, strokes | 128 | 256 | 384 | 512 | 128 | 256 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| not throttled | 0.093 | 0.165 | 0.244 | 0.320 | 0.093 | 0.179 |
| throttled | 0.335 | 0.627 | 0.937 | 1.235 | 0.326 | 0.622 |

| Painting on cuda | Optimize (s), mean | Rasterize (s), mean | PSNR (dB) |
| :--- | ---: | ---: | ---: |
| fixed, not throttled | 5.81 | 1.37 | 20.73 |
| progressive, not throttled | 12.12 | 1.32 | 21.01 |
| fixed, throttled | 7.03 | 2.69 | 20.73 |
| progressive, throttled | 21.42 | 2.76 | 21.01 |
