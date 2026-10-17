# Painting step profile (cuda)

| Variant | Optimize (s), mean of reps | PSNR (dB) |
| :--- | ---: | ---: |
| loop composite (as built) | 5.85 | 20.73 |
| closed-form composite | 5.23 | 20.48 |
| autocast float16 | 5.61 | 20.59 |
| autocast bfloat16 | 5.63 | 20.55 |

Profile of one painting: 143571 kernel launches; 7.26 s under the profiler, of which the GPU ran kernels for 4.89 s.

| CPU step, strokes | 128 | 256 | 384 | 512 | 128 | 256 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| not throttled | 0.095 | 0.165 | 0.257 | 0.323 | 0.091 | 0.166 |
| throttled | 0.332 | 0.631 | 0.935 | 1.249 | 0.328 | 0.632 |

| Painting on cuda | Optimize (s), mean | Rasterize (s), mean | PSNR (dB) |
| :--- | ---: | ---: | ---: |
| fixed, not throttled | 5.78 | 1.23 | 20.73 |
| progressive, not throttled | 11.84 | 1.35 | 21.01 |
| fixed, throttled | 7.04 | 2.71 | 20.73 |
| progressive, throttled | 21.29 | 2.74 | 21.01 |
