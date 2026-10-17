# Painting step profile (cuda)

| Variant | Optimize (s), mean of reps | PSNR (dB) |
| :--- | ---: | ---: |
| loop composite (as built) | 5.83 | 20.73 |
| closed-form composite | 5.26 | 20.48 |
| autocast float16 | 4.32 | 20.86 |
| autocast bfloat16 | 4.43 | 20.63 |

Profile of one painting: 143571 kernel launches; 7.19 s under the profiler, of which the GPU ran kernels for 4.72 s.

| CPU step, strokes | 128 | 256 | 384 | 512 | 128 | 256 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| not throttled | 0.095 | 0.168 | 0.252 | 0.314 | 0.093 | 0.170 |
| throttled | 0.330 | 0.636 | 0.941 | 1.258 | 0.331 | 0.636 |
