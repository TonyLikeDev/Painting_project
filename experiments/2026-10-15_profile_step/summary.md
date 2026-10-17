# Painting step profile (cuda)

| Variant | Optimize (s), mean of reps | PSNR (dB) |
| :--- | ---: | ---: |
| loop composite (as built) | 5.84 | 20.73 |
| closed-form composite | 5.22 | 20.48 |
| autocast float16 | 4.25 | 20.86 |
| autocast bfloat16 | 4.34 | 20.63 |

Profile of one painting: 143571 kernel launches, 5.49 s CPU, 9.43 s GPU kernels.

| CPU step, strokes | 128 | 256 | 384 | 512 | 128 | 256 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| not throttled | 0.098 | 0.169 | 0.246 | 0.321 | 0.092 | 0.165 |
| throttled | 0.325 | 0.637 | 0.924 | 1.234 | 0.330 | 0.628 |
