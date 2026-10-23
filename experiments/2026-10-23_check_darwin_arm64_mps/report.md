# Device check

Run on 2026-10-23T14:05:03 (commit `3f879c9`, working tree dirty: True).

- device: `mps`; Python 3.14.4; torch 2.14.0; macOS-15.6-arm64-arm-64bit-Mach-O
- backends: cuda False, mps True; CPU threads 4
- reference checkout present: True; pretrained checkpoints: oilpaintbrush_light

| Step | Result | Seconds |
| :--- | :--- | ---: |
| environment | ok | 1.1 |
| dependencies | ok | 1.3 |
| renderers | ok | 4.1 |
| training | ok | 8.7 |
| painting | ok | 76.1 |

## Dependencies

torch 2.14.0, torchvision 0.29.0, numpy 2.5.2, scipy 1.18.1, opencv-python 5.0.0.93, pillow 12.3.0, pyyaml 6.0.3, einops 0.8.2, matplotlib 3.11.1, scikit-image 0.26.0, torchmetrics 1.9.0, lpips 0.1.4, imageio 2.37.4, imageio-ffmpeg 0.6.0, psutil 7.2.2, pytest 9.1.1, gradio 6.29.0

ffmpeg: `/Users/tony/Documents/Painting_project/venv/lib/python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1`

## Renderer speed (64 strokes per call)

| Device | Renderer | Forward (ms) | Forward + backward (ms) | Peak memory (GB) |
| :--- | :--- | ---: | ---: | ---: |
| mps | oilpaintbrush light | 17.7 | 26.8 | 0.11 |
| mps | oilpaintbrush full | - | - | checkpoint not downloaded |
| cpu | oilpaintbrush light | 23.4 | 95.2 | - |
| cpu | oilpaintbrush full | - | - | checkpoint not downloaded |

## Training smoke test

`epoch   2/2 | loss  6.86582 | psnr fg 15.94 alpha  9.50 mean 12.72* | ssim 0.288/0.009 | lr 2.0e-04 |    3.5 s |     921 strokes/s`

## Painting smoke test (apple, 5 x 5 grid)

| Loss | Device | Strokes | Steps | Optimize (s) | Final 512 px render (s) | PSNR (dB) | SSIM |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| pixel | mps | 100 | 500 | 14.2 | 0.3 | 16.57 | 0.433 |
| pixel + Sinkhorn | mps | 100 | 500 | 60.6 | 0.3 | 16.67 | 0.437 |
