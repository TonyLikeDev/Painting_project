# Device check

Run on 2026-10-02T14:11:36 (commit `543777c`, working tree dirty: False).

- device: `mps`; Python 3.14.4; torch 2.14.0; macOS-15.6-arm64-arm-64bit-Mach-O
- backends: cuda False, mps True; CPU threads 4
- reference checkout present: True; pretrained checkpoints: oilpaintbrush_light

| Step | Result | Seconds |
| :--- | :--- | ---: |
| environment | ok | 5.5 |
| dependencies | ok | 9.9 |
| tests | ok | 14.7 |
| renderers | ok | 5.0 |
| training | ok | 7.6 |

## Dependencies

torch 2.14.0, torchvision 0.29.0, numpy 2.5.2, scipy 1.18.1, opencv-python 5.0.0.93, pillow 12.3.0, pyyaml 6.0.3, einops 0.8.2, matplotlib 3.11.1, scikit-image 0.26.0, torchmetrics 1.9.0, lpips 0.1.4, imageio 2.37.4, imageio-ffmpeg 0.6.0, psutil 7.2.2, pytest 9.1.1, gradio 6.29.0

ffmpeg: `/Users/tony/Documents/Painting_project/venv/lib/python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1`

## Tests

{'passed': 136, 'skipped': 8}

Skipped:
- tests/test_renderer.py:185: /Users/tony/Documents/Painting_project/stylized-neural-painting/checkpoints_G_markerpen/last_ckpt.pt has not been downloaded
- tests/test_renderer.py:185: /Users/tony/Documents/Painting_project/stylized-neural-painting/checkpoints_G_markerpen_light/last_ckpt.pt has not been downloaded
- tests/test_renderer.py:185: /Users/tony/Documents/Painting_project/stylized-neural-painting/checkpoints_G_oilpaintbrush/last_ckpt.pt has not been downloaded
- tests/test_renderer.py:185: /Users/tony/Documents/Painting_project/stylized-neural-painting/checkpoints_G_rectangle/last_ckpt.pt has not been downloaded
- tests/test_renderer.py:185: /Users/tony/Documents/Painting_project/stylized-neural-painting/checkpoints_G_rectangle_light/last_ckpt.pt has not been downloaded
- tests/test_renderer.py:185: /Users/tony/Documents/Painting_project/stylized-neural-painting/checkpoints_G_watercolor/last_ckpt.pt has not been downloaded
- tests/test_renderer.py:185: /Users/tony/Documents/Painting_project/stylized-neural-painting/checkpoints_G_watercolor_light/last_ckpt.pt has not been downloaded
- tests/test_renderer.py:205: /Users/tony/Documents/Painting_project/stylized-neural-painting/checkpoints_G_oilpaintbrush/last_ckpt.pt has not been downloaded

## Renderer speed (64 strokes per call)

| Device | Renderer | Forward (ms) | Forward + backward (ms) | Peak memory (GB) |
| :--- | :--- | ---: | ---: | ---: |
| mps | oilpaintbrush light | 17.3 | 25.8 | 0.11 |
| mps | oilpaintbrush full | - | - | checkpoint not downloaded |
| cpu | oilpaintbrush light | 22.8 | 126.0 | - |
| cpu | oilpaintbrush full | - | - | checkpoint not downloaded |

## Training smoke test

`epoch   2/2 | loss  6.86582 | psnr fg 15.94 alpha  9.50 mean 12.72* | ssim 0.288/0.009 | lr 2.0e-04 |    2.4 s |    1316 strokes/s`
