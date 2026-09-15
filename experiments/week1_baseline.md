# Week 1 baseline: original Stylized Neural Painting on the MacBook

Date: 2026-09-05
Machine: MacBook, Apple M4, 16 GB unified memory, macOS 24.6 (Darwin)
Environment: `venv/` in the project root, Python 3.14.4, torch 2.14.0, torchvision 0.29.0, opencv-python 5.0.0, numpy 2.5.2
Repo: `stylized-neural-painting/` (original 2021 code) with the patch in `original_repo_modern_torch.patch`
Renderer: `checkpoints_G_oilpaintbrush_light` (`zou-fusion-net-light`), downloaded from the authors' Google Drive

## Command

```
python demo_prog.py --img_path ./test_images/apple.jpg --canvas_color white \
  --max_m_strokes 500 --max_divide 5 --renderer oilpaintbrush \
  --renderer_checkpoint_dir checkpoints_G_oilpaintbrush_light \
  --net_G zou-fusion-net-light --disable_preview --output_dir <dir>
```

Pixel loss only (`--with_ot_loss` off), progressive grid 1x1 to 5x5, 512 px canvas, 500 strokes max.

## Results

| Profile | Wall time (s) | Final G_loss | Final step_acc (PSNR-like) | Output dir |
| :--- | ---: | ---: | ---: | :--- |
| `cpu` (M4) | 252.3 | 0.0498 | 22.20 | `stylized-neural-painting/output/` |
| `mps` (M4) | 118.4 | 0.0503 | 22.24 | `stylized-neural-painting/output_mps/` |

Speedup mps over cpu: 2.1x. Quality is the same within run-to-run noise (random stroke initialization, no fixed seed).

## Patches needed to run the 2021 code on modern PyTorch

1. `painter.py`: `torch.load(..., weights_only=False)`. PyTorch 2.6 made `weights_only=True` the default and the 2020 checkpoint pickles a numpy scalar, so loading failed with `UnpicklingError`.
2. `painter.py`: `map_location='cpu'` for the checkpoint, then `.to(device)` as before. Works for cuda, mps, and cpu.
3. `painter.py`, `networks.py`, `demo_prog.py`: device selection now prefers `cuda`, then `mps`, then `cpu`.

No dtype changes were needed: all stroke parameter tensors are already float32, which `mps` requires.

## Notes for the report

- Only `PixelLoss` and `SinkhornLoss` are built by default; the VGG losses are constructed only for style transfer, so `torchvision.models.vgg16(pretrained=True)` was not exercised. Check that call when Week 9 stretch goals start, since the `pretrained` argument is deprecated.
- `pytorch_batch_sinkhorn.py` line 55 has an invalid escape sequence warning (`\e` in a docstring). Harmless, but fix in the new package.
- Per-stroke frames (500 PNGs) plus an MP4 are written per run; about 340 MB per output directory.

## Still to do in Week 1

- Same baseline on the home desktop (RTX 3070) for the `cuda` row.
- Baselines for watercolor, marker pen, and tape renderers (download the other three lightweight checkpoints).
- Fix a random seed before any numbers go into a table.
