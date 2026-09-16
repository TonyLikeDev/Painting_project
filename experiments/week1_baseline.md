# Week 1 baseline: the original Stylized Neural Painting implementation

Reference rows for the "original implementation" column of every later comparison.
All runs use the authors' pretrained lightweight renderers and the original code,
patched only to run on modern PyTorch, on Apple MPS and on OpenCV 5, and to accept
a random seed (`experiments/original_repo_modern_torch.patch`).

## Protocol

```
python demo_prog.py --img_path ./test_images/apple.jpg --canvas_color <white|black> \
  --max_m_strokes 500 --max_divide 5 --renderer <brush> \
  --renderer_checkpoint_dir checkpoints_G_<brush>_light --net_G zou-fusion-net-light \
  --disable_preview --seed 0 --output_dir <dir>
```

Pixel loss only (`--with_ot_loss` off), progressive grid 1x1 to 5x5, 512 px canvas,
500 strokes, seed 0. Canvas colour follows the reference demos: white for oil paint
and watercolour, black for marker pen and coloured tape. Wall time covers the whole
process, including model loading and the final 500-frame PNG and MP4 render, which
is a substantial and roughly device-independent share of it.

`step_acc` is the value the original code prints; despite the name it is the PSNR in
dB between the current canvas and the target, measured on the block batch of the
final grid level, not on the finished 512 px image. It is recorded because it is the
only quality number the original code emits, and it makes the three devices
comparable to each other. Chapter 6 reports proper PSNR, SSIM and LPIPS on the
finished image instead.

## Results

Desktop rows: `experiments/2026-09-16_week1_baseline_desktop/` (runner:
`scripts/run_original_baselines.sh`, commit `c9d0528`). MacBook rows: 2026-09-05,
oil brush only, before the seed flag existed.

| Profile | Machine | Brush | Wall time (s) | Final G_loss | Final step_acc (dB) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| `cuda` | RTX 3070 | oil paint | 71.4 | 0.05062 | 22.04 |
| `cuda` | RTX 3070 | marker pen | 56.4 | 0.13856 | 14.02 |
| `cuda` | RTX 3070 | watercolour | 63.2 | 0.04904 | 22.45 |
| `cuda` | RTX 3070 | coloured tape | 60.2 | 0.04630 | 22.71 |
| `cpu` | i5-13400F | oil paint | 185.2 | 0.05118 | 22.00 |
| `cpu` | i5-13400F | marker pen | 168.4 | 0.17104 | 11.96 |
| `cpu` | i5-13400F | watercolour | 178.3 | 0.04919 | 22.43 |
| `cpu` | i5-13400F | coloured tape | 173.9 | 0.04656 | 22.74 |
| `mps` | Apple M4 | oil paint | 118.4 | 0.0503 | 22.24 |
| `cpu` | Apple M4 | oil paint | 252.3 | 0.0498 | 22.20 |

Device speed-ups on the oil brush, same work in each case:

| Comparison | Speed-up |
| :--- | ---: |
| RTX 3070 over the i5-13400F CPU | 2.6x |
| Apple M4 GPU over the Apple M4 CPU | 2.1x |
| RTX 3070 over the Apple M4 GPU | 1.7x |

## Observations

- **The GPU advantage is modest**, 2.6x on the desktop. The optimizer runs 500
  RMSprop steps on a few hundred stroke parameters through a small renderer, so the
  tensors are tiny and the workload is latency-bound rather than throughput-bound.
  The per-stroke CPU rasterization and PNG writing at the end are device-independent
  and cost the same everywhere. This is useful for the project: the system is usable
  without a GPU, which matters for the interface and the demo.
- **Quality is device-independent**, as it should be. Oil paint reaches 22.0 dB on
  all three profiles, and the small residual differences come from the order of
  floating-point reductions, not from the algorithm.
- **The marker pen is the outlier**: 14.0 dB on CUDA and 12.0 dB on the CPU, against
  22 dB for the other three brushes, with the loss three to four times higher. The
  brush paints flat, single-colour, tangent-aligned strokes on a black canvas, so it
  cannot represent smooth gradients, and the apple photograph is mostly smooth
  gradient. The gap between the two devices on this brush (2 dB) is much larger than
  on any other, which suggests that the marker runs are not fully converged and sit
  in a shallow part of the loss surface where run-to-run variation is large. Two
  things follow for later weeks. First, the aesthetic survey matters here: this
  brush is meant to look graphic rather than photographic, and PSNR punishes it for
  doing its job. Second, the marker pen is the brush most likely to benefit from the
  Sinkhorn term, so it should be included in the Week 5 ablation rather than dropped
  as a fallback.
- **Coloured tape scores highest on PSNR** (22.7 dB) despite being the crudest brush,
  which is a good reminder that pixel fidelity alone does not rank these brushes in
  the order a viewer would.

## Patches the 2021 code needed

1. `painter.py`: `torch.load(..., weights_only=False)`. PyTorch 2.6 changed the
   default to `True` and the 2020 checkpoint pickles a NumPy scalar, so loading
   failed with an `UnpicklingError`.
2. `painter.py`: load the checkpoint with `map_location='cpu'` and then move the
   network to the device. The original branched on `torch.cuda.is_available()`,
   which breaks for MPS.
3. `painter.py`, `networks.py`, `demo_prog.py`: device selection now prefers CUDA,
   then MPS, then CPU.
4. `renderer.py`: cast colour and alpha tuples to Python `float` before passing them
   to `cv2.circle` and `cv2.fillPoly`. OpenCV 5 rejects NumPy `float32` scalars as
   scalar colour arguments, which broke the watercolour, marker pen and tape
   rasterizers. The oil brush is unaffected because it goes through `warpAffine`,
   which is why the MacBook oil-only run in the first session did not surface this.
   Found by the pixel-parity test of the re-implemented rasterizer.
5. `demo_prog.py`: added a `--seed` flag that seeds Python, NumPy and PyTorch.
   Without it, stroke initialization and block shuffling are unseeded and no number
   is reproducible. Default `-1` keeps the original behaviour.

No dtype changes were needed: all stroke parameter tensors are already `float32`,
which MPS requires.

## Notes for later weeks

- Only `PixelLoss` and `SinkhornLoss` are constructed by default; the VGG losses are
  built only for style transfer, so `torchvision.models.vgg16(pretrained=True)` was
  never exercised. That call is deprecated and must be replaced with the `weights=`
  API when the Week 9 stretch goals start.
- `pytorch_batch_sinkhorn.py` line 55 has an invalid escape sequence (`\e` in a
  docstring). Harmless, fixed in the new package.
- Each run writes 500 per-stroke PNGs plus an MP4, 24 to 165 MB per output directory
  depending on the brush. The runner keeps only the final PNG, the stroke `.npz` and
  the log; frame dumps are regenerated on demand and never committed.
- The MacBook rows predate the `--seed` flag and are therefore unseeded. They are
  kept as the record of the first working baseline, but the desktop rows are the ones
  a later comparison should cite. Re-run the MacBook profile with `--seed 0` when
  that machine is next available, so all three profiles share one protocol.
