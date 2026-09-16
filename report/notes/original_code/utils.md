# Note: `utils.py` and `morphology.py` (original repo)

## 1. Purpose

`utils.py` is the grab-bag module: the synthetic stroke dataset and loaders for renderer training, image/patch conversion for the grid, PSNR/SSIM helpers, and the affine-warp code that stamps the oil-brush texture onto the canvas. `morphology.py` provides two small differentiable morphological operators (3x3 erosion and dilation) that the painter applies to the neural renderer's output to sharpen stroke edges.

## 2. Key classes and functions

| Name | Inputs / outputs | What it does |
| :--- | :--- | :--- |
| `M_RENDERING_SAMPLES_PER_EPOCH = 50000` | constant | Virtual epoch length; validation set is `1/20` of it (2500). |
| `PairedDataAugmentation(img_size, ...)` | two images to two tensors | Paired flips, rotations, crops for image-to-image tasks. **Not used** by the stroke dataset; also calls `TF.resize(..., interpolation=3)` with a raw integer, which newer torchvision rejects. |
| `StrokeDataset(args, is_train)` | dict `{'A': [d, 1, 1], 'B': [3, W, W], 'ALPHA': [3, W, W]}` | Creates a `renderer.Renderer(train=True)` with `CANVAS_WIDTH = 32` when `'-light' in args.net_G` else `128`. `__getitem__` ignores `idx`: it samples `random_stroke_params()`, calls `draw_stroke()`, and returns the parameters, the foreground, and the alpha map as tensors. The dataset is therefore infinite and generated on the fly, with no disk storage. |
| `get_renderer_loaders(args)` | `{'train': DataLoader, 'val': DataLoader}` | `batch_size = args.batch_size` (64 by default), `shuffle=True`, `num_workers=4`. |
| `set_requires_grad(nets, flag)` | - | Freezes the renderer during stroke search. |
| `make_numpy_grid`, `tensor2img` | tensor to numpy image | Visualisation helpers used by the imitator. |
| `cpt_ssim`, `cpt_psnr`, `cpt_cos_similarity` | numpy metrics | skimage SSIM; PSNR with `PIXEL_MAX=1.0`; cosine similarity. Used for validation in `imitator.py`. |
| `cpt_batch_psnr(img, img_gt, PIXEL_MAX)` | torch scalar | `20 log10(PIXEL_MAX / sqrt(MSE))` over the **whole batch** (one MSE), not the mean of per-image PSNRs. This is the "step_acc" printed by the painter. |
| `rotate_pt(pt, rotate_center, theta, return_int=True)` | `(x, y)` | Rotation with a sign convention `x' = dx cos + dy sin`, `y' = -dx sin + dy cos` (clockwise in image coordinates), result truncated to `int`. |
| `img2patches(img, m_grid, s)` | `[m^2, 3, s, s]` tensor | `cv2.resize` to `(m s, m s)` **ignoring aspect ratio**, then a double Python loop; cell index is `y_id * m + x_id` (row-major). |
| `patches2img(img_batch, m_grid, to_numpy=True)` | `[m s, m s, 3]` numpy or `[1, 3, m s, m s]` tensor | Inverse stitch; the tensor path is used by style transfer. |
| `create_transformed_brush(brush, canvas_w, canvas_h, x0, y0, w, h, theta, R0..B2)` | `(brush_rgb uint8, brush_alpha uint8)` | Alpha is the **binary** silhouette `brush > 0`; colour is a vertical gradient from `(R0, G0, B0)` to `(R2, G2, B2)` across the brush rows, multiplied by the grayscale texture (so the texture shows as darkening, not as transparency). The affine chain is translate to centre, scale `(w / bw, h / bh)`, rotate `theta`, translate `(x0, y0)`, applied with `cv2.warpAffine(INTER_AREA, BORDER_CONSTANT)`. |
| `build_scale_matrix`, `build_transformation_matrix`, `update_transformation_matrix` | 2x3 matrices | Compose affine transforms via 3x3 promotion. |
| `morphology.Erosion2d(m=1)` | `[B, C, H, W]` to same | Pads with `1e9`, `nn.Unfold(2m+1)`, `min` over the window, per channel in a Python loop, written back **into the input tensor**. |
| `morphology.Dilation2d(m=1)` | same | Same with `-1e9` padding and `max`. |

In the painter the foreground is dilated and the alpha eroded by one pixel (3x3) right after the renderer, mirroring the 2x2 `cv2.dilate` / `cv2.erode` that the procedural renderer applies at inference: the alpha shrinks so the soft anti-aliased rim disappears, and the foreground grows so no dark border is composited under the rim.

## 3. Data flow

- `StrokeDataset` to `imitator.Imitator` (training) via `get_renderer_loaders`; `train_imitator.py` wraps the loaders under `if __name__ == '__main__'`, which is required for `num_workers=4` on Windows and macOS (spawn start method).
- `img2patches` / `patches2img` to `painter.py` (target patches, preview stitching, style-transfer canvas) and `demo_prog.py` (re-splitting the rendered canvas at the next grid level).
- `create_transformed_brush` to `renderer._draw_oilpaintbrush` only.
- `cpt_batch_psnr` to `painter._compute_acc`; `cpt_psnr`, `cpt_ssim` to `imitator._compute_acc`.
- `morphology` to `painter._forward_pass` only.

## 4. Quirks, bugs and technical debt found

- `StrokeDataset` never resets `rderr.canvas`, so the renderer keeps compositing every sampled stroke onto one ever-growing canvas; the targets (`foreground`, `stroke_alpha_map`) are unaffected, but the compositing work per sample is wasted.
- Worker processes inherit the unseeded Python `random` state; PyTorch re-seeds `random` per worker, but NumPy is not re-seeded, so any NumPy sampling in workers would repeat across workers (the dataset happens to use only Python `random`).
- `img2patches` resizes with the default bilinear interpolation and squashes the image to a square; the aspect ratio is only restored at save time by resizing the *output*, so all optimisation happens on a distorted target.
- `Erosion2d` / `Dilation2d` write into their input with `x[:, [i], :, :] = channel`. It works because the input is a non-leaf activation and the saved tensors of `pad`/`unfold`/`min` do not need the original values, but it is an in-place op on an autograd node and the per-channel loop runs three unfolds where one `max_pool2d(kernel 3, stride 1, padding 1)` (and `-max_pool2d(-x)` for erosion) would do.
- `import pdb` left in `morphology.py`; `Subset`, `glob`, `plt` imported and unused in `utils.py`.
- `create_transformed_brush` rebuilds the colour map with a Python loop over brush rows on every stroke (394 or 197 rows) and converts through `uint8` twice.
- `rotate_pt` truncates to `int` before `fillPoly`, adding up to one pixel of jitter to rectangle and marker corners.
- `cpt_batch_psnr` reports batch-level PSNR, which is what the Week 1 baseline table calls "step_acc (PSNR-like)"; per-image PSNR would be the standard metric.

## 5. What changes in the new `neural_painter` package

- `core/image_io.py`: `load_image`, `resize` with an aspect-ratio option (letterbox or centre-crop), `to_tensor` / `to_numpy`, device placement; replaces the `cv2.imread` + `cvtColor` + `resize` copies in the painters.
- `core/grid.py`: `split_into_grid(img, m)` and `stitch_grid(patches, m)` as vectorised `unfold`-style reshapes on tensors, with the cell index convention documented and tested; the progressive loop in Week 7 builds on it.
- `core/morphology.py`: `erode(x, k)` / `dilate(x, k)` as functional wrappers over `max_pool2d`, out-of-place, one call for all channels, with a test that they match `cv2.erode` / `cv2.dilate` on a binary mask.
- `core/procedural_rasterizer.py`: takes over `create_transformed_brush` with a vectorised colour ramp (`np.linspace` outer product), textures loaded once from the package assets.
- `pipeline/train_renderer.py`: the on-the-fly dataset moves here as `SyntheticStrokeDataset(rasterizer, n_per_epoch, seed)` with a per-worker `numpy.random.Generator` seeded from `worker_info.seed`, and the canvas reset per sample.
- Metrics move to `torchmetrics` (`PeakSignalNoiseRatio`, `StructuralSimilarityIndexMeasure`) plus `lpips`, computed per image and averaged, as required by `RESEARCH_PLAN.md` section 3.3.
- `tests/test_image_io.py` and a grid round-trip test (`stitch(split(x)) == x`).
