# Note: `painter.py`, `demo.py`, `demo_prog.py` (original repo)

## 1. Purpose

`painter.py` holds the **stroke-search engine**: it owns the neural renderer, the procedural renderer, the losses, and the stroke parameter tensors, and it implements error-map guided stroke initialisation, the differentiable forward pass with sequential compositing, and the final CPU rendering to images and video. `demo.py` (fixed grid), `demo_prog.py` (progressive grid), and `demo_nst.py` (style transfer) are thin argument parsers plus one `optimize_x` loop each that drive the three `PainterBase` subclasses.

## 2. Key classes and functions

| Name | Inputs / outputs | What it does |
| :--- | :--- | :--- |
| `PainterBase(args)` | - | Builds `renderer.Renderer(CANVAS_WIDTH=args.canvas_size=512)`, `net_G = define_G(...)`, `PixelLoss(p=1)`, `SinkhornLoss(0.01, 5)`, creates `output_dir`. |
| `_load_checkpoint()` | - | `torch.load(<dir>/last_ckpt.pt)`, loads `model_G_state_dict`, `eval()`. Exits the process if the file is missing. |
| `initialize_params()` | sets `x_ctt [m^2, S, d_shape]`, `x_color [m^2, S, d_color]`, `x_alpha [m^2, S, 1]` | Uniform `np.random.rand` on the device; `S = m_strokes_per_block`. |
| `stroke_sampler(anchor_id)` | writes row `anchor_id` of the three tensors | Error map `E = sum_c |img_batch - G_final_pred_canvas|` per grid cell, blurred with a box kernel of size `out_size / 8` (16 px for the full net, 4 px for the light one), raised to the **4th power**, then passed to `renderer.random_stroke_params_sampler` together with the target patch. |
| `_forward_pass()` | sets `G_pred_canvas [m^2, 3, W, W]` | Concatenates the three tensors to `x [m^2, S, d]`, reshapes strokes `0..anchor_id` to `[m^2 (anchor_id+1), d, 1, 1]`, runs `net_G`, applies `Dilation2d(m=1)` to the foreground and `Erosion2d(m=1)` to the alpha, reshapes to `[m^2, anchor_id+1, 3, W, W]`, and composites strokes in order: `canvas = fg * a + canvas * (1 - a)`. All strokes up to the current anchor are re-rendered and receive gradient every step. |
| `_backward_x()` | `G_loss` | `beta_L1 * pixel + (beta_ot * sinkhorn if with_ot_loss)`, then `backward()`. |
| `_compute_acc()` | PSNR | `utils.cpt_batch_psnr` between the predicted canvas batch and the target batch (single MSE over the whole batch). |
| `_normalize_strokes(v)` | `[m^2, S, d]` numpy | Maps per-cell coordinates to canvas coordinates: `y = y_id/m + y/m`, `x = x_id/m + x/m`, radii `/= m`. Index sets differ per brush (Bezier: xs `{0, 4}`, ys `{1, 5}`, rs `{6, 7}`; oil and tape: xs `{0}`, ys `{1}`, rs `{2, 3}`). Note the Bezier middle control point `(x1, y1)` is relative, so it is left untouched. |
| `_shuffle_strokes_and_reshape(v)` | `[1, m^2 S, d]` | Shuffles the grid-cell order with Python `random.shuffle`, then interleaves so stroke `j` of every cell comes before stroke `j+1`. |
| `_render(v, save_jpgs, save_video)` | final image `[H, W, 3]` | CPU rasterises every stroke with `check_stroke` + `draw_stroke` at 512 px, resizes each frame to the output size (aspect ratio restored only if `--keep_aspect_ratio`), writes one PNG per stroke, an MP4 (`MP4V`, 40 fps), the input, and the final image. |
| `_save_stroke_params(v)` | `<name>_strokes.npz` | Splits `v` into `x_ctt`, `x_color`, `x_alpha` and saves them. |
| `Painter(args)` | `demo.py` | Fixed `m_grid`; image resized to `(out_size * m_grid)^2` ignoring aspect ratio; `S = max_m_strokes / m_grid^2` (20 for 500 strokes at 5x5). |
| `ProgressivePainter(args)` | `demo_prog.py` | `S = max_m_strokes / sum_{i=1}^{max_divide} i^2` = `500 // 55 = 9` strokes per cell at every level; image resized to `(out_size * max_divide)^2`. |
| `NeuralStyleTransfer(args)` | `demo_nst.py` | Loads an `.npz`, builds `VGGStyleLoss`, optimises colour (and optionally shape) parameters against a 128x128 blurred style image with `_backward_x_sty`. |

**`demo_prog.optimize_x` loop.** For `m_grid = 1 .. max_divide`: split the target into `m_grid^2` patches; set the starting canvas to the patches of the *procedurally rendered* result of all previous levels (`CANVAS_tmp`); initialise parameters; `RMSprop(centered=True, lr=0.002)`; for each anchor `0..S-1`: sample one new stroke per cell, then `iters_per_stroke = int(500 / S)` (55 for `S = 9`) steps of: clamp shape params to **[0.1, 0.9]** and colour and alpha to [0, 1], forward, print, backward, clamp again, step. After the level, normalise, shuffle, append to `PARAMS`, and re-render everything on the CPU to get the next level's starting canvas. Finally save the `.npz` and render with video. Total gradient steps: `5 levels x 9 anchors x 55 iters = 2475`, and the number of strokes actually placed is `9 x 55 = 495`, not 500.

`demo.py` is the same loop at one fixed grid level with a plain (non-centred) `RMSprop` and with the canvas reset to solid white or black every iteration.

## 3. Data flow

- Inputs: `args` from the demo parser (image path, renderer name, checkpoint dir, `canvas_color`, `canvas_size`, `max_m_strokes`, `max_divide` or `m_grid`, `beta_L1`, `with_ot_loss`, `beta_ot`, `lr`, `output_dir`, `disable_preview`), brush textures via `renderer.py`, and the checkpoint via `_load_checkpoint`.
- Internal: `utils.img2patches` / `patches2img` for the grid, `morphology` for the 3x3 open/close of the neural output, `loss` for the objective, `networks` for the renderer.
- Outputs to `output_dir`: `<name>_strokes.npz`, `<name>_input.png`, `<name>_final.png`, `<name>_animated.mp4`, and `<name>_rendered_stroke_XXXX.png` for every stroke (about 340 MB per run with 500 strokes at 512 px, measured in Week 1).

## 4. Quirks, bugs and technical debt found

- Module-level `device` global duplicated across `painter.py`, `demo_prog.py`, `demo.py`, `networks.py`, `loss.py`, `pytorch_batch_sinkhorn.py`.
- `torch.load` without `weights_only=False` fails on PyTorch 2.6+ because the checkpoint pickles a NumPy scalar; `map_location=None` when CUDA exists forces CUDA. Both fixed by our patch, which also adds `mps` and `--seed`.
- `_shuffle_strokes_and_reshape` uses the unseeded Python `random`, `initialize_params` uses `np.random`, and the sampler uses both, so a run is not reproducible without seeding all three RNGs.
- `_drawing_step_states` prints every step and calls `utils.patches2img` (a CPU stitch of the whole canvas) even with `--disable_preview`, which is pure overhead in headless runs.
- `_render` rasterises every stroke on the CPU and writes a PNG per stroke; this dominates wall time for the light renderer and fills the disk.
- `self.img_path.split('/')[-1][:-4]` assumes forward slashes and a three-letter extension, so a Windows path with backslashes or a `.jpeg` file produces a wrong output name.
- `os.mkdir` (not `makedirs`) for the output dir; `exit()` inside a method for a missing checkpoint.
- `--lr` default is `0.002` but its help string says `0.005`.
- `max_m_strokes` is an upper bound only; the real count is `S * sum(i^2)`, and the printed "strokes: a / b" uses different meanings in `Painter` and `ProgressivePainter`.
- `Erosion2d` / `Dilation2d` mutate the renderer output in place with a per-channel Python loop (see `utils.md`).
- The progressive loop does not carry the optimiser state across levels and re-draws all `anchor_id + 1` strokes per step; cost per step grows linearly with the anchor index.
- There is no notion of a progress callback, seed, or structured logging; results are only visible through `print`.

## 5. What changes in the new `neural_painter` package

- `pipeline/stroke_sampler.py`: `error_map(target, canvas, blur_div=8, power=4)` and `sample_initial_strokes(...)` as pure functions with an explicit `numpy.random.Generator`.
- `pipeline/painter_engine.py`: `PainterEngine` with `optimize_block(target_patches, canvas_patches, n_strokes, iters_per_stroke)` returning stroke tensors and per-step metrics (loss, PSNR, wall time); Adam or RMSprop selectable; clamping ranges from the config; a `progress_callback(step_info)` hook for the UI.
- `pipeline/progressive_painter.py`: the `1x1 .. MxM` loop built on `core/grid.py` (`split` / `stitch` with aspect-ratio-preserving resize) with dynamic stroke allocation and learning-rate scheduling; the inter-level canvas is rendered by the differentiable canvas on the device, with the procedural render kept as an optional "domain-gap reset".
- `core/differentiable_canvas.py`: batched compositing of `[N, S, 3, H, W]` stroke stacks in one call.
- `export/video_recorder.py`: frames are rendered only when a video is requested, with a frame stride, straight to MP4 or GIF; no per-stroke PNG dump.
- Output naming via `pathlib.Path(...).stem`, `makedirs`, seeds and git commit hash logged per run in `experiments/<date>_<name>/`.
- `tests/test_optimizer.py`: loss decreases over 50 steps on a toy target.
