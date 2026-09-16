# Note: `renderer.py` (original repo)

## 1. Purpose

`renderer.py` is the **procedural, non-differentiable ground-truth rasterizer**. It turns one stroke parameter vector in $[0,1]^d$ into an RGB foreground patch and an alpha map using OpenCV, composites it onto a NumPy canvas, and also holds the per-brush parameter layout (`d`, `d_shape`, `d_color`, `d_alpha`). It is used both to generate the synthetic training pairs for the neural renderer and to produce the final high-resolution painting after optimization.

## 2. Key classes and functions

| Name | Inputs / outputs | What it does |
| :--- | :--- | :--- |
| `_random_floats(low, high, size)` | list of Python floats | Uniform samples via `random.uniform` (Python RNG, not NumPy). |
| `_normalize(x, width)` | float in $[0,1]$ to int pixel | `int(x * (width - 1) + 0.5)`. |
| `Renderer(renderer, CANVAS_WIDTH=128, train=False, canvas_color='black')` | - | Sets `d / d_shape / d_color / d_alpha` = 12/8/3/1 (`markerpen`), 15/8/6/1 (`watercolor`), 12/5/6/1 (`oilpaintbrush`), 9/5/3/1 (`rectangle`). Oil loads four grayscale brush PNGs from the **relative** path `./brushes/`. Canvas is `float32 [W, W, 3]`: 128 when training the full renderer, 32 for the light one, 512 at painting time. |
| `create_empty_canvas()` | - | Ones (white) or zeros (black) canvas. |
| `random_stroke_params()` | sets `stroke_params` | $\theta \sim \mathcal{U}(0,1)^d$; used by the training dataset. |
| `random_stroke_params_sampler(err_map, img)` | `err_map [h, w]`, `img [h, w, 3]` | Resizes the error map to `W x W`, clips negatives, normalizes to a PDF, draws one pixel index with `np.random.choice`, converts to `(cx, cy)`. Bezier brushes put all three control points at that pixel; oil and tape use it as the centre. Sizes `U(0.1, 0.25)`, colour copied from the target pixel, alpha `U(0.98, 1.0)` for oil and watercolour, `U(0.8, 0.98)` for marker and tape, tape rotation fixed at 0. |
| `check_stroke()` | bool | Skip a stroke whose largest size parameter (indices 6, 7 for Bezier brushes; 2, 3 for oil and tape) is below **0.025**. |
| `_draw_watercolor()` | fills `foreground`, `stroke_alpha_map`, updates `canvas` | Control point $x_1 = x_0 + (x_2 - x_0)\,x_1$ (relative parameterization). Radii `int(1 + r * W // 4)`. **100 samples** along the quadratic Bezier, each an anti-aliased `cv2.circle` on `uint8` buffers with colour and radius linearly interpolated from end 0 to end 2. |
| `_draw_markerpen()` | same | Same curve, but a single radius (`radius2` is ignored) and a rotated square aligned to the tangent (`theta = arctan2(dx, dy) - pi/2`) is filled at each of the 100 samples. Draws nothing if $|x_0 - x_2| + |y_0 - y_2| < 4$ px. |
| `_draw_rectangle()` | same | `w, h = int(1 + v * W // 4)`, `theta = pi * theta`, four corners rotated with `utils.rotate_pt`, `cv2.fillPoly`. |
| `_draw_oilpaintbrush()` | same | `w, h = int(1 + v * W)` (full canvas scale, unlike the other brushes), `theta = pi * theta`. Picks the *large* texture when `w*h/W^2 > 0.1`, else *small*; vertical when `h > w`. Delegates to `utils.create_transformed_brush`. |
| `_update_canvas()` | canvas | $C \leftarrow F \odot A + C \odot (1 - A)$. |

At inference (`train=False`) every brush additionally dilates the foreground and erodes the alpha with a 2x2 kernel before compositing, which removes dark halos at anti-aliased edges.

## 3. Data flow

- **Training**: `utils.StrokeDataset` calls `random_stroke_params()` then `draw_stroke()` and reads back `foreground` and `stroke_alpha_map` as the targets `B` and `ALPHA` for `imitator.py`.
- **Painting**: `painter.PainterBase` owns one `Renderer` at `CANVAS_WIDTH=args.canvas_size` (512). `stroke_sampler` uses `random_stroke_params_sampler`; `_render` loops over all optimized strokes, calling `check_stroke` and `draw_stroke` on the CPU to produce the final image and the video frames.
- **Layout**: `d_shape / d_color / d_alpha` are read by `networks.define_G` to size the generator inputs and by `painter._save_stroke_params` to split the `.npz` file.

## 4. Quirks, bugs and technical debt found

- Brush textures are loaded with a cwd-relative path, so the demos only work when run from the repo root.
- Two RNGs are mixed: Python `random` for parameters and sizes, `np.random` for the error-map index. Neither is seeded anywhere, so no run is reproducible (our patch adds `--seed` to `demo_prog.py`).
- `markerpen` carries a `radius2` parameter that the rasterizer never uses; the neural renderer still receives it as input.
- Oil strokes use a different size scale (`* W`) than the other three brushes (`* W // 4`), so "w = 0.25" means a quarter of the canvas for oil but one sixteenth for tape.
- Everything is drawn on `uint8` buffers for anti-aliasing and converted back to `float32`, which quantizes colour gradients to 256 levels.
- `matplotlib.pyplot` is imported but unused; the `train` flag toggles the 2x2 morphology in four separate copies of the same three lines.
- The 100-sample Bezier loop issues 200 OpenCV draw calls per stroke; rendering 500 strokes at 512 px is the slow tail of every demo run.
- `check_stroke` uses a hard-coded 0.025 threshold that is not exposed as an argument.

## 5. What changes in the new `neural_painter` package

- `core/stroke_models.py`: typed dataclasses (`OilStroke`, `WatercolorStroke`, `MarkerStroke`, `TapeStroke`) with the same layouts, bounds validation, and `to_vector` / `from_vector`; the layout constants move to `configs/*.yaml`.
- `core/procedural_rasterizer.py`: same OpenCV drawing semantics (so the pretrained checkpoints stay valid) but with a `numpy.random.Generator` passed in for seeded sampling, brush textures resolved from the package directory, batched rasterization of many strokes, and the size scale and skip threshold as explicit parameters.
- `pipeline/stroke_sampler.py`: takes over `random_stroke_params_sampler` as a pure function of (error map, target image, RNG) that returns a tensor batch instead of mutating `stroke_params`.
- `core/differentiable_canvas.py`: the `_update_canvas` equation implemented once in torch and shared by the neural and the procedural path.
- `tests/test_rasterizer.py`: one stroke of each brush renders inside the canvas, alpha in $[0,1]$, foreground non-empty, deterministic under a fixed seed.
