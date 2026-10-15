# Technical documentation: the render module

This note documents the part of `neural_painter` that turns stroke parameters into pixels: the brush
definitions, the procedural rasterizer, the neural renderer and the compositing step. It is the reference for
anyone who uses the four brushes or adds a fifth one. The design and the experiments are in Chapter 4
(Sections 4.1 and 4.3); this note says *where things are and how to use them*.

## 1. Files

| File | Role |
| :--- | :--- |
| `neural_painter/configs/<brush>.yaml` | One file per brush: parameter names and layout, renderer checkpoints, default canvas colour, the ranges new strokes start from (`sampler`), default stroke budget |
| `neural_painter/core/stroke_models.py` | `BrushSpec` (the YAML as an object: indices of the shape / colour / opacity parts, `split`, `join`), typed `Stroke` classes, `validate_params`, `clip_params` |
| `neural_painter/core/procedural_rasterizer.py` | `ProceduralRasterizer`: the OpenCV ground truth, a pixel-exact port of the 2021 `renderer.py`; `initial_params_at` (how a new stroke starts); `render_sequence` (the final 512 px painting) |
| `neural_painter/models/neural_renderer.py` | `NeuralRenderer` (the differentiable surrogate), `load_original_checkpoint` (the 2021 weights), `load_renderer` (either kind of checkpoint), `save_checkpoint`, `benchmark_renderer` |
| `neural_painter/core/differentiable_canvas.py` | `blank_canvas`, `composite`, `composite_sequence`: $C_k = A_k \odot F_k + (1 - A_k) \odot C_{k-1}$ |
| `neural_painter/core/morphology.py` | 3 x 3 dilation of the foreground and erosion of the opacity, the differentiable stand-in for the rasterizer's inference-time 2 x 2 step |
| `neural_painter/pipeline/train_renderer.py` | Training a renderer on strokes rasterized on the fly (Section 4.1.5) |
| `neural_painter/assets/brushes/*.png` | The four grey-scale oil-brush textures (small / large, vertical / horizontal) |

All parameters are in $[0, 1]$. Images are RGB `float32`, `(H, W, 3)` in NumPy and `(N, 3, H, W)` in PyTorch.

## 2. The four brushes

| | Oil paint | Watercolour | Marker pen | Coloured tape |
| :--- | :--- | :--- | :--- | :--- |
| YAML / repository name | `oil_brush.yaml` / `oilpaintbrush` | `watercolor.yaml` / `watercolor` | `marker_pen.yaml` / `markerpen` | `tape.yaml` / `rectangle` |
| Parameters (shape / colour / opacity) | 12 (5 / 6 / 1) | 15 (8 / 6 / 1) | 12 (8 / 3 / 1) | 9 (5 / 3 / 1) |
| Geometry | textured patch: centre, width, height, angle | quadratic Bezier, control point relative to the chord | quadratic Bezier, control point relative to the chord | rotated rectangle: centre, half extents, angle |
| Footprint | brush texture scaled to $(1 + wW) \times (1 + hW)$ px, rotated by $\pi\theta$; the large texture when it covers more than 10 % of the canvas | 100 anti-aliased discs along the curve, radius from $1 + r_0 W/4$ to $1 + r_2 W/4$ px | 100 anti-aliased squares of half width $1 + r_0 W/4$ px, turned with the curve tangent ($r_2$ is carried but unused) | one anti-aliased rectangle of half extents $1 + wW/4$, $1 + hW/4$ px |
| Colour | gradient along the brush from $(r_0, g_0, b_0)$ to $(r_2, g_2, b_2)$ | gradient along the curve from $(r_0, g_0, b_0)$ to $(r_2, g_2, b_2)$ | one colour | one colour |
| Opacity | always 1 (the parameter is carried for a uniform interface and forced to 1) | the parameter, uniform over the stroke | the parameter, uniform over the stroke | the parameter, uniform over the stroke |
| Not drawn when | largest size parameter $\le 0.025$ | largest radius parameter $\le 0.025$ | largest radius parameter $\le 0.025$, or end points less than 4 px apart (L1) | largest size parameter $\le 0.025$ |
| Default canvas | white | white | black | black |

$W$ is the canvas side in pixels (32 or 128 for the renderers' training images, 512 for the final painting).
Every footprint is drawn on `uint8` buffers (so OpenCV's anti-aliasing applies) and converted to `float32`; at
inference the foreground is dilated and the opacity eroded with a 2 x 2 kernel, which hides the seams between
strokes. Training images (`train=True`) skip that step.

What the brushes do *not* model: the rasterizer of the reference has no bleeding, wet-edge or paper-texture effect
for the watercolour and no stroke-end shading for the marker pen. Such effects could be added to the rasterizer,
but every pretrained renderer was trained on the reference rasterizer, so an effect would need a renderer
retrained on it. This is left as future work and is why "blur / bleed" is not among the tuned properties of
Section 4.3.

## 3. How a new stroke starts

`ProceduralRasterizer.initial_params_at(cx, cy, color, **overrides)` gives the parameters of a new stroke whose centre
the error map chose (`pipeline/stroke_sampler.py`). The colour is the target's at the centre. The ranges come from the
YAML `sampler` section, and a run can override them through `PainterConfig` (`init_size_range`, `init_alpha_range`,
`init_chord`) or `run_ablation.py` (`NAME=0,init_size_range=0.05:0.15`):

| Key | Meaning | Brushes |
| :--- | :--- | :--- |
| `size_range` | width / height (oil, tape) or the two radii (watercolour, marker) are drawn uniformly from it | all |
| `alpha_range` | starting opacity | all (oil ignores it) |
| `theta_range` | starting angle | oil, tape |
| `chord` | length of a new Bezier stroke, in block units; 0 starts it as a dot (the reference) | watercolour, marker pen |

The defaults after the Week 6 tuning (Section 4.3.3 of the report; each was checked on ten held-out images):

| | Oil paint | Watercolour | Marker pen | Coloured tape |
| :--- | :--- | :--- | :--- | :--- |
| `size_range` | [0.2, 0.4] (reference [0.1, 0.25]) | [0.2, 0.4] (reference [0.1, 0.25]) | [0.1, 0.25] (not tuned) | [0.2, 0.4] (reference [0.1, 0.25]) |
| `alpha_range` | [0.98, 1.0] (ignored) | [0.5, 0.7] (reference [0.98, 1.0]) | [0.8, 0.98] (tuned, unchanged) | [0.8, 0.98] (tuned, unchanged) |
| `theta_range` | [0, 1] | - | - | [0, 0] |
| `chord` | - | 0 (a dot) | 0.4 (reference 0) | - |

To paint exactly as the 2021 reference starts its strokes, pass `init_size_range=(0.1, 0.25)`, for the watercolour also
`init_alpha_range=(0.98, 1.0)`, and for the marker pen `init_chord=0`.

`chord` was added in Week 6. The reference starts a marker stroke with all three Bezier points on one pixel. The
rasterizer does not draw a marker stroke whose end points are less than 4 px apart, and the light (32 px) neural renderer
learnt this, so such a stroke renders nothing and gets no gradient: it never starts to grow
(`experiments/2026-10-10_diagnostic_marker_chord`). With `chord > 0` the stroke starts as a straight segment of that
length, in a random direction, with its control point in the middle, centred on the point the error map chose; near a
border the centre is moved inwards, axis by axis, only as far as needed for both ends to lie inside the painter's shape
range (default $[0.1, 0.9]$). `chord = 0` makes no extra random draw, so a seeded run gives exactly the strokes of the Week 5
painter (`tests/test_brush_init.py` checks the generator state); like the Week 3 port, that start matches the reference
sampler in distribution, not draw for draw, because the reference draws with Python's `random`. A start length above the
width of the shape range, a negative one, or one given for the oil brush or tape is refused.

## 4. Neural renderers

`NeuralRenderer(brush, light)` maps `(N, d)` parameters to a foreground and an opacity map `(N, 3, S, S)`, with $S = 32$
(light, about 5.4 M parameters) or 128 (full, about 18.1 M). A shape decoder sees the shape parameters and predicts a
mask; a colour decoder sees all parameters; the outputs are `colour * mask` and `alpha * mask` (`mask` alone for oil).

| Source | How to load | Notes |
| :--- | :--- | :--- |
| The authors' 2021 weights (`stylized-neural-painting/checkpoints_G_<brush>[_light]/last_ckpt.pt`) | `load_original_checkpoint(original_checkpoint_path(brush, light), brush, light)` | outputs identical to the reference network; loaded with `weights_only=True` |
| A renderer trained here (`checkpoints/<run>/best.pt`) | `load_renderer(path)` | brush, size and output activation are read from the file |

Train one with `python -m neural_painter.pipeline.train_renderer --brush <brush> --light --epochs 100` (resumable with
`--resume`, time-boxed with `--max-minutes`) and score it with `scripts/renderer_fidelity.py --ours LABEL=PATH`.

## 5. Adding a brush

1. Write `configs/<name>.yaml` with the parameter names, the shape / colour / opacity split, `layout` (which parameters are
   x and y coordinates and sizes, so that `core/grid.py` can move a stroke from a block to the whole image), the renderer
   section, the default canvas and the `sampler` ranges.
2. Add a `_draw_<name>` branch to `ProceduralRasterizer.render_stroke`, a branch to `initial_params_at` and to `is_visible`.
3. Add a `Stroke` subclass in `core/stroke_models.py` if typed strokes are wanted (tests, SVG export, UI).
4. Train its neural renderer with `train_renderer.py` and check it with `renderer_fidelity.py`.
5. The tests that loop over `BRUSHES` (`tests/conftest.py`) then cover it; add the name there.
