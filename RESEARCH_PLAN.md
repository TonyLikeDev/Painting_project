# Research Plan: Photo-to-Painting with a Differentiable Neural Renderer

**Project title (from proposal):** Xây dựng hệ thống tái tạo tranh nghệ thuật từ ảnh thực tế bằng kỹ thuật cọ vẽ khả vi (Differentiable Neural Renderer)
**Student:** Nguyễn Lê Hoàng (24020003), class 24CSE, VNUK, University of Danang
**Supervisor:** Nguyễn Văn Thọ
**Academic year:** 2026-2027

This document merges the course proposal (`Do an co so Nguyen Le Hoang.pdf`) with the engineering design in `PLAN.md`. It keeps the proposal's 10-week timeline as the master schedule and pulls the architecture, stroke formulations, losses, directory layout, and modernization goals from `PLAN.md`. Every week lists what to build, what to measure, and what to write down, so the final report and a later research paper can be assembled from the weekly outputs instead of written from scratch.

---

## 0. Decisions that resolve conflicts between the two source documents

| Topic | Proposal says | PLAN.md says | Decision |
| :--- | :--- | :--- | :--- |
| Language and GUI | Section 4.1: Java/Python/C, Swing/JavaFX or console. Section 2.2: Web App / Desktop GUI | Python package, Gradio web UI | **Python 3.11+, PyTorch 2.x, Gradio web UI.** Section 4.1 of the proposal is template text and should be corrected to match. |
| Neural renderer | Week 4: "install / fine-tune" a renderer | Phase 2: train the renderer from scratch on synthetic strokes | **Both, in order.** Week 4 first loads the original pretrained checkpoints as a baseline, then trains our own renderer in the new package. Our own renderer is a paper contribution; the pretrained one is the fallback. |
| Style transfer, SVG export | Not mentioned | Phase 4 | **Stretch goals in Week 9.** They are not required for the course deliverable. |
| Evaluation and benchmarking | Weeks 7 and 9: aesthetic quality and processing time on several hardware configs | Not scheduled | **Added as first-class work** with named metrics, ablations, and a results table plan (Section 3). |
| Report, demo, defense | Week 10 | Not scheduled | **Kept in Week 10**, with drafting spread across Weeks 1 to 9. |
| Grid rendering order | Week 3, before the renderer | Phase 3, after the optimizer | **Week 3 builds the image splitter utility only.** The coarse-to-fine progressive loop is built in Week 7 once the optimizer exists. |
| Framework choice | PyTorch or TensorFlow | PyTorch 2.x | **PyTorch 2.x** with `mps`, `cuda`, and `cpu` backends. |

---

## 1. Objectives (from the proposal, unchanged)

**General objective.** Build a complete system that automatically converts a real photograph into a stroke-based painting using a deep-learning differentiable neural renderer, and shows the painting process as a real-time or time-lapse animation.

**Specific objectives.**
1. Master the theory of stroke-based rendering (SBR) and differentiable neural rendering, including Pixel Loss and Optimal Transport / Sinkhorn Loss.
2. Implement and tune a model that produces stroke parameter sequences (position, curvature, thickness, color, opacity) for oil paint, watercolor, marker pen, and colored tape.
3. Build an automatic pipeline: input, preprocessing, grid subdivision, stroke optimization, final rendering, and time-lapse export (GIF/MP4).
4. Build a user-friendly web interface with brush style, stroke density, and grid controls, and evaluate output quality and processing time on different hardware.

**Course deliverables.** Working software with GUI, a 40 to 60 page report (theory, system design, experiments), and a 5 to 10 minute demo video.

---

## 2. Technical design (condensed from PLAN.md, aligned with the reference repo)

### 2.1 Core idea
Standard vector rasterizers are not differentiable, so stroke parameters cannot be optimized by gradient descent. A neural renderer $G_\phi$ is trained as a differentiable surrogate: given parameters $\theta$ it outputs a foreground patch $\hat{F}$ and an alpha mask $\hat{A}$. Strokes are composited onto the canvas with

$$C_k = \hat{A}_k \odot \hat{F}_k + (1 - \hat{A}_k) \odot C_{k-1}$$

and $\theta$ is optimized against the target image with backpropagation.

### 2.2 Stroke parameterizations (match `stylized-neural-painting/renderer.py`)

| Brush | Repo name | Parameter vector | Dim | Shape / color / alpha split |
| :--- | :--- | :--- | :---: | :--- |
| Oil paint | `oilpaintbrush` | $x_c, y_c, w, h, \theta, R_0, G_0, B_0, R_2, G_2, B_2, A$ | 12 | 5 / 6 / 1 |
| Watercolor | `watercolor` | $x_0, y_0, x_1, y_1, x_2, y_2, r_0, r_2, R_0, G_0, B_0, R_2, G_2, B_2, A$ | 15 | 8 / 6 / 1 |
| Marker pen | `markerpen` | $x_0, y_0, x_1, y_1, x_2, y_2, r_0, r_2, R, G, B, A$ | 12 | 8 / 3 / 1 |
| Colored tape | `rectangle` | $x_c, y_c, w, h, \theta, R, G, B, A$ | 9 | 5 / 3 / 1 |

### 2.3 Losses
- **Pixel loss**: $L_1$ or Charbonnier between canvas and target (`loss.py: PixelLoss`).
- **Sinkhorn optimal transport loss**: entropy-regularized Wasserstein distance between canvas and target mass distributions, giving gradients even when a stroke does not overlap its target region (`loss.py: SinkhornLoss`, `pytorch_batch_sinkhorn.py`).
- **Perceptual / style loss** (stretch): VGG Gram matrix (`loss.py: VGGStyleLoss`).

### 2.4 Target package layout (from PLAN.md)

```
neural_painter/
├── configs/            oil_brush.yaml, watercolor.yaml, marker_pen.yaml, tape.yaml
├── core/               stroke_models.py, procedural_rasterizer.py, morphology.py, differentiable_canvas.py, image_io.py
├── models/             neural_renderer.py (FusionNet), vgg_perceptual.py
├── losses/             pixel_loss.py, sinkhorn.py, style_loss.py
├── pipeline/           train_renderer.py, stroke_sampler.py, painter_engine.py, progressive_painter.py
├── export/             video_recorder.py, svg_exporter.py (stretch)
├── app/                gradio_app.py
├── tests/              test_rasterizer.py, test_renderer.py, test_optimizer.py
├── scripts/            train_all_brushes.sh, run_paint.py, benchmark.py
├── experiments/        one folder per experiment: config.yaml, results.csv, figures/
├── pyproject.toml
└── README.md
```

### 2.5 Modernization goals carried over from PLAN.md
PyTorch 2.x with optional `torch.compile`; `mps` / `cuda` / `cpu` support; typed, tested, modular package with YAML configs; Gradio UI with stroke-by-stroke playback; SVG export and style transfer as stretch goals.

---

## 3. Research framing (what the paper will claim and measure)

### 3.1 Research questions
- **RQ1.** Does adding the Sinkhorn OT loss to the pixel loss improve reconstruction fidelity and convergence speed of stroke optimization, and by how much per brush type?
- **RQ2.** How does progressive coarse-to-fine grid rendering compare with single-pass full-image optimization in quality, stroke efficiency, and run time?
- **RQ3.** How do quality and run time scale with the stroke budget across brush materials?
- **RQ4.** Can a re-implemented, modernized renderer match the fidelity of the original 2021 pretrained renderer while running on consumer hardware (Apple Silicon, CPU) with lower memory?

### 3.2 Contributions to claim
1. A clean, modular, cross-platform re-implementation of differentiable stroke-based painting (PyTorch 2.x, `mps` / `cuda` / `cpu`).
2. A systematic ablation of loss functions and grid strategies across four brush materials.
3. A hardware benchmark of quality versus time versus memory on GPU, Apple Silicon, and CPU.
4. An interactive web system with real-time progress and time-lapse export.

### 3.3 Metrics
| Metric | Measures | Tool |
| :--- | :--- | :--- |
| PSNR, SSIM | Pixel fidelity to the target | `torchmetrics` or `skimage` |
| LPIPS | Perceptual similarity | `lpips` package |
| Sinkhorn distance | Distributional distance | our `losses/sinkhorn.py` |
| Renderer PSNR vs ground truth | Renderer fidelity to the procedural rasterizer | Week 4 test set |
| Strokes to reach PSNR X | Stroke efficiency | logged per run |
| Wall time (s), peak memory (GB) | Cost | `time`, `torch.cuda.max_memory_allocated`, `psutil` |
| Aesthetic score (1 to 5) | Human preference | small survey, at least 10 raters, Week 9 |

### 3.4 Planned ablations and comparisons
- Loss: pixel only, pixel + Sinkhorn, pixel + Sinkhorn + perceptual (stretch).
- Rendering mode: full image, fixed $M \times M$ grid, progressive $1 \to M$ grid.
- Stroke budget: 100, 300, 500, 1000 strokes.
- Brush material: oil, watercolor, marker, tape, on the same image set.
- Renderer: original pretrained checkpoint versus our retrained renderer.
- Hardware: CUDA GPU, Apple Silicon `mps`, CPU.

### 3.5 Test image set
Use the repo's `test_images/` (apple, sunflowers, diamond, monalisa, and the rest) plus 10 to 20 self-collected photos covering portraits, landscapes, still life, and high-texture scenes. Fix the set and resolution in Week 3 and never change it afterwards so all tables are comparable.

### 3.6 Experiment logging rules
- Every run writes `experiments/<date>_<name>/config.yaml`, `results.csv`, final PNG, stroke `.npz`, and a log with seed, device, and git commit hash.
- Fix random seeds. Report mean and standard deviation over 3 seeds for headline numbers.
- Never overwrite an experiment folder. Add a new one.

### 3.7 Hardware profiles and machine roles

| Profile | Machine | Device | Role |
| :--- | :--- | :--- | :--- |
| `cuda` | Home desktop: NVIDIA RTX 3070 (8 GB VRAM), Intel i5-13400F | CUDA | Renderer training, long optimization runs, CUDA benchmark rows |
| `mps` | MacBook: Apple M4, 16 GB unified memory | Metal (MPS) | Day-to-day development, UI work, Apple Silicon benchmark rows |
| `cpu` | Home desktop i5-13400F (and M4 CPU as a second data point) | CPU | CPU benchmark rows |

Rules for working across the two machines:
- Keep the code in a git repository with a remote so both machines run the same commit; log the commit hash in every experiment.
- Checkpoints and experiment outputs are not committed. Sync them with a shared folder or copy them manually and record the source.
- The 8 GB VRAM on the RTX 3070 is enough for the full-size renderers at inference and for training with batch size 32 to 64. If training runs out of memory, halve the batch size before changing anything else.
- Every headline table reports all three profiles; ablations may be run on `cuda` only and then spot-checked on `mps`.

---

## 4. Ten-week schedule (proposal timeline, PLAN.md content merged in)

Each week has four parts: **Build**, **Measure**, **Write**, and **Exit criteria**. "Write" items go straight into the report and paper.

### Week 1: Literature review and proposal finalization
**Build**
- Create the virtual environment; install `torch>=2.0`, `torchvision`, `opencv-python`, `numpy`, `scipy`, `einops`, `gradio`, `pyyaml`, `pytest`, `torchmetrics`, `lpips`.
- Clone `stylized-neural-painting`, download the four pretrained renderer checkpoints (full and lightweight), and run `demo_prog.py` on `test_images/apple.jpg` with `oilpaintbrush` to get a working baseline.
- Do the environment setup on both machines: the MacBook (`mps`) and the home desktop (`cuda`). Initialize a git repository with a remote so the two stay in sync.
- Record device details for each machine (GPU, driver and CUDA version, PyTorch version, RAM) in `experiments/hardware.md`.
- Patch the original repo's device selection (`painter.py`, `networks.py`, `demo_prog.py` hard-code CUDA-or-CPU) so it can also use `mps`. Keep the patch as a small diff for the report.

**Measure**
- Baseline run time and output for one image per brush using the original code, on all three profiles (`cuda` on the desktop, `mps` and `cpu` on the MacBook). Save these as the "original implementation" reference rows.

**Write**
- Annotated bibliography: Stylized Neural Painting (Zou et al. 2021), Learning to Paint (Huang et al. 2019), Paint Transformer (Liu et al. 2021), Neural Painters (Nakano 2019), Hertzmann 1998 painterly rendering, Sinkhorn distances (Cuturi 2013), differentiable rendering surveys.
- Correct Section 4.1 of the proposal to "Python, PyTorch, Gradio web app" and submit the final proposal.
- Draft the Related Work section outline.

**Exit criteria**: baseline demo runs end to end; proposal submitted; bibliography file exists.

### Week 2: Theory and system architecture
**Build**
- Create the `neural_painter/` package skeleton with all directories, empty modules, `pyproject.toml`, and four YAML brush configs holding the parameter dimensions from Section 2.2.
- Draw the system architecture diagram (optimization loop, modules, data flow) and the renderer architecture diagram.

**Measure**
- None. Read `renderer.py`, `networks.py`, `loss.py`, `pytorch_batch_sinkhorn.py`, and `painter.py` in the original repo and write a one-page note per file on what it does and what will change.

**Write**
- Theory chapter draft: stroke parameterization, compositing equation, pixel loss, Sinkhorn loss derivation and algorithm, why OT fixes vanishing gradients, style loss (short).
- Method section skeleton for the paper.

**Exit criteria**: package skeleton imports cleanly; two diagrams saved as SVG/PNG; theory draft of at least 8 pages.

### Week 3: Image input and preprocessing module
**Build**
- `core/image_io.py`: load JPG/PNG, sample-image picker, crop, resize with aspect-ratio option, normalize to $[0,1]$ tensors, device placement.
- `core/stroke_models.py`: dataclasses with bounds validation for all four brushes.
- `core/procedural_rasterizer.py`: OpenCV ground-truth rasterizer for Bézier and elliptical/rectangular strokes, brush texture loading, alpha and color-gradient interpolation, uniform random parameter sampling.
- `core/grid.py`: split an image into an $M \times M$ grid of patches and stitch results back. Only the splitter this week; the progressive loop comes in Week 7.
- `tests/test_rasterizer.py` and `tests/test_image_io.py`.

**Measure**
- Freeze the evaluation image set (Section 3.5) at a fixed resolution and record it in `experiments/dataset.md`.

**Write**
- Dataset and preprocessing subsection with a figure showing input, crop, normalized result, and grid split.

**Exit criteria**: tests pass; a random stroke of each brush type renders correctly through the procedural rasterizer; dataset frozen.

### Week 4: Differentiable renderer and basic stroke model (oil paint)
**Build**
- `models/neural_renderer.py`: FusionNet-style generator with a shape decoder (MLP + transposed convolutions or PixelShuffle producing $\hat{A}$) and a color decoder producing $\hat{F}$, and a fusion step $\hat{S} = \hat{A} \odot \hat{F}$. Include an adapter that loads the original `zou-fusion-net` checkpoints so both renderers share one interface.
- `pipeline/train_renderer.py`: on-the-fly synthetic ground truth from the procedural rasterizer, $L_1$ plus SSIM or cosine loss, device selection, checkpointing, validation images.
- `core/differentiable_canvas.py`: alpha compositing of a stroke batch onto a canvas.
- `tests/test_renderer.py`: single-stroke render, gradient flow check (finite differences versus autograd on a few parameters), and output range checks.
- Train the oil paint renderer on the home desktop (`cuda`, batch size 64, halve on out-of-memory). Use the MacBook for a short smoke-test run only, to confirm `mps` training works.

**Measure**
- Renderer fidelity: PSNR and SSIM of our renderer versus the procedural rasterizer on 1,000 held-out random strokes. Same numbers for the original pretrained renderer. Target: PSNR above the original or within 1 dB of it.
- Training curves (loss versus epoch) and training time per epoch per device.

**Write**
- Renderer architecture subsection with the diagram, a table of layer sizes, and the fidelity table (ours versus original). This is the first result for RQ4.

**Exit criteria**: gradient test passes; oil renderer checkpoint saved; fidelity table filled in.

### Week 5: Stroke optimization against the target image
**Build**
- `losses/pixel_loss.py` and `losses/sinkhorn.py` (batched, GPU/MPS-friendly, entropy-regularized, configurable `epsilon` and iteration count).
- `pipeline/stroke_sampler.py`: error map $E = \|I_{target} - C_k\|_1$ and probability-weighted stroke initialization.
- `pipeline/painter_engine.py`: single-block Adam optimization over stroke parameters, clamping to valid ranges, stroke ordering, checkpoint of stroke `.npz` files.
- `tests/test_optimizer.py`: loss decreases over 50 steps on a toy target.
- Run full-image mode and fixed-grid mode on the evaluation set with the oil brush.

**Measure**
- **Ablation A (RQ1)**: pixel only versus pixel + Sinkhorn, 3 seeds, 500 strokes, oil brush. Report PSNR, SSIM, LPIPS, and steps to reach a PSNR threshold.
- Loss-weight sweep for the Sinkhorn term (for example 0.1, 1, 10) and `epsilon` sweep.
- **Ablation B, first half (RQ2)**: full image versus fixed $M \times M$ grid.

**Write**
- Optimization subsection; ablation table A; a qualitative figure showing target, pixel-only result, and pixel + Sinkhorn result on three images.

**Exit criteria**: optimizer test passes; ablation A table complete with means and standard deviations.

### Week 6: Multi-material brushes
**Build**
- Fill in `configs/watercolor.yaml`, `configs/marker_pen.yaml`, `configs/tape.yaml` and the corresponding procedural rasterizers and texture assets.
- Train renderers for watercolor, marker pen, and tape on the home desktop via `scripts/train_all_brushes.sh`; if time is short, load pretrained checkpoints and fine-tune instead.
- Tune per-material properties: transparency, blur/bleed, stroke shape limits, default stroke budget.

**Measure**
- Renderer fidelity table extended to all four brushes.
- Same three images painted with each of the four brushes at 500 strokes: PSNR, SSIM, LPIPS, run time.

**Write**
- Brush material subsection: parameter table, material-property table, a 3 x 4 qualitative figure (images by brushes).
- Update the technical documentation of the render module.

**Exit criteria**: all four renderer checkpoints exist; comparison figure and table produced.

### Week 7: End-to-end automatic pipeline and performance
**Build**
- `pipeline/progressive_painter.py`: coarse-to-fine loop $1\times1 \to 2\times2 \to \dots \to M\times M$ with dynamic stroke allocation and learning-rate scheduling, built on `core/grid.py`.
- `scripts/run_paint.py`: one command from image to painting with a progress callback that reports strokes done, current loss, and elapsed time.
- Performance work: batch strokes per grid cell, optional `torch.compile`, mixed precision where supported, memory-aware batch sizing for `mps` and CPU.
- `scripts/benchmark.py`: runs the evaluation set under a named hardware profile and writes `results.csv`.

**Measure**
- **Ablation B, second half (RQ2)**: progressive grid versus full image versus fixed grid.
- **Scaling study (RQ3)**: stroke budget 100 / 300 / 500 / 1000 for each brush; quality and time curves.
- **Hardware benchmark, first pass (RQ4)**: time and peak memory on all three profiles, `cuda` on the desktop and `mps` and `cpu` on the MacBook, using `scripts/benchmark.py` with the same commit on both machines.

**Write**
- Pipeline subsection with the end-to-end diagram; ablation B table; quality-versus-strokes and time-versus-strokes plots.

**Exit criteria**: one command paints any image in the set with a live progress readout; benchmark CSV exists for at least two devices.

### Week 8: User interface
**Build**
- `app/gradio_app.py`: drag-and-drop upload or sample picker, brush selector, stroke-count slider, grid mode selector, canvas color, start button, live progress bar and intermediate canvas preview, before/after comparison, and a stroke-by-stroke playback slider.
- Wire the app to `progressive_painter.py` through the progress callback.
- Error handling for oversized images and unsupported formats.

**Measure**
- Usability test with 5 to 8 users: task completion, time to first painting, issues found. Fix the top issues.

**Write**
- System and interface section with annotated screenshots; user test summary.

**Exit criteria**: a non-developer can produce a painting from the UI without help.

### Week 9: Export, evaluation, and stretch goals
**Build**
- `export/video_recorder.py`: MP4 and GIF time-lapse from the blank canvas, with selectable frame stride and resolution.
- Download buttons in the UI for PNG/JPG, MP4, and GIF.
- Stretch, only if Weeks 4 to 8 are on schedule: `export/svg_exporter.py` for Bézier strokes; `losses/style_loss.py` and a style-transfer mode on stroke parameters (color only, color + texture).

**Measure**
- Full evaluation on the frozen set: our system versus the original implementation (pretrained renderer, original code) at equal stroke budgets: PSNR, SSIM, LPIPS, time.
- **Hardware benchmark, final**: rerun `scripts/benchmark.py` on the frozen set and final commit for `cuda` (RTX 3070), `mps` (M4), and `cpu` (i5-13400F).
- Aesthetic survey: at least 10 raters score outputs of four brushes on five images, 1 to 5.
- Collect user feedback, fix bugs, and re-run any affected experiments.

**Write**
- Complete the Experiments section: all tables and figures finalized, each with a caption and a one-paragraph interpretation. Draft Limitations.

**Exit criteria**: every table in Section 3 is filled; video export works from the UI.

### Week 10: Report, demo, and defense
**Build**
- Package the software: `pip install -e .`, README with quick start, `run_paint.py` and `gradio_app.py` entry points, checkpoint download script.
- Full regression test run.

**Measure**
- Final sanity run of the benchmark on the packaged build to make sure reported numbers match.

**Write**
- Assemble the 40 to 60 page report from Weeks 1 to 9: theory, system design, experiments, conclusion, appendix with configs.
- Record and edit the 5 to 10 minute demo video.
- Defense slides.
- Compile the paper draft using the outline in Section 5.

**Exit criteria**: report, video, slides, and packaged code delivered.

---

## 5. Paper outline and where each part comes from

| Paper section | Source weeks | Key figures and tables |
| :--- | :--- | :--- |
| Abstract | 10 | none |
| 1. Introduction | 1, 2 | teaser figure: photo, four brush outputs |
| 2. Related work | 1 | none |
| 3. Method: stroke model, neural renderer, losses | 2, 4, 5 | architecture diagram, parameter table |
| 4. System: pipeline, progressive rendering, UI | 3, 7, 8 | pipeline diagram, UI screenshot |
| 5. Experiments: setup | 3, 9 | dataset table, hardware table |
| 5.1 Renderer fidelity (RQ4) | 4, 6 | renderer PSNR table |
| 5.2 Loss ablation (RQ1) | 5 | ablation A table, qualitative figure |
| 5.3 Grid strategy (RQ2) | 5, 7 | ablation B table |
| 5.4 Stroke budget scaling (RQ3) | 7 | quality and time curves |
| 5.5 Brush materials | 6 | 3 x 4 qualitative grid |
| 5.6 Hardware benchmark (RQ4) | 7, 9 | time and memory table |
| 5.7 User study | 8, 9 | aesthetic score table |
| 6. Limitations and future work | 9 | none |
| 7. Conclusion | 10 | none |

---

## 6. Risks and fallbacks

| Risk | Signal | Fallback |
| :--- | :--- | :--- |
| Renderer training too slow on the development machine | Week 4 training exceeds 2 days per brush | Use the original pretrained checkpoints through the adapter for all painting experiments; report our renderer for oil paint only and note it as future work for the rest. |
| Sinkhorn loss too slow or unstable on `mps` | Optimization step time more than 5x the pixel-only step | Compute Sinkhorn on downsampled canvases, reduce iterations, or fall back to CPU for that term. |
| Home desktop unavailable when needed | A training or benchmark week arrives without access to it | Use a free Colab GPU for that run and label it clearly in the results, since it is not the RTX 3070 profile. |
| RTX 3070 runs out of VRAM during renderer training | CUDA out-of-memory at batch size 64 | Halve the batch size, then switch to the lightweight renderer architecture; report the configuration used. |
| Weeks 4 to 6 overrun | Week 7 starts late | Drop the Week 9 stretch goals first, then reduce the ablation grid to oil and marker brushes. |
| Aesthetic survey has too few raters | Fewer than 10 responses | Report it as a pilot study and rely on LPIPS as the perceptual metric. |

---

## 7. Weekly checklist template

Copy this into `experiments/week_XX.md` at the start of each week.

```
# Week XX: <title>
## Build
- [ ] ...
## Measure
- [ ] experiment name, config path, results path
## Write
- [ ] section drafted, figure saved to report/figures/
## Exit criteria
- [ ] ...
## Notes and deviations
```
