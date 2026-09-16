# Paper — Section 3: Method (skeleton)

Working title: *Photo-to-Painting with a Differentiable Neural Renderer: A Modular Re-implementation and Ablation of Stylized Neural Painting.*

This skeleton follows the paper outline in Section 5 of `RESEARCH_PLAN.md` (Method draws on Weeks 2, 4 and 5; the System section, not covered here, draws on Weeks 3, 7 and 8). Each subsection lists the theory subsection of `report/chapters/02_theory.md` it condenses and the weekly result that will fill it. Placeholders are in *italics*. Target length for the whole section: 1.5 to 2 pages in a two-column format, one architecture figure and one parameter table.

## 3.1 Problem formulation

*One paragraph.* Given a photograph $I$, find an ordered list of stroke vectors $\boldsymbol{\Theta} = (\boldsymbol{\theta}_1, \dots, \boldsymbol{\theta}_N)$, $\boldsymbol{\theta}_k \in [0,1]^d$, that minimizes $\mathcal{L}(\mathcal{R}(\boldsymbol{\Theta}), I)$ where $\mathcal{R}$ composites rasterized strokes with the "over" operator. State that $\mathcal{R}$ is non-differentiable and that we optimize through a learned surrogate $G_\phi$ instead; state the four brush materials handled by one pipeline.

- Source: theory 2.1 and 2.3.
- Filled by: Week 2 (formulation is fixed now; no experiment needed).
- Figure: teaser (photo and four brush outputs) is placed in the Introduction, not here.

## 3.2 Stroke model

*One paragraph plus Table 1.* Give the four parameter vectors and the shape / colour / opacity split (12 = 5+6+1 for oil, 9 = 5+3+1 for tape, 15 = 8+6+1 for watercolour, 12 = 8+3+1 for marker). Explain in two sentences the geometric meaning of each family (textured rotated patch, solid rotated rectangle, quadratic Bezier with discs or tangent-aligned squares), the relative control point $P_1 = P_0 + (P_2 - P_0) \odot (x_1, y_1)$, and the compositing equation $C_k = \hat{A}_k \odot F_k + (1 - \hat{A}_k) \odot C_{k-1}$.

- Source: theory 2.2 (Table 2.1) and 2.3.
- Filled by: Week 3 (`core/stroke_models.py` and `core/procedural_rasterizer.py`; the figure of one random stroke per brush rendered by the procedural rasterizer is the qualitative check).

## 3.3 Neural renderer

*Two paragraphs plus Figure 2 (architecture).* Describe the fusion renderer: shape decoder (MLP + PixelShuffle) fed only the shape parameters, colour decoder (transposed-convolution stack) fed the full vector, fusion $\hat{F} = c \odot m$, $\hat{A} = A\,m$. Give the training recipe in one sentence (on-the-fly uniform strokes, $L_2$ on foreground and opacity, 50k samples per epoch) and the full versus light variants (128 versus 32 pixel output). State that the pretrained 2021 checkpoints are loaded through the same interface as our retrained renderer so that both can be compared under identical painting code.

- Source: theory 2.4 (Table 2.2 gives the layer sizes; the paper keeps a condensed version).
- Filled by: Week 4 (`models/neural_renderer.py`, `pipeline/train_renderer.py`; renderer-fidelity table ours versus original, PSNR and SSIM on 1,000 held-out strokes, is reported in Experiments 5.1 for RQ4 and referenced from here). Week 6 extends it to the other three brushes.

## 3.4 Losses

*Two paragraphs.* (a) Pixel term: $L_1$ (reference) and Charbonnier (ours), with PSNR as the fidelity metric. (b) Optimal-transport term: canvas and target as normalized intensity measures on a $24 \times 24$ grid, squared-Euclidean cost, entropic regularization $\varepsilon = 0.01$, five log-domain Sinkhorn iterations, loss $\langle \pi, C \rangle$; one sentence on why the pixel gradient vanishes for non-overlapping strokes and OT does not. Total loss $\mathcal{L} = \beta_{L1}\mathcal{L}_{\text{pix}} + \beta_{\text{OT}}\mathcal{L}_{\text{OT}}$ with defaults $1$ and $0.1$. Style loss (Gram matrices) is mentioned in one sentence as an optional term used only for the style-transfer mode.

- Source: theory 2.5, 2.6 (the 1-D example stays in the report, not the paper) and 2.7.
- Filled by: Week 5 (`losses/pixel_loss.py`, `losses/sinkhorn.py`; Ablation A pixel-only versus pixel + OT, 3 seeds, 500 strokes, oil brush, and the $\beta_{\text{OT}}$ / $\varepsilon$ sweeps feed Experiments 5.2 for RQ1). Week 9 stretch fills the style-loss sentence if the mode is built.

## 3.5 Optimization and progressive rendering

*Two paragraphs plus Algorithm 1.* (a) Stroke initialization from the error map $E = \sum_c |I_c - C_c|$, box-blurred with kernel $S/8$, raised to the fourth power and normalized; initial colour from the target pixel. (b) Grid modes: full image, fixed $m \times m$ grid, and progressive $1 \times 1 \to M \times M$ with $N_{\text{cell}} = \lfloor N / \sum_{m=1}^{M} m^2 \rfloor$ strokes per cell, joint RMSprop updates of all strokes in a cell for $\lfloor 500 / N_{\text{cell}} \rfloor$ steps per added stroke, clamping to $[0.1, 0.9]$ for shape and $[0, 1]$ for colour and opacity, and the cell-to-canvas coordinate map (offset by cell index, divide sizes by $m$). Algorithm 1 is the progressive loop in pseudo-code.

- Source: theory 2.8.
- Filled by: Week 5 (`pipeline/stroke_sampler.py`, `pipeline/painter_engine.py`; Ablation B first half, full image versus fixed grid) and Week 7 (`pipeline/progressive_painter.py`; Ablation B second half and the stroke-budget scaling study feed Experiments 5.3 and 5.4 for RQ2 and RQ3).

## 3.6 Implementation details

*One paragraph.* PyTorch 2.x package with `cuda`, `mps` and `cpu` backends; optional `torch.compile` and mixed precision; strokes of all grid cells batched on the device; canvas between grid levels re-rasterized procedurally at full resolution; defaults (canvas 512, $N = 500$, $M = 5$, learning rate $0.002$, $\beta_{L1} = 1$, $\beta_{\text{OT}} = 0.1$); seeds fixed and every run logged with device, commit hash and config. Hardware profiles: RTX 3070 (8 GB), Apple M4 (16 GB unified), i5-13400F CPU.

- Source: RESEARCH_PLAN.md Sections 2.5, 3.6 and 3.7; theory 2.8.3 for the defaults.
- Filled by: Week 7 (`scripts/run_paint.py`, `scripts/benchmark.py`; hardware benchmark first pass) and Week 9 (final benchmark for Experiments 5.6, RQ4). Week 1 supplies the original-implementation reference rows from `experiments/week1_baseline.md`.

## Figures and tables owned by this section

| Item | Content | Produced in |
| :--- | :--- | :--- |
| Table 1 | Stroke parameter vectors and dimension splits for the four brushes | Week 2 (from theory Table 2.1) |
| Figure 2 | Fusion renderer architecture (shape branch, colour branch, fusion) and the optimization loop around it | Week 2 (`report/figures/renderer_architecture.*`, `report/figures/system_architecture.*`) |
| Algorithm 1 | Progressive coarse-to-fine stroke optimization | Week 7 |
