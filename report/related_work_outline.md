# Related Work: Section Outline

Target length: 2 to 3 pages in the course report, about three quarters of a page in the paper. The order runs from classical heuristics, to learned painting agents, to differentiable renderers, then to the loss and stylization literature, so that the reader reaches our design choices (learned renderer, pixel plus Sinkhorn loss, progressive grid) already knowing the alternatives. Bracketed keys refer to `report/bibliography.md`.

## 2.1 Stroke-based rendering: from heuristics to optimization
- Argue that SBR represents an image as an ordered list of strokes rather than pixels [Haeberli 1990], was automated with gradient-aligned placement and edge clipping [Litwinowicz 1997], and gained coarse-to-fine layering that paints only where the residual to a reference is high [Hertzmann 1998].
- Argue that the taxonomy of [Hertzmann 2003] (greedy, energy minimization, rendering by example) frames the field, that the energy-minimization view is the one we take, and that its historic obstacle was a non-differentiable rasterizer that forced heuristic or stochastic search.
- Link forward: our error-map stroke sampler and the progressive 1x1 to MxM grid are the differentiable descendants of Hertzmann's rules.

## 2.2 Learning-based painting agents
- Argue that reinforcement learning agents learned stroke policies without renderer gradients [Xie et al. 2013; Ganin et al. 2018], at a high sample cost because every reward requires a call to a black-box renderer.
- Argue that a learned differentiable environment [Zheng et al. 2019] or renderer [Huang et al. 2019] made agent training gradient-based and far cheaper; the Bézier stroke parameterization and pixel-shuffle renderer of Learning to Paint are reused by later work, including the shape decoder in ours.
- Argue that feed-forward set prediction [Liu et al. 2021] removes per-image optimization for speed but restricts stroke materials, and that loss shaping [Schaldenbrand and Oh 2021] shows stroke order can be controlled through the objective alone.
- Position: these methods amortize the search into a network; our method, following [Zou et al. 2021], searches per image, which is slower but supports arbitrary materials and objectives and needs no policy training.

## 2.3 Differentiable rendering and neural renderers
- Argue that differentiable rendering is either analytic, as in the vector rasterizer of [Li et al. 2020] and the 3D methods surveyed by [Kato et al. 2020], or a learned surrogate, as in Neural Painters [Nakano 2019].
- Argue that learned surrogates trade exact gradients for arbitrary brush textures and a smoother loss landscape, and that the dual-pathway renderer of [Zou et al. 2021] improves fidelity for textured strokes over single-branch designs.
- Position: we re-implement the dual-pathway renderer in PyTorch 2.x, train it from scratch on synthetic strokes, and measure its fidelity against the procedural rasterizer and against the original checkpoints (RQ4).

## 2.4 Losses for stroke optimization
- Argue that pixel losses have vanishing gradients when a stroke does not overlap its target region, and that entropic optimal transport [Cuturi 2013; Peyré and Cuturi 2019] gives a distributional distance whose gradient is non-zero at any distance and is cheap enough for an inner optimization loop.
- Argue that [Zou et al. 2021] introduced the Sinkhorn term for painting but reported only a limited ablation; our RQ1 quantifies its effect per brush material, per loss weight and per epsilon.

## 2.5 Stylization and evaluation
- Argue that the Gram-matrix style loss of [Gatys et al. 2016] transfers from pixels to stroke parameters [Zou et al. 2021; Kotovenko et al. 2021], which is our Week 9 stretch goal.
- Argue that evaluating paintings needs perceptual metrics: SSIM [Wang et al. 2004] and LPIPS [Zhang et al. 2018] complement PSNR, and human ratings remain the reference, hence the aesthetic survey.

## Positioning of this work
Prior work established the components separately: classical SBR gave the coarse-to-fine, residual-driven placement rules; learning-based agents showed that a differentiable renderer turns stroke search into gradient descent; and Stylized Neural Painting combined a dual-pathway neural renderer with pixel and Sinkhorn losses and progressive grid rendering. What is missing is a systematic account of how much each of these pieces contributes and what they cost on ordinary hardware. The original implementation targets a 2020 PyTorch and CUDA stack, reports no ablation of the Sinkhorn weight or the grid strategy across brush materials, and offers no interactive interface. This project fills that gap in four ways, matching the contributions in RESEARCH_PLAN.md section 3: (1) a clean, modular, cross-platform re-implementation on PyTorch 2.x with CUDA, Apple Silicon (MPS) and CPU back ends, including a renderer retrained from scratch and compared with the original checkpoints; (2) a systematic ablation of loss functions (pixel only, pixel plus Sinkhorn, with weight and epsilon sweeps) and grid strategies (full image, fixed grid, progressive grid) across four brush materials and several stroke budgets; (3) a hardware benchmark of quality versus time versus memory on an RTX 3070, an Apple M4 and a desktop CPU; and (4) an interactive Gradio web system with live progress and time-lapse export. Together these turn a research prototype into a measured, reproducible and usable system, and the ablation and benchmark tables are the results a later paper would report.
