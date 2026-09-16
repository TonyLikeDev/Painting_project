# Notes on the original `stylized-neural-painting` code (Week 2)

One page per file, written after reading the 2021 reference implementation (commit `e7a66b2`, patched as in `experiments/original_repo_modern_torch.patch`). Each note has the same five parts: purpose, key classes and functions, data flow, quirks and technical debt, and what changes in the new `neural_painter` package.

| Note | Covers | One line |
| :--- | :--- | :--- |
| [renderer.md](renderer.md) | `renderer.py` | OpenCV ground-truth rasterizer, parameter layouts of the four brushes, error-map stroke sampler. |
| [networks.md](networks.md) | `networks.py` | The `ZouFCNFusion` neural renderer (shape decoder + colour decoder), its light 32 px variant, and unused ablation generators. |
| [loss.md](loss.md) | `loss.py` | Pixel loss, Sinkhorn OT loss wrapper, VGG perceptual and Gram style losses. |
| [pytorch_batch_sinkhorn.md](pytorch_batch_sinkhorn.md) | `pytorch_batch_sinkhorn.py` | Batched log-domain Sinkhorn algorithm and its numerical shortcuts. |
| [painter.md](painter.md) | `painter.py`, `demo.py`, `demo_prog.py` | Stroke search engine: initialisation, differentiable forward pass, progressive grid loop, final rendering. |
| [utils.md](utils.md) | `utils.py`, `morphology.py` | Synthetic stroke dataset, grid split and stitch, metrics, oil-brush affine warp, differentiable 3x3 erosion and dilation. |
