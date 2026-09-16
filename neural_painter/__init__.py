"""neural_painter: photo-to-painting with a differentiable neural renderer.

A modular, cross-platform (CUDA / Apple MPS / CPU) re-implementation of
Stylized Neural Painting (Zou et al., CVPR 2021) built on PyTorch 2.x.

Package map (RESEARCH_PLAN.md, section 2.4):

- ``core``      stroke models, procedural rasterizer, canvas compositing, grid utilities, image I/O
- ``models``    neural renderer (FusionNet) and VGG feature extractor
- ``losses``    pixel, Sinkhorn optimal-transport and style losses
- ``pipeline``  renderer training, stroke sampling, stroke optimization, progressive painting
- ``export``    video / GIF time-lapse and SVG export
- ``app``       Gradio web interface
"""

__version__ = "0.1.0"
