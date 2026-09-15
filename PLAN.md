# Neural Painting System: Recreation & Modernization Plan

This document outlines the complete architectural design, mathematical foundations, system modules, and step-by-step implementation plan to recreate and modernize **Stylized Neural Painting** from scratch.

---

## 1. System Overview & Core Concept

### 1.1 The Core Problem
Traditional image-to-painting or artistic filtering methods operate at the pixel level (predicting RGB values for each pixel). While visually interesting, they do not simulate the physical creation process of human artists who work with vector strokes, variable brush pressure, paint blending, and progressive layers.

However, standard vector rasterizers (OpenCV, Cairo, Skia) are **non-differentiable**: gradients cannot flow back from pixel errors to vector stroke parameters (such as center coordinates, rotation angle, brush width/height, control points, and color gradients).

### 1.2 The Solution: Neural Differentiable Rendering
To make vector painting optimizable via gradient descent:
1. We train a **Neural Renderer ($G_\phi$)** as a differentiable surrogate for the physical brush.
2. Given stroke parameters $\theta$, $G_\phi(\theta)$ produces an RGB foreground patch $\hat{F}$ and an opacity mask $\hat{A}$.
3. We optimize stroke parameters $\theta$ directly against a target image $I_{\text{target}}$ using standard backpropagation and gradient descent.

```
+-------------------------------------------------------------------------+
|                        OPTIMIZATION LOOP                                |
|                                                                         |
|  Target Image (I_target)                                                |
|         |                                                               |
|         v                                                               |
|   [Error Map] ---> [Stroke Sampler] ---> Initial Stroke Params (θ)      |
|                           ^                       |                     |
|                           | (Backprop Gradients)  v                     |
|                 [Loss: L_pixel + L_sinkhorn] <--- [Neural Renderer G_φ] |
|                                                           |             |
|                                                           v             |
|  Canvas (C_k) ----------------------------------> [Alpha Blend C_k+1]   |
+-------------------------------------------------------------------------+
```

---

## 2. Mathematical Foundations & Parameter Formulations

### 2.1 Stroke Formulations

| Brush Style | Parameter Vector ($\theta$) | Dimensions | Description |
| :--- | :--- | :---: | :--- |
| **Oil Paint Brush** | $[x_c, y_c, w, h, \theta, R_0, G_0, B_0, R_1, G_1, B_1, \alpha]$ | 12 | Elliptical patch with texture mask, color gradient, rotation |
| **Watercolor / Ink** | $[x_0, y_0, x_1, y_1, x_2, y_2, r_0, r_2, R_0, G_0, B_0, R_2, G_2, B_2, \alpha]$ | 15 | Quadratic Bézier spline with variable endpoint radii and color gradient |
| **Marker Pen** | $[x_0, y_0, x_1, y_1, x_2, y_2, r_0, r_2, R, G, B, \alpha]$ | 12 | Quadratic Bézier spline with solid color and variable line width |
| **Tape / 8-Bit** | $[x_c, y_c, w, h, \theta, R, G, B, \alpha]$ | 9 | Rectangular / square patch with uniform color |

### 2.2 Differentiable Canvas Composition
When placing stroke $k$ with predicted foreground $\hat{F}_k$ and alpha mask $\hat{A}_k$ onto canvas $C_{k-1}$:
$$C_k = \hat{A}_k \odot \hat{F}_k + (1 - \hat{A}_k) \odot C_{k-1}$$
where $\odot$ denotes element-wise Hadamard product.

### 2.3 Loss Functions
1. **Pixel-Level Reconstruction Loss (Charbonnier / Smooth $L_1$):**
   $$\mathcal{L}_{\text{pixel}} = \sqrt{\|C_k - I_{\text{target}}\|^2 + \epsilon^2}$$
2. **Sinkhorn Optimal Transport Loss (Wasserstein Distance):**
   When strokes do not overlap with the target region, standard pixel loss gradients vanish. Sinkhorn distance computes mass transportation cost between distributions:
   $$\mathcal{L}_{\text{sinkhorn}} = \langle P^*, M \rangle$$
   providing smooth, continuous gradients guiding strokes toward the target position regardless of spatial distance.
3. **Perceptual / Style Loss (Gram Matrix):**
   $$\mathcal{L}_{\text{style}} = \sum_{l} \frac{1}{4 N_l^2 M_l^2} \|G(\phi_l(C_k)) - G(\phi_l(I_{\text{style}}))\|_F^2$$

---

## 3. Project Directory Structure

```
neural_painter/
├── configs/                       # Experiment and brush configs (YAML)
│   ├── oil_brush.yaml
│   ├── watercolor.yaml
│   └── marker_pen.yaml
├── core/                          # Fundamental primitives & vector math
│   ├── __init__.py
│   ├── stroke_models.py           # Parametric stroke dataclasses
│   ├── procedural_rasterizer.py   # Ground-truth CPU OpenCV rasterizer
│   ├── morphology.py              # Texture masking & dilation
│   └── differentiable_canvas.py   # Alpha compositing & canvas state
├── models/                        # Neural network architectures
│   ├── __init__.py
│   ├── neural_renderer.py         # Differentiable stroke generator (FusionNet)
│   └── vgg_perceptual.py          # Pretrained feature extractor
├── losses/                        # Loss functions
│   ├── __init__.py
│   ├── pixel_loss.py              # L1 / Charbonnier loss
│   ├── sinkhorn.py                # Differentiable Sinkhorn distance
│   └── style_loss.py              # Gram matrix style transfer loss
├── pipeline/                      # Training & Inference pipelines
│   ├── __init__.py
│   ├── train_renderer.py          # Synthetic renderer training script
│   ├── stroke_sampler.py          # Error-map guided initialization
│   ├── painter_engine.py          # Stroke parameter optimizer
│   └── progressive_painter.py     # Multi-scale coarse-to-fine painting
├── export/                        # Export utilities
│   ├── __init__.py
│   ├── svg_exporter.py            # SVG vector path generator
│   └── video_recorder.py          # MP4 / GIF timelapse generator
├── app/                           # Web application UI
│   └── gradio_app.py              # Interactive painting demo
├── tests/                         # Unit tests
│   ├── test_rasterizer.py
│   ├── test_renderer.py
│   └── test_optimizer.py
├── scripts/
│   ├── train_all_brushes.sh
│   └── run_paint.py
├── pyproject.toml
└── README.md
```

---

## 4. Phased Implementation Roadmap

### Phase 1: Core Mathematical Primitives & Procedural Rasterizer
- [ ] Setup virtual environment and modern dependencies (`torch>=2.0`, `torchvision`, `opencv-python`, `numpy`, `scipy`, `einops`, `gradio`).
- [ ] Implement `core/stroke_models.py` with dataclasses and bounds validation for all brush types.
- [ ] Implement `core/procedural_rasterizer.py`:
  - Quadratic Bézier curve rasterization using OpenCV.
  - Brush texture loading, alpha blending, and color gradient interpolation.
  - Synthetic parameter random sampling ($\theta \sim \mathcal{U}(0, 1)^d$).
- [ ] Implement `tests/test_rasterizer.py` to verify procedural stroke generation.

### Phase 2: Neural Renderer Architecture & Training Pipeline
- [ ] Implement `models/neural_renderer.py`:
  - **Shape Decoder**: Multi-layer perceptron (MLP) + ConvTranspose2d / PixelShuffle to predict alpha mask $\hat{A} \in [0, 1]^{H \times W}$.
  - **Color/Texture Decoder**: Coordinate + color projection to predict RGB foreground $\hat{F} \in [0, 1]^{3 \times H \times W}$.
  - **Fusion Module**: Produces final stroke representation $\hat{S} = \hat{A} \odot \hat{F}$.
- [ ] Implement `pipeline/train_renderer.py`:
  - On-the-fly ground truth dataset generation (no disk storage overhead).
  - Train generator with $L_1$ loss + SSIM / Cosine loss.
  - Support Apple Silicon (`mps`), NVIDIA (`cuda`), and `cpu`.
  - Validate model convergence ($> 0.98$ PSNR vs ground truth).

### Phase 3: Optimization Engine & Progressive Painter
- [ ] Implement `losses/sinkhorn.py` (GPU-accelerated entropy-regularized optimal transport).
- [ ] Implement `losses/pixel_loss.py` and `losses/style_loss.py`.
- [ ] Implement `pipeline/stroke_sampler.py`:
  - Error map calculation: $E = \|I_{\text{target}} - C_k\|_1$.
  - Probability density sampling for stroke placement.
- [ ] Implement `pipeline/painter_engine.py`:
  - Single-block gradient descent on stroke parameters with Adam optimizer.
- [ ] Implement `pipeline/progressive_painter.py`:
  - Coarse-to-fine multi-scale rendering loop ($1\times1 \to 2\times2 \to 4\times4 \to \dots \to M\times M$ grid subdivision).
  - Dynamic stroke count and learning rate scheduling.

### Phase 4: Artistic Extensions, Style Transfer & Exporters
- [ ] Implement Neural Style Transfer on vector stroke parameters (transfer color only vs transfer color + texture).
- [ ] Implement `export/svg_exporter.py`: Convert optimized Bézier strokes directly into standard SVG vector files.
- [ ] Implement `export/video_recorder.py`: Render high-definition video timelapses (MP4 / GIF) of the progressive painting process.

### Phase 5: Interactive Web UI & Production Packaging
- [ ] Implement `app/gradio_app.py`:
  - Drag-and-drop image upload.
  - Brush type selection (Oil, Watercolor, Marker, 8-bit).
  - Live preview slider showing painting progress stroke by stroke.
  - Download buttons for Final Image, Video Timelapse, and SVG Vector file.
- [ ] Package codebase with clean CLI entrypoints (`run_paint.py`).

---

## 5. Modern Improvements & Key Differentiators

| Aspect | Original 2021 Implementation | Modernized Recreation |
| :--- | :--- | :--- |
| **Framework** | Legacy PyTorch 1.7 | Modern PyTorch 2.x with `torch.compile` support |
| **Hardware** | Optimized mainly for CUDA | Cross-platform: Apple Silicon (`mps`), CUDA, and CPU |
| **Export Formats** | PNG/JPG raster images | Full **SVG Vector export** (usable with pen plotters & Illustrator) |
| **Stylization** | Basic VGG Gram matrix | Dual-mode: VGG-19 Gram matrix + modern perceptual features |
| **Interface** | CLI argument parsing | Interactive **Gradio Web UI** with real-time brush playback |
| **Architecture** | Monolithic scripts | Modular, typed, tested Python package with YAML configs |

---

## 6. Getting Started

To initialize the project environment and run the test suite:

```bash
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install development dependencies
pip install torch torchvision numpy scipy opencv-python matplotlib gradio pyyaml pytest

# 3. Start development in core/
```
