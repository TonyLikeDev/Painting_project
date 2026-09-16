# Chapter 2. Theoretical Background

This chapter collects the theory that the system in this project is built on. It follows the reference implementation of Stylized Neural Painting (Zou et al. 2021) closely, because the first goal of the project is to re-implement that method faithfully and the second goal is to measure what each of its components contributes. Where the text says "in the reference code", it refers to the files `renderer.py`, `networks.py`, `loss.py`, `pytorch_batch_sinkhorn.py`, `painter.py` and `demo_prog.py` of the original repository, which were read line by line while writing this chapter. All statements about scaling factors, kernel sizes, iteration counts and layer shapes come from that code.

## 2.1 Stroke-based rendering

### 2.1.1 Pixel space versus stroke space

Most image-to-image methods, from classical filters to modern generative networks, produce their output by predicting a colour for every pixel. This is the natural representation for a display, but it is not the representation that an artist works in. A painter builds an image out of a small number of *strokes*, each carrying a shape, a colour, a thickness and a degree of opacity, and the order in which the strokes are laid down matters because later strokes cover earlier ones. The result is a *vector* representation: a list of stroke parameter vectors $\{\boldsymbol{\theta}_1, \dots, \boldsymbol{\theta}_N\}$ that can be re-rendered at any resolution, edited one stroke at a time, played back as an animation, or converted to other vector formats.

Stroke-based rendering (SBR) is the family of techniques that turn a source image into such a list of strokes. The question every SBR method must answer is the same: *given a target image $I$, which strokes, in which order, reproduce $I$ well while still looking like a painting?* Formally, if $\mathcal{R}$ is a renderer that turns a sequence of strokes into a raster image, SBR looks for

$$
\boldsymbol{\Theta}^\star = \arg\min_{\boldsymbol{\Theta} = (\boldsymbol{\theta}_1,\dots,\boldsymbol{\theta}_N)} \; \mathcal{L}\big(\mathcal{R}(\boldsymbol{\Theta}),\, I\big),
$$

where $\mathcal{L}$ measures how far the rendered canvas is from the target. The difficulty is that $\mathcal{R}$ is normally a rasterizer, and rasterizers are not differentiable with respect to $\boldsymbol{\theta}$ (Section 2.3). Different lines of work have dealt with this in different ways.

### 2.1.2 A short history

**Greedy and heuristic methods.** Haeberli (1990) introduced the idea of an image as an ordered list of brush strokes and let a user place them interactively, sampling colour from the source photograph. Litwinowicz (1997) automated the placement for video, aligning short strokes with image gradients so that the strokes follow edges. Hertzmann (1998) proposed the coarse-to-fine scheme that most later work inherits: the image is painted in layers with decreasing brush radius, each layer is compared against a version of the reference blurred at that radius, and new strokes are placed only where the error between the current canvas and the blurred reference is large. Strokes are curved splines that follow the gradient field. None of these methods optimizes the strokes against a loss; they place strokes by rules and never revisit them.

**Reinforcement learning.** Learning to Paint (Huang et al. 2019) frames painting as a sequential decision problem. An agent observes the canvas and the target and emits a small batch of quadratic Bezier strokes per step; a neural renderer draws them, and the reward is the decrease of a learned (WGAN-based) distance to the target. The neural renderer is trained beforehand to imitate a procedural rasterizer, which is the same idea that this project uses, but the stroke parameters themselves are predicted by a policy network rather than optimized directly.

**Feed-forward prediction.** Paint Transformer (Liu et al. 2021) removes the sequential agent and predicts a whole set of strokes at once with a transformer, trained on synthetic stroke images so that no painting dataset is needed. It is fast, but the strokes are only as good as the single forward pass, and the method is tied to the stroke types it was trained on.

**Neural renderer plus direct optimization.** Neural Painters (Nakano 2019) trained a generative network to imitate the MyPaint brush engine and then optimized stroke parameters by gradient descent through that network, which also made "intrinsic style transfer" possible by adding a perceptual loss. Stylized Neural Painting (Zou et al. 2021) is the method this project reproduces. It trains a dedicated neural renderer per brush material, optimizes strokes directly against the target with a pixel loss and an optimal-transport loss, uses a progressive grid so that the number of strokes can be large, and supports oil paint, watercolour, marker pen and coloured tape. Because the strokes are optimized rather than predicted, the same pipeline works for any brush for which a renderer can be trained, and it can be combined with neural style transfer by changing the loss.

### 2.1.3 Why strokes

Three properties motivate the stroke representation in this project. First, *editability and resolution independence*: the output is a list of a few hundred parameter vectors, and the canvas can be rasterized at any size. Second, *process*: the ordered list is exactly what is needed to show the painting being made, stroke by stroke, which is one of the required features of the system. Third, *style control*: changing the brush material changes the renderer, not the optimization, so the same photograph can be painted in four materials with one code path.

## 2.2 Stroke parameterization

Every stroke is a vector $\boldsymbol{\theta} \in [0,1]^d$. Keeping all parameters in the unit interval is what makes the later optimization simple: the same clamp and the same learning rate apply to every coordinate, and the neural renderer is trained on uniform samples from the same cube. Each vector is split into three groups, *shape* ($d_s$ entries), *colour* ($d_c$ entries) and *opacity* ($d_a = 1$ entry), and the renderer treats the groups differently (Section 2.4). The reference code uses a square canvas of side $W$ pixels and maps a normalized coordinate to a pixel index with

$$
\operatorname{px}(x) = \big\lfloor x\,(W-1) + 0.5 \big\rfloor .
$$

Table 2.1 lists the four brushes. In the text below the rotation angle is written $\varphi$ to keep it apart from the parameter vector $\boldsymbol{\theta}$; in the code it is called `theta`.

**Table 2.1.** Stroke parameter vectors for the four brush materials.

| Brush | Name in code | Parameter vector $\boldsymbol{\theta}$ | $d$ | $d_s / d_c / d_a$ |
| :--- | :--- | :--- | :---: | :---: |
| Oil paint | `oilpaintbrush` | $(x_c, y_c, w, h, \varphi, R_0, G_0, B_0, R_2, G_2, B_2, A)$ | 12 | 5 / 6 / 1 |
| Coloured tape | `rectangle` | $(x_c, y_c, w, h, \varphi, R, G, B, A)$ | 9 | 5 / 3 / 1 |
| Watercolour | `watercolor` | $(x_0, y_0, x_1, y_1, x_2, y_2, r_0, r_2, R_0, G_0, B_0, R_2, G_2, B_2, A)$ | 15 | 8 / 6 / 1 |
| Marker pen | `markerpen` | $(x_0, y_0, x_1, y_1, x_2, y_2, r_0, r_2, R, G, B, A)$ | 12 | 8 / 3 / 1 |

### 2.2.1 Oil paint brush

The oil brush is a textured, rotated, scaled patch. The centre is $(\operatorname{px}(x_c), \operatorname{px}(y_c))$, the patch size in pixels is

$$
w_{px} = \lfloor 1 + w\,W \rfloor, \qquad h_{px} = \lfloor 1 + h\,W \rfloor,
$$

so a single stroke can cover the whole canvas, and the rotation is $\varphi = \pi\,\theta_{\varphi}$ with $\theta_\varphi \in [0,1]$, which covers every orientation because the brush texture is symmetric under a half turn in appearance. The texture comes from one of four greyscale PNG images shipped with the reference code: a *small* and a *large* brush, each in a *vertical* and a *horizontal* version. The large texture is chosen when the patch is big, $w_{px} h_{px} / W^2 > 0.1$, and the vertical version is chosen when $h_{px} > w_{px}$. The chosen texture $T$ (with pixel size $b_w \times b_h$) is coloured with a gradient along its length: row $i$ of the texture receives the colour

$$
c(i) = (1-t)\,c_0 + t\,c_2, \qquad t = i / b_h,
$$

where $c_0 = (R_0, G_0, B_0)$ and $c_2 = (R_2, G_2, B_2)$, and the foreground is the product of texture intensity and colour. The opacity mask is binary, $\mathbb{1}[T > 0]$. The coloured texture and its mask are then placed on the canvas with one affine transform,

$$
M = T(x_c, y_c)\; R(\varphi)\; S\!\left(\tfrac{w_{px}}{b_w}, \tfrac{h_{px}}{b_h}\right)\; T\!\left(-\tfrac{b_w}{2}, -\tfrac{b_h}{2}\right),
$$

applied with area interpolation. The opacity parameter $A$ is present in the vector for uniformity but is *not used* by the oil brush: oil paint is treated as opaque, and, as Section 2.4 shows, the neural renderer for this brush also forces $A = 1$.

### 2.2.2 Coloured tape (rectangle)

The tape stroke is a solid rotated rectangle with a single colour. The centre is mapped with $\operatorname{px}$, and the half-extents are

$$
w_{px} = 1 + \lfloor w\,W / 4 \rfloor, \qquad h_{px} = 1 + \lfloor h\,W / 4 \rfloor,
$$

so the longest side of a tape stroke is at most about half of the canvas. The four corners $(x_c \pm w_{px}, y_c \pm h_{px})$ are rotated about the centre by $\varphi = \pi\,\theta_\varphi$ and the polygon is filled with anti-aliasing. The foreground is the colour $(R, G, B)$ inside the polygon and the opacity map takes the value $A$ inside the polygon and $0$ outside.

### 2.2.3 Watercolour and marker pen (quadratic Bezier strokes)

Both curved brushes share the same eight shape parameters. Two of the three control points of a quadratic Bezier curve are the endpoints $P_0 = (x_0, y_0)$ and $P_2 = (x_2, y_2)$. The middle control point is not stored in absolute coordinates; it is stored *relative to the bounding box of the endpoints*,

$$
P_1 = \big(x_0 + (x_2 - x_0)\,x_1,\; y_0 + (y_2 - y_0)\,y_1\big),
$$

with $x_1, y_1 \in [0,1]$. This keeps the curve inside the box spanned by its endpoints, bounds the curvature, and, as Section 2.8 will show, makes the relative control point invariant to the change of coordinates between grid cells. The curve is

$$
B(t) = (1-t)^2 P_0 + 2t(1-t)\,P_1 + t^2 P_2, \qquad t \in [0, 1],
$$

and the rasterizer samples it at $t_i = i/100$, $i = 0, \dots, 99$. The two radii are mapped to pixels as $r_{px} = 1 + \lfloor r\,W/4 \rfloor$.

*Watercolour* draws, at every sample $B(t_i)$, a filled disc whose radius and colour are interpolated along the curve,

$$
r(t) = (1-t)\,r_0 + t\,r_2, \qquad c(t) = (1-t)\,c_0 + t\,c_2,
$$

so a watercolour stroke can taper and change hue from one end to the other. The opacity map gets the constant value $A$ inside every disc.

*Marker pen* draws, at every sample, a *square* of half-side $r_0$ that is rotated to align with the tangent of the curve. With the derivative $B'(t) = 2(t-1)P_0 + 2(1-2t)P_1 + 2tP_2 = (\dot{x}, \dot{y})$, the rotation angle used by the code is $\psi = \operatorname{atan2}(\dot{x}, \dot{y}) - \pi/2$. The stroke has a single colour $(R, G, B)$ and uses only the first radius; $r_2$ is parsed but ignored, which is why the marker vector has $d_c = 3$ and still $d_s = 8$. A marker stroke whose endpoints are closer than four pixels in Manhattan distance, $|x_0 - x_2| + |y_0 - y_2| < 4$, is not drawn at all, because a chisel tip needs a direction.

### 2.2.4 Validity and sampling

Random strokes for renderer training are drawn uniformly, $\boldsymbol{\theta} \sim \mathcal{U}(0,1)^d$. At painting time the reference code additionally discards strokes that are too small to matter before rasterizing the final canvas: a stroke is kept only if its largest size parameter ($\max(w, h)$ or $\max(r_0, r_2)$) exceeds $0.025$.

## 2.3 Procedural rasterization and alpha compositing

### 2.3.1 Foreground, opacity and the compositing equation

Each brush's rasterizer produces two images of the canvas size: a foreground $F \in [0,1]^{3 \times W \times W}$ that holds the stroke's colour wherever the stroke covers the canvas (and black elsewhere), and an opacity map $\hat{A} \in [0,1]^{3 \times W \times W}$, stored with three identical channels for convenience, that is $A$ inside the stroke, $0$ outside, and takes intermediate values on the anti-aliased boundary. A stroke is laid onto the current canvas $C_{k-1}$ with the standard "over" operator,

$$
C_k = \hat{A}_k \odot F_k + (1 - \hat{A}_k) \odot C_{k-1},
\tag{2.1}
$$

where $\odot$ is the element-wise product. Equation (2.1) is linear in $F_k$, in $\hat{A}_k$ and in $C_{k-1}$, so it is differentiable with respect to all three, and applying it $N$ times in sequence gives the final canvas $C_N = \mathcal{R}(\boldsymbol{\theta}_1, \dots, \boldsymbol{\theta}_N)$ as a differentiable function of the *images* $F_k, \hat{A}_k$. The canvas starts as $C_0 = \mathbf{0}$ (black) or $C_0 = \mathbf{1}$ (white). Because the operator is applied in order, stroke order is part of the representation: the last stroke is always fully visible where it is opaque.

### 2.3.2 The dilate / erode trick

At inference time (but not while generating renderer training data) the reference rasterizer dilates the foreground with a $2 \times 2$ kernel and erodes the opacity map with a $2 \times 2$ kernel before compositing. The reason is the anti-aliased boundary. On a boundary pixel the foreground image is a blend of the stroke colour and black, and the opacity is fractional; compositing such a pixel darkens the stroke edge and leaves a thin dark halo. Eroding the mask by one pixel and dilating the foreground by one pixel means that every pixel where the mask is non-zero now reads a fully coloured foreground value, and the halo disappears. The painting engine applies the same idea to the neural renderer's outputs with $3 \times 3$ max-pooling (dilation) on the predicted foreground and $3 \times 3$ min-pooling (erosion) on the predicted opacity, implemented with `unfold` so that gradients still flow (max and min are piecewise differentiable).

### 2.3.3 Why the rasterizer is not differentiable

Everything in Section 2.2 that happens *after* the parameters are turned into pixels breaks the gradient chain from $C_N$ back to $\boldsymbol{\theta}$:

1. **Integer rounding.** $\operatorname{px}(\cdot)$, the pixel sizes $w_{px}, h_{px}, r_{px}$ and the sampled curve points are cast to integers. As functions of $\boldsymbol{\theta}$ they are piecewise constant, so their derivative is zero almost everywhere and undefined at the jumps.
2. **Scan conversion.** `cv2.fillPoly` and `cv2.circle` decide, pixel by pixel, whether the pixel centre lies inside a polygon or a disc. The output is a coverage decision, and there is no expression for "how much the coverage of pixel $(i, j)$ changes when $x_0$ moves by $\delta$" inside the library. Anti-aliasing softens the edge but does not expose a derivative.
3. **Texture warping and branching.** The oil brush selects one of four textures by a threshold on area and aspect ratio, which is a discontinuous choice, and then calls `cv2.warpAffine`, which resamples the texture without providing gradients with respect to the affine matrix.
4. **Validity gates.** The minimum-size check and the four-pixel marker rule are hard thresholds.

Some of these steps could be replaced by differentiable operations (a soft rasterizer, `grid_sample` for the warp), and that is one route taken in the differentiable rendering literature. The route taken by Zou et al. (2021), and by this project, is different: keep the procedural rasterizer exactly as it is, because it defines the ground truth appearance of each brush, and learn a neural network that imitates it.

## 2.4 The neural renderer as a differentiable surrogate

### 2.4.1 Learning objective

The neural renderer $G_\phi$ is a network with parameters $\phi$ that maps a single stroke vector $\boldsymbol{\theta} \in [0,1]^d$ to a predicted foreground $\hat{F} = G^F_\phi(\boldsymbol{\theta})$ and a predicted opacity map $\hat{A} = G^A_\phi(\boldsymbol{\theta})$ of size $3 \times S \times S$, where $S = 128$ for the full renderers and $S = 32$ for the lightweight renderers. It is trained by supervised regression against the procedural rasterizer. Training data are generated on the fly: for every sample a vector is drawn uniformly, $\boldsymbol{\theta} \sim \mathcal{U}(0,1)^d$, rasterized at resolution $S$ *without* the dilate/erode step, and the pair $(F, \hat{A}^{\text{gt}})$ is the target. The reference code uses 50,000 such samples per epoch for training and 2,500 for validation, a batch size of 64, Adam with learning rate $2 \times 10^{-4}$ and $\beta = (0.9, 0.999)$, a step schedule that divides the learning rate by 10 every 100 epochs, and up to 400 epochs. The loss is the squared pixel error on both outputs,

$$
\mathcal{L}_{\text{render}}(\phi) = \mathbb{E}_{\boldsymbol{\theta}}\Big[\, \big\| G^F_\phi(\boldsymbol{\theta}) - F(\boldsymbol{\theta}) \big\|_2^2 + \big\| G^A_\phi(\boldsymbol{\theta}) - \hat{A}^{\text{gt}}(\boldsymbol{\theta}) \big\|_2^2 \,\Big],
\tag{2.2}
$$

with the mean taken over pixels and channels, and validation fidelity is reported as PSNR between predicted and rasterized images. No dataset of paintings is involved at any point; the renderer learns the brush, not the art.

### 2.4.2 The FusionNet architecture

Zou et al. (2021) observed that a single decoder does two jobs of different character: the *shape* of a stroke is a sharp binary region that depends only on the shape parameters, while the *colour* is a smooth field that depends on the colour parameters (and, through the gradient, on shape as well). Their "fusion" renderer therefore has two branches (Figure 2.1, drawn in the architecture diagram of Week 2):

- a **shape decoder** (class `PixelShuffleNet`, called `huangnet` in the code after Huang et al. 2019, whose renderer it follows) that receives *only* the shape group $\boldsymbol{\theta}_s \in [0,1]^{d_s}$ and outputs a mask $m \in \mathbb{R}^{3 \times S \times S}$;
- a **colour decoder** (class `DCGAN`) that receives the *whole* vector $\boldsymbol{\theta}$ as a $d \times 1 \times 1$ code and outputs a colour field $c \in \mathbb{R}^{3 \times S \times S}$ (the network actually has six output channels; only the first three are used);
- a **fusion step**

$$
\hat{F} = c \odot m, \qquad \hat{A} = A \cdot m,
\tag{2.3}
$$

where $A$ is the scalar opacity parameter broadcast over the image. For the oil brush $A$ is replaced by the constant $1$ inside the network, matching the opaque procedural brush of Section 2.2.1.

The shape decoder is an MLP that expands the few shape parameters into a coarse feature map, followed by convolutions and pixel-shuffle upsampling. The colour decoder is a DCGAN-style stack of transposed convolutions with batch normalization. Table 2.2 gives the layer shapes of both branches for the full ($S = 128$) and light ($S = 32$) variants; $d_s$ and $d$ are the brush-specific dimensions from Table 2.1. Every convolution is $3 \times 3$ with stride 1 and padding 1; every transposed convolution has kernel 4, and all but the first have stride 2 and padding 1 (the first has stride 1 and no padding, turning the $1 \times 1$ code into a $4 \times 4$ map). Neither branch has an output non-linearity; the outputs are raw regression values, which the loss (2.2) pushes into $[0, 1]$.

**Table 2.2.** Layer sizes of the fusion renderer. Output shapes are channels × height × width.

| Branch | Full renderer (`zou-fusion-net`, $S=128$) | Light renderer (`zou-fusion-net-light`, $S=32$) |
| :--- | :--- | :--- |
| Shape decoder, MLP | Linear $d_s \to 512$, ReLU; $512 \to 1024$, ReLU; $1024 \to 2048$, ReLU; $2048 \to 4096$, ReLU; reshape to $16 \times 16 \times 16$ | Linear $d_s \to 512$, ReLU; $512 \to 1024$, ReLU; $1024 \to 2048$, ReLU; reshape to $8 \times 16 \times 16$ |
| Shape decoder, conv | Conv $16 \to 32$, ReLU ($32 \times 16 \times 16$); Conv $32 \to 32$, PixelShuffle(2) ($8 \times 32 \times 32$); Conv $8 \to 16$, ReLU; Conv $16 \to 16$, PixelShuffle(2) ($4 \times 64 \times 64$); Conv $4 \to 8$, ReLU; Conv $8 \to 12$, PixelShuffle(2) ($3 \times 128 \times 128$) | Conv $8 \to 64$, ReLU ($64 \times 16 \times 16$); Conv $64 \to 12$, PixelShuffle(2) ($3 \times 32 \times 32$) |
| Colour decoder | ConvT $d \to 512$ ($512 \times 4 \times 4$); ConvT $512 \to 512$ ($8 \times 8$); ConvT $512 \to 256$ ($16 \times 16$); ConvT $256 \to 128$ ($32 \times 32$); ConvT $128 \to 64$ ($64 \times 64$); ConvT $64 \to 6$ ($6 \times 128 \times 128$); BatchNorm + ReLU after every layer except the last | ConvT $d \to 512$ ($512 \times 4 \times 4$); ConvT $512 \to 256$ ($8 \times 8$); ConvT $256 \to 128$ ($16 \times 16$); ConvT $128 \to 6$ ($6 \times 32 \times 32$); BatchNorm + ReLU after every layer except the last |
| Fusion | $\hat{F} = c_{0:3} \odot m$, $\hat{A} = A\, m$ | same |
| Parameters (oil brush, $d = 12$, $d_s = 5$), counted from the layer definitions | shape decoder $\approx 11.04$ M, colour decoder $\approx 7.05$ M, total $\approx 18.1$ M | shape decoder $\approx 2.64$ M, colour decoder $\approx 2.73$ M, total $\approx 5.4$ M |

Weights are initialized from $\mathcal{N}(0, 0.02^2)$ (the pix2pix convention) and batch-normalization layers run in evaluation mode at painting time, so the renderer is a deterministic function of $\boldsymbol{\theta}$ during optimization. The light renderer trades resolution for speed and memory: a $32 \times 32$ stroke patch is enough when the canvas is split into a fine grid (Section 2.8), and the reference authors report that it is roughly three times faster.

### 2.4.3 The gradient path from pixels to parameters

With the neural renderer in place the whole painting pipeline is one differentiable computation graph. Reading it backwards from the loss:

1. The loss (Sections 2.5 to 2.7) is a differentiable function of the final canvas $C_N$.
2. $C_N$ is obtained from $C_0$ by $N$ applications of (2.1). Each application is bilinear in $(\hat{A}_k, F_k)$ and linear in $C_{k-1}$, so $\partial C_N / \partial \hat{F}_k$ and $\partial C_N / \partial \hat{A}_k$ exist for every $k$; a stroke that is later covered by opaque strokes simply receives a zero gradient in the covered pixels, which is the correct behaviour.
3. $\hat{F}_k = c(\boldsymbol{\theta}_k) \odot m(\boldsymbol{\theta}_{k,s})$ and $\hat{A}_k = A_k\, m(\boldsymbol{\theta}_{k,s})$, so gradients reach the colour branch output, the mask, and the opacity scalar directly through products.
4. The colour branch is a composition of transposed convolutions, batch-norm affine maps and ReLUs; the shape branch is a composition of linear layers, ReLUs, convolutions and pixel-shuffles (a fixed permutation). All are differentiable almost everywhere, so PyTorch autograd produces $\partial \mathcal{L} / \partial \boldsymbol{\theta}_{k}$ for the shape, colour and opacity groups of every stroke.

The renderer's own weights $\phi$ are frozen during painting; only the stroke vectors are optimized. What the gradient means is worth stating in words: $\partial \mathcal{L} / \partial x_c$ tells the optimizer in which direction to move a stroke so that the *rendered image* becomes closer to the target, even though no rasterizer was ever differentiated. The surrogate is only as faithful as its training, which is why renderer fidelity (PSNR against the procedural rasterizer on held-out strokes) is one of the quantities this project measures.

## 2.5 Pixel loss

The basic reconstruction loss compares the canvas with the target pixel by pixel. The reference painting engine uses the mean absolute error,

$$
\mathcal{L}_{\text{pix}}(C, I) = \frac{1}{3 H W} \sum_{c, i, j} \big| C_{c,i,j} - I_{c,i,j} \big|,
\tag{2.4}
$$

(class `PixelLoss` with $p = 1$; the renderer training uses the same class with $p = 2$). An option `ignore_color` averages the three channels before comparing, which turns (2.4) into a luminance-only loss; it is used during style transfer so that the pixel term constrains structure but not colour. In the new package the smooth variant

$$
\mathcal{L}_{\text{Char}}(C, I) = \frac{1}{3HW}\sum_{c,i,j} \sqrt{\big(C_{c,i,j} - I_{c,i,j}\big)^2 + \epsilon^2}
\tag{2.5}
$$

(Charbonnier loss, a smooth $L_1$) will be offered as well: it has the robustness of $L_1$ to outliers and a well-defined gradient at zero error, which avoids the sign flips of $|\cdot|$ near convergence. With $\epsilon \to 0$ it reduces to (2.4).

Pixel losses are tied to the fidelity metric used throughout the experiments. The peak signal-to-noise ratio of a canvas with values in $[0,1]$ is

$$
\operatorname{PSNR}(C, I) = 10 \log_{10} \frac{1}{\operatorname{MSE}(C, I)}, \qquad \operatorname{MSE} = \frac{1}{3HW}\sum_{c,i,j}\big(C_{c,i,j} - I_{c,i,j}\big)^2,
$$

so minimizing the squared error maximizes PSNR directly, and minimizing $L_1$ does so approximately. The reference code prints PSNR at every optimization step under the name `step_psnr` / `step_acc`.

## 2.6 Optimal transport and the Sinkhorn loss

### 2.6.1 Why a pixel loss is not enough

Consider a single stroke rendered at a position that does not overlap the region of the target it should eventually cover. Under (2.4) the error is the sum of two disjoint contributions: the stroke's pixels are wrong (they should be background) and the target region's pixels are wrong (they should be painted). Moving the stroke slightly does not change either contribution as long as the supports stay disjoint, so the loss is locally constant in the stroke's position and $\partial \mathcal{L}_{\text{pix}} / \partial x_c = 0$. The only thing the pixel gradient can do with such a stroke is shrink it or fade it to background colour, which is exactly the wrong response. Gradients appear only once the stroke and the target region overlap, and then they act only on the overlapping boundary. This is the *vanishing gradient* problem of pixel losses in stroke optimization, and it is the reason Zou et al. (2021) add a loss that measures *how far mass has to move*, not *how much mass is in the wrong place*.

### 2.6.2 Monge, Kantorovich and the Wasserstein distance

Optimal transport (OT) compares two distributions of mass by the cheapest way of turning one into the other. Let $\mu = \sum_i \mu_i \delta_{x_i}$ and $\nu = \sum_j \nu_j \delta_{y_j}$ be two discrete measures with equal total mass ($\sum_i \mu_i = \sum_j \nu_j = 1$) supported on points $x_i, y_j \in \mathbb{R}^2$, and let $C_{ij} = c(x_i, y_j)$ be the cost of moving one unit of mass from $x_i$ to $y_j$. Monge's formulation looks for a map that sends each source point to one target point; Kantorovich's relaxation, which always has a solution, looks for a *transport plan* $\pi \in \mathbb{R}_{\ge 0}^{n \times m}$ with $\pi_{ij}$ the amount of mass moved from $x_i$ to $y_j$:

$$
W(\mu, \nu) = \min_{\pi \in \Pi(\mu, \nu)} \langle \pi, C \rangle = \min_{\pi} \sum_{i,j} \pi_{ij} C_{ij}
\quad\text{s.t.}\quad \sum_j \pi_{ij} = \mu_i,\;\; \sum_i \pi_{ij} = \nu_j .
\tag{2.6}
$$

With $c(x, y) = \|x - y\|_2^2$, $W$ is the squared 2-Wasserstein distance. Unlike a pixel-wise comparison, (2.6) depends on the *geometry* of the supports: two identical bumps at distance $\Delta$ have cost proportional to $\Delta^2$ however far apart they are, and the cost decreases smoothly as they approach.

### 2.6.3 Entropic regularization and the Sinkhorn algorithm

Problem (2.6) is a linear program with $nm$ unknowns; for two $24 \times 24$ images that is $576^2 \approx 3.3 \times 10^5$ unknowns per image pair, too slow to solve exactly inside every optimization step. Cuturi (2013) proposed adding an entropy term,

$$
W_\varepsilon(\mu, \nu) = \min_{\pi \in \Pi(\mu,\nu)} \langle \pi, C \rangle - \varepsilon\, H(\pi), \qquad H(\pi) = -\sum_{ij} \pi_{ij}(\log \pi_{ij} - 1),
\tag{2.7}
$$

which makes the problem strictly convex and gives its solution the form

$$
\pi^\star_{ij} = \exp\!\Big(\frac{u_i + v_j - C_{ij}}{\varepsilon}\Big) = a_i\, K_{ij}\, b_j, \qquad K_{ij} = e^{-C_{ij}/\varepsilon},
$$

for two dual vectors $u \in \mathbb{R}^n$, $v \in \mathbb{R}^m$ (equivalently the scalings $a = e^{u/\varepsilon}$, $b = e^{v/\varepsilon}$). The marginal constraints then reduce to alternately rescaling rows and columns of $K$, the classical Sinkhorn–Knopp iteration: $a \leftarrow \mu / (K b)$, $b \leftarrow \nu / (K^\top a)$. Every iteration is a matrix-vector product, so it runs on a GPU in batched form, and the whole computation is a chain of differentiable operations, which is what makes the loss usable inside backpropagation (Peyré and Cuturi 2019).

For small $\varepsilon$ the scalings $a, b$ overflow, so the iteration is carried out in the log domain on the duals. Writing

$$
M_{ij}(u, v) = \frac{-C_{ij} + u_i + v_j}{\varepsilon}, \qquad \operatorname{lse}_j(M)_i = \log \sum_j e^{M_{ij}},
$$

the updates become

$$
u_i \leftarrow \varepsilon\big(\log \mu_i - \operatorname{lse}_j(M(u, v))_i\big) + u_i, \qquad
v_j \leftarrow \varepsilon\big(\log \nu_j - \operatorname{lse}_i(M(u, v)^\top)_j\big) + v_j .
\tag{2.8}
$$

Expanding $\operatorname{lse}_j(M(u,v))_i = u_i/\varepsilon + \log\sum_j e^{(v_j - C_{ij})/\varepsilon}$ shows that (2.8) is exactly the standard dual update $u_i = \varepsilon \log \mu_i - \varepsilon \log \sum_j e^{(v_j - C_{ij})/\varepsilon}$, written so that the previous $u$ is added and cancelled inside the log-sum-exp, which is numerically convenient. After $L$ iterations the plan is $\pi = \exp(M(u, v))$ and the loss returned is the transport cost of that plan,

$$
\mathcal{L}_{\text{OT}} = \langle \pi, C \rangle = \sum_{ij} \pi_{ij} C_{ij} .
\tag{2.9}
$$

Note that (2.9) is the *sharp* cost without the entropy term, and that the reference code uses the raw cost rather than the debiased Sinkhorn divergence $2W_\varepsilon(\mu,\nu) - W_\varepsilon(\mu,\mu) - W_\varepsilon(\nu,\nu)$ (available as `sinkhorn_normalized` but switched off by `normalize=False`). Gradients flow through $\pi$ into $u, v$ (the $L$ iterations are unrolled by autograd) and into $\mu$, hence into the canvas.

### 2.6.4 The Sinkhorn loss as implemented for painting

The reference class `SinkhornLoss` turns a canvas and a target into measures as follows.

1. **Downsampling.** If the patch side exceeds 24 pixels, canvas and target are resized to $24 \times 24$ with area interpolation. OT is used for coarse placement, and $576$ support points keep the $576 \times 576$ cost matrix small.
2. **Support.** Both measures live on the same grid of normalized pixel centres, $x_{ij} = (i/h,\; j/w)$ for $i < h$, $j < w$, so the cost matrix is $C_{(ij),(kl)} = (i - k)^2/h^2 + (j - l)^2/w^2$, the squared Euclidean distance on the unit square (`cost_matrix` with $p = 2$). It is computed once per call and is the same for canvas and target.
3. **Masses.** One colour channel $\kappa \in \{R, G, B\}$ is chosen uniformly at random at every call, and the masses are the pixel intensities of that channel: $\mu \propto C_{\kappa,:,:}$ (canvas) and $\nu \propto I_{\kappa,:,:}$ (target). Each is clamped to be non-negative, shifted by $10^{-9}$ so that no mass is exactly zero (which would make $\log \mu$ infinite), and normalized to sum to one. Picking one channel divides time and memory by three; over many steps all channels are used.
4. **Iterations.** $\varepsilon = 0.01$ on the unit square and $L = 5$ iterations of (2.8), starting from $u = v = 0$; the log-sum-exp adds $10^{-6}$ inside the logarithm to avoid $\log 0$.
5. **Output.** The cost (2.9) is computed per patch in the batch and averaged over the batch. The painting loss is

$$
\mathcal{L} = \beta_{\text{L1}}\,\mathcal{L}_{\text{pix}} + \beta_{\text{OT}}\,\mathcal{L}_{\text{OT}},
\tag{2.10}
$$

with defaults $\beta_{\text{L1}} = 1$, $\beta_{\text{OT}} = 0.1$, and the OT term is switched on by the flag `--with_ot_loss`.

Two consequences of the normalization in step 3 deserve a remark. Because both measures are normalized to unit mass, the OT term is insensitive to the overall brightness of the canvas and only measures *where* the intensity sits; the pixel term remains responsible for absolute values. And because mass is intensity, on a white canvas most of the mass is background rather than paint; how much this matters in practice is one of the questions the Week 5 ablation will answer rather than assume.

### 2.6.5 A one-dimensional worked example

Take a one-dimensional canvas on $[0, 1]$ and two non-overlapping bumps: the stroke $s(x) = \mathbb{1}[|x - a| \le \rho]$ and the target $t(x) = \mathbb{1}[|x - b| \le \rho]$ with $|a - b| > 2\rho$. Both have the same mass $2\rho$.

*Pixel loss.* $\mathcal{L}_{\text{pix}} = \int |s - t|\,dx = 4\rho$, a constant independent of $a$ as long as the supports are disjoint, so $\partial \mathcal{L}_{\text{pix}} / \partial a = 0$. Nothing in the pixel loss tells the stroke which way to go.

*Optimal transport.* After normalization both measures are uniform on intervals of equal length, so the optimal plan is the translation that shifts one interval onto the other, and with squared cost

$$
\mathcal{L}_{\text{OT}} = \int_{-\rho}^{\rho} \frac{1}{2\rho}\,(a - b)^2\, dx = (a - b)^2, \qquad \frac{\partial \mathcal{L}_{\text{OT}}}{\partial a} = 2\,(a - b).
$$

The gradient is non-zero for any $a \neq b$, its sign points from the stroke towards the target, and its magnitude grows with the distance, so a gradient step moves the stroke in the right direction regardless of overlap. Once the stroke arrives, $(a - b)^2 \to 0$ and the pixel loss takes over to refine shape and colour. With entropic regularization the plan is smeared by a scale $\sqrt{\varepsilon}$ (on the unit square, $\varepsilon = 0.01$ corresponds to a standard deviation of about $0.1$, or two to three cells of the $24 \times 24$ grid), so the cost is a slightly smoothed version of $(a - b)^2$, and the gradient is still monotone in $a - b$. The two-dimensional case behaves in the same way with $\|\cdot\|_2^2$ in place of $(a - b)^2$.

## 2.7 Style loss

Style transfer is a stretch goal of this project, but its loss is part of the reference code and is short to state. Following Gatys et al. (2016), style is represented by the Gram matrices of VGG feature maps. Let $\Phi_l(X) \in \mathbb{R}^{c_l \times h_l w_l}$ be the (flattened) activation of layer block $l$ of a VGG-16 network, pre-trained on ImageNet and evaluated on an input normalized with the ImageNet mean and standard deviation and resized to $224 \times 224$. The Gram matrix is

$$
G_l(X) = \frac{1}{c_l h_l w_l}\, \Phi_l(X)\, \Phi_l(X)^\top \in \mathbb{R}^{c_l \times c_l},
$$

and the style loss between the canvas $C$ and a style image $I_{\text{sty}}$ is

$$
\mathcal{L}_{\text{sty}} = \sum_{l \in \mathcal{B}} \big\| G_l(C) - G_l(I_{\text{sty}}) \big\|_F^2 .
\tag{2.11}
$$

The set of blocks $\mathcal{B}$ selects what is transferred. Transfer mode 0 (*colour only*) uses the first two blocks (`features[0:4]` and `features[4:9]` of VGG-16), whose Gram matrices mainly encode colour statistics and fine texture; transfer mode 1 (*colour and texture*) uses four blocks (up to `features[16:23]`), which adds larger-scale texture. Max-pooling layers are replaced by average pooling, which gives smoother gradients for optimization, as recommended by Gatys et al. During style transfer the pixel term of (2.10) is evaluated in luminance only (`ignore_color=True`) so that content is preserved while colour is free to follow the style, and the strokes' parameters, not the pixels, are what is optimized, which keeps the result a valid painting. The reference code also contains a perceptual (feature-matching) loss on the same VGG blocks, which is not used by default.

## 2.8 Stroke initialization and progressive rendering

### 2.8.1 Error-map sampling

Gradient descent from a uniformly random stroke is slow, and a random stroke is often placed where the canvas is already correct. The reference engine instead samples each new stroke from a probability map derived from the current error. For each grid cell (Section 2.8.2) with target patch $I$ and current canvas patch $C$,

$$
E = \sum_{c \in \{R,G,B\}} |I_c - C_c|, \qquad
\tilde{E} = \big(\operatorname{blur}_{k}(E)\big)^4, \qquad
p = \frac{\tilde{E}}{\sum \tilde{E}},
\tag{2.12}
$$

where $\operatorname{blur}_k$ is a box filter of side $k = \lfloor S / 8 \rfloor$ pixels and the fourth power sharpens the map so that the largest errors dominate. A pixel $(i^\star, j^\star) \sim p$ is drawn, resized to the renderer's canvas, and becomes the stroke's position: the centre $(x_c, y_c)$ for oil and tape, and all three Bezier control points for watercolour and marker (the stroke starts as a dot and the optimizer stretches it). The initial colour is the target colour at that pixel, duplicated for the two-colour brushes. The remaining shape parameters are drawn from narrow ranges: sizes $w, h$ or $r_0, r_2 \sim \mathcal{U}(0.1, 0.25)$, oil orientation $\theta_\varphi \sim \mathcal{U}(0,1)$, tape orientation $0$. Initial opacity is $\mathcal{U}(0.98, 1)$ for oil and watercolour and $\mathcal{U}(0.8, 0.98)$ for marker and tape. If the error map is identically zero the sampler falls back to a uniform map.

### 2.8.2 Grid subdivision

A neural renderer emits a stroke on a patch of $S \times S$ pixels ($S = 32$ or $128$). To paint a $512 \times 512$ canvas with fine detail, the image is divided into an $m \times m$ grid of cells; every cell is resized to $S \times S$, receives its own strokes with coordinates expressed *inside the cell*, and all $m^2$ cells are optimized together as one batch on the GPU. This is the fixed-grid mode of the reference `demo.py`. With $N$ strokes in total, each cell gets $\lfloor N / m^2 \rfloor$ strokes.

After optimization the per-cell coordinates are mapped back to the full canvas. For a cell in row $\rho$ and column $\gamma$ (zero-based),

$$
x^{\text{global}} = \frac{\gamma}{m} + \frac{x^{\text{cell}}}{m}, \qquad
y^{\text{global}} = \frac{\rho}{m} + \frac{y^{\text{cell}}}{m}, \qquad
\text{sizes: } w, h, r_0, r_2 \;\to\; \frac{w}{m}, \frac{h}{m}, \frac{r_0}{m}, \frac{r_2}{m},
\tag{2.13}
$$

and rotation angles are unchanged. For the Bezier brushes only the two endpoints are transformed; the relative control point $(x_1, y_1)$ of Section 2.2.3 is defined with respect to the endpoints and therefore needs no change, which is the payoff of that parameterization. The re-normalized strokes of all cells are then rasterized *procedurally* onto a single full-resolution canvas, so that the final image is the true brush appearance and not the neural approximation.

### 2.8.3 Progressive (coarse-to-fine) schedule

The progressive mode of `demo_prog.py` runs the grid at every size from $1 \times 1$ up to $M \times M$. The stroke budget $N$ is divided evenly per cell across all levels,

$$
N_{\text{cell}} = \left\lfloor \frac{N}{\sum_{m=1}^{M} m^2} \right\rfloor,
\qquad\text{so that the total number of strokes is } N_{\text{cell}} \sum_{m=1}^{M} m^2 \le N .
\tag{2.14}
$$

For the default $N = 500$ and $M = 5$ this gives $N_{\text{cell}} = 9$ strokes per cell and $9 \times 55 = 495$ strokes in total: nine large strokes at level 1, thirty-six at level 2, and so on to 225 small strokes at level 5. At each level:

1. the target is split into $m^2$ patches, and the canvas painted at the previous levels is rasterized procedurally at full resolution and split the same way, so every cell starts from what coarser levels have already painted;
2. the $N_{\text{cell}}$ strokes of every cell are initialized (Section 2.8.1) one at a time: after stroke $k$ is sampled, all strokes $0, \dots, k$ of the cell are re-rendered, composited in order onto the cell's starting canvas with (2.1), and their parameters are updated jointly with RMSprop (centred, learning rate $0.002$) for $\lfloor 500 / N_{\text{cell}} \rfloor$ steps, which is 55 steps for $N_{\text{cell}} = 9$;
3. after every update the shape parameters are clamped to $[0.1, 0.9]$ and the colour and opacity parameters to $[0, 1]$, so that strokes cannot leave their cell or take invalid values;
4. the finished strokes are mapped to global coordinates with (2.13), their order is randomized across cells, and they are appended to the global stroke list.

Because every step at a level re-renders all strokes added so far in that cell, the cost per step grows linearly with $k$, and the total number of renderer evaluations per level is about $m^2 \sum_{k=1}^{N_{\text{cell}}} k \cdot 55$. The cells are batched, so on a GPU the wall time is dominated by the sequential loop over steps rather than by $m^2$.

The schedule is the neural analogue of Hertzmann's layers. Coarse levels use few, large strokes that fix the global colour layout; fine levels add many small strokes only where the error map is still large, because the error-driven sampler is evaluated against the canvas inherited from coarser levels. It also solves a capacity problem: a renderer with a $32 \times 32$ output cannot draw a small detail of a $512 \times 512$ image directly, but at level 5 each cell is a $102 \times 102$ region of the full canvas, so a stroke of size $0.1$ in cell coordinates is about ten pixels wide, which is the detail scale a painting needs. The comparison of progressive, fixed-grid and full-image modes at equal stroke budgets is the subject of research question RQ2.

## 2.9 Summary of notation

| Symbol | Meaning | Where defined |
| :--- | :--- | :--- |
| $I$ | Target image, $3 \times H \times W$, values in $[0,1]$ | 2.1 |
| $C_k$ | Canvas after $k$ strokes; $C_0$ is black or white | 2.3 |
| $\boldsymbol{\theta}_k \in [0,1]^d$ | Parameter vector of stroke $k$ | 2.2 |
| $\boldsymbol{\theta}_s, \boldsymbol{\theta}_c, A$ | Shape, colour and opacity groups, sizes $d_s, d_c, d_a = 1$ | 2.2 |
| $\varphi$ | Rotation angle, $\varphi = \pi \theta_\varphi$ (`theta` in code) | 2.2 |
| $P_0, P_1, P_2$ | Control points of a quadratic Bezier; $P_1$ stored relative to the endpoints' box | 2.2.3 |
| $W$, $S$ | Rasterizer canvas side; neural renderer output side ($S = 128$ or $32$) | 2.2, 2.4 |
| $\operatorname{px}(x)$ | Normalized coordinate to pixel index, $\lfloor x(W-1) + 0.5 \rfloor$ | 2.2 |
| $F_k$, $\hat{A}_k$ | Foreground and opacity map of stroke $k$ (rasterized or predicted) | 2.3, 2.4 |
| $\mathcal{R}$ | Renderer mapping a stroke list to a canvas | 2.1 |
| $G_\phi$, $\phi$ | Neural renderer and its weights; $m$ mask, $c$ colour field | 2.4 |
| $\mathcal{L}_{\text{pix}}, \mathcal{L}_{\text{Char}}$ | $L_1$ and Charbonnier pixel losses | 2.5 |
| $\mu, \nu$ | Normalized intensity measures of canvas and target (one channel, $24 \times 24$) | 2.6 |
| $C_{ij}$, $\pi$, $u, v$ | OT cost matrix, transport plan, dual potentials | 2.6 |
| $\varepsilon$, $L$ | Entropic regularization ($0.01$) and Sinkhorn iterations ($5$) | 2.6 |
| $\mathcal{L}_{\text{OT}}$ | Sinkhorn transport cost $\langle \pi, C \rangle$ | 2.6 |
| $\beta_{\text{L1}}, \beta_{\text{OT}}$ | Loss weights, defaults $1$ and $0.1$ | 2.6 |
| $G_l$, $\mathcal{L}_{\text{sty}}$ | VGG Gram matrix of block $l$; style loss | 2.7 |
| $E$, $p$ | Error map and stroke-placement probability map | 2.8 |
| $m$, $M$ | Current grid size and maximum grid size | 2.8 |
| $N$, $N_{\text{cell}}$ | Total stroke budget; strokes per cell per level | 2.8 |
| $(\rho, \gamma)$ | Row and column of a grid cell | 2.8 |

## References cited in this chapter

- Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport. *NeurIPS*.
- Gatys, L. A., Ecker, A. S., Bethge, M. (2016). Image style transfer using convolutional neural networks. *CVPR*.
- Haeberli, P. (1990). Paint by numbers: Abstract image representations. *SIGGRAPH*.
- Hertzmann, A. (1998). Painterly rendering with curved brush strokes of multiple sizes. *SIGGRAPH*.
- Huang, Z., Heng, W., Zhou, S. (2019). Learning to paint with model-based deep reinforcement learning. *ICCV*.
- Litwinowicz, P. (1997). Processing images and video for an impressionist effect. *SIGGRAPH*.
- Liu, S., Lin, T., He, D., Li, F., Deng, R., Li, X., Ding, E., Wang, H. (2021). Paint Transformer: Feed forward neural painting with stroke prediction. *ICCV*.
- Nakano, R. (2019). Neural painters: A learned differentiable constraint for generating brushstroke paintings. *arXiv:1904.08410*.
- Peyré, G., Cuturi, M. (2019). Computational optimal transport. *Foundations and Trends in Machine Learning* 11(5–6).
- Zou, Z., Shi, T., Qiu, S., Yuan, Y., Shi, Z. (2021). Stylized neural painting. *CVPR*.
