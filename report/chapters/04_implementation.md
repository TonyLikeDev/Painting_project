# 4. Implementation

This chapter describes how the method of Chapter 2 is implemented in the `neural_painter`
package. It grows week by week; this version covers the neural renderer (Week 4) and the
optimization of strokes against a target image (Week 5). Unless a
sentence says otherwise, every number in it comes from a script or test in the repository, and
the experiment folder it was produced in is named next to it.

## 4.1 The neural renderer

### 4.1.1 Interface

`models/neural_renderer.py` defines `NeuralRenderer`, an ordinary `torch.nn.Module` whose forward
pass maps a batch of stroke vectors of shape `(N, d)`, with entries in $[0, 1]$, to a foreground
and an opacity map, each of shape `(N, 3, S, S)`. The brush supplies $d$, the shape dimension $d_s$
and the output size $S$ through its YAML configuration (Section 2.2), so one class serves all
four brushes at both sizes ($S = 128$ for the full renderer, $S = 32$ for the light one). The
architecture is the fusion network of Section 2.4.2 with the layer sizes of Table 2.2: a shape
decoder that sees only the shape parameters, a colour decoder that sees the whole vector, and
the fusion of equation (2.3).

Two kinds of weights share the class: the authors' 2021 checkpoints and renderers trained in
this project. A single function, `load_renderer`, opens either kind of file, so the painter never
needs to know which one it holds, and both can be compared under identical painting code.

### 4.1.2 Differences from the reference network

The reference implementation (`networks.py`) is reproduced exactly in what it computes. What
changed is how it is written and one redundancy that could be removed without changing any output
(Table 4.1).

**Table 4.1.** Differences between the reference network and `NeuralRenderer`.

| Reference | This implementation | Consequence |
| :--- | :--- | :--- |
| A module-level `device` global; a tensor is created on it inside `forward` | A plain `nn.Module`, moved with `.to(device)` | Runs on CUDA, MPS and CPU without edits |
| `x.squeeze()` in the shape decoder | An explicit reshape | A batch of one is no longer collapsed into a vector |
| The oil brush is recognised by the string `'oilpaintbrush'` | `alpha_forced_to_one` in the brush YAML | Opacity handling is configuration, not code |
| The colour decoder has 6 output channels and discards 3 | 3 output channels | 3,072 (full) and 6,144 (light) parameters fewer, identical outputs |
| No output non-linearity | Optional sigmoid on both heads: on for newly trained renderers, off for the 2021 weights | New renderers are bounded in $[0, 1]$ |

One property of the oil brush deserves a sentence, because the Week 2 reading of the reference
code got it slightly wrong. The opacity *map* of an oil stroke ignores the opacity parameter
$A$ completely, as intended (the output is exactly the mask). But the colour decoder receives all
$d$ parameters, $A$ included, so the *foreground* still depends on it weakly. Measured on 256 random
strokes through the pretrained oil renderers, changing $A$ from 0 to 1 changes the foreground by
about 1 % (light) and 1.5 % (full) on average, and the gradient with respect to $A$ is about 1 % of
the gradient with respect to the colour parameters (a one-off measurement; the test
`test_pretrained_oil_foreground_depends_only_weakly_on_the_alpha_parameter` keeps the effect
below 5 % and the opacity map exactly independent). The painter therefore keeps $A$ fixed for the
oil brush.

### 4.1.3 Loading the pretrained renderers

The adapter `load_original_checkpoint` maps an original `last_ckpt.pt` onto `NeuralRenderer`: the
key prefixes `huangnet.` and `dcgan.` become `shape_decoder.` and `color_decoder.`, and the last
weight of the colour decoder is cut to its first three output channels. All other layer names are
unchanged on purpose, so the conversion is a rename and a slice, not a re-implementation.

The conversion was checked three ways. (i) For every one of the 8 pretrained checkpoints (four
brushes, both sizes) the foreground and opacity produced by the original network and by the adapted
network on the same random strokes differ by exactly $0$ in the measurement (float32, CPU); the
test suite requires agreement to $10^{-6}$. (ii) The same comparison is made for original networks
with random weights and random batch-normalization statistics, which exercises the key mapping
without needing the downloads. (iii) Rendered strokes of the pretrained light oil renderer are
compared with the procedural rasterizer (Section 4.1.6).

The checkpoints are Python pickles. Loading uses `torch.load(..., weights_only=True)` with a small
allow-list for the NumPy scalar that the authors stored as `best_val_acc`, so a file that contains
anything else is refused; full unpickling is available only through an explicit `trusted=True`.

### 4.1.4 Correctness of the gradient

The renderer exists to provide $\partial \hat{F} / \partial \boldsymbol{\theta}$ and
$\partial \hat{A} / \partial \boldsymbol{\theta}$ (Section 2.4.3), so this is tested directly. For
the four light renderers and the full oil renderer, with random weights and statistics, the
gradient of a random linear functional of $(\hat{F}, \hat{A})$ computed by autograd is compared with
central finite differences, $\varepsilon = 10^{-6}$, in float64, for *every* stroke parameter. They
agree to a relative tolerance of $10^{-5}$.

### 4.1.5 Training

Training follows the reference recipe (Section 2.4.1) by default, so a retrained renderer is
directly comparable to the 2021 one: uniform random stroke vectors, rasterized on the fly in
training mode at the renderer's own resolution; 50,000 strokes per epoch; batch 64; Adam with
learning rate $2 \times 10^{-4}$; the learning rate divided by 10 every quarter of the run (the
original divides every 100 of 400 epochs); the loss $100 \cdot \tfrac{1}{2}(\mathrm{MSE}_F +
\mathrm{MSE}_A)$; and $\mathcal{N}(0, 0.02^2)$ initialization. Two variants are available:
`--loss l1_ssim` and `--activation none`.

The engineering is dominated by one fact: *the teacher is the bottleneck*. Rasterizing one oil
stroke costs 2.1 ms on one core at both resolutions, because the cost is the $394 \times 394$ brush
texture and not the canvas, so a single thread feeds only 475 strokes per second and a 50,000-stroke
epoch would take 105 s while the GPU waited. The rasterizer is OpenCV and NumPy work that releases
the interpreter lock, so strokes are produced by a pool of threads inside the training process:
1,476 strokes per second with 4 threads and 1,809 with 8. Worker processes were tried first and
abandoned: each re-imports PyTorch, and on Windows the memory commit of eight of them ran the
machine out of commit memory. Every batch is generated from the seed (run seed, pass, batch index),
so a run is reproducible whatever the number of threads and a resumed run sees exactly the strokes
an uninterrupted one would have seen. Checkpoints are written atomically, and `--resume` continues
the learning curve, the optimizer and the schedule.

On the desktop (RTX 3070, i5-13400F) the light oil renderer trains at about 1,600 strokes per
second, 31 s per epoch of 50,000 strokes.

### 4.1.6 Fidelity

*Protocol.* Every renderer is scored on the same fixed set of $n = 1{,}000$ held-out random
strokes per brush and size, rasterized in training mode. PSNR is computed from the mean squared
error over the whole set and all pixels and channels,

$$
\mathrm{PSNR} = 10 \log_{10} \frac{1}{\mathrm{MSE}}, \tag{4.1}
$$

separately for the foreground and the opacity map, and averaged; this is the quantity the reference
code reports as its validation accuracy. SSIM is the mean over strokes (Gaussian window, 11 pixels,
$\sigma = 1.5$).

*The protocol reproduces the authors' numbers.* Table 4.2 scores the eight pretrained renderers. The
last two columns compare our measurement with the validation accuracy that the authors stored in
their checkpoints, measured on their own random validation set; the two agree within 0.25 dB for
every renderer, which validates the adapter, the rasterizer in training mode and the metric at once.

**Table 4.2.** The pretrained renderers on 1,000 held-out strokes (experiment
`2026-10-02_fidelity_week4`, `scripts/renderer_fidelity.py`).

| Brush | Size | Epochs | PSNR fg | PSNR $\alpha$ | PSNR mean (ours) | Reported by the authors | SSIM fg | SSIM $\alpha$ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Oil paint | 128 | 43 | 27.85 | 22.56 | 25.20 | 25.26 | 0.942 | 0.934 |
| Oil paint | 32 | 400 | 27.38 | 21.38 | 24.38 | 24.31 | 0.944 | 0.920 |
| Watercolour | 128 | 484 | 30.87 | 30.49 | 30.68 | 30.83 | 0.973 | 0.970 |
| Watercolour | 32 | 400 | 25.80 | 25.72 | 25.76 | 25.84 | 0.925 | 0.917 |
| Marker pen | 128 | 198 | 28.84 | 28.58 | 28.71 | 28.75 | 0.962 | 0.953 |
| Marker pen | 32 | 400 | 24.82 | 24.36 | 24.59 | 24.83 | 0.914 | 0.903 |
| Coloured tape | 128 | 412 | 33.59 | 33.46 | 33.53 | 33.61 | 0.987 | 0.985 |
| Coloured tape | 32 | 400 | 28.39 | 28.27 | 28.33 | 28.23 | 0.959 | 0.957 |

Two things stand out. Oil paint is the weakest renderer at both sizes, and the authors' released
*full* oil renderer was trained for only 43 epochs, far fewer than the other seven, so its score
mixes the difficulty of the brush with a short training. And for the oil brush the opacity map is
clearly the harder output (22.6 and 21.4 dB, against 27.9 and 27.4 dB for the foreground), while
the other three brushes score nearly the same on both outputs.

*Our renderer.* The light oil renderer was retrained with the recipe of Section 4.1.5 for 100
epochs, which took 51 minutes on the RTX 3070 (run `2026-10-02_train_oilpaintbrush_light_100ep`).
The reference trains the same network for 400 epochs, so this is a quarter of the training, with the
learning rate divided by 10 every 25 epochs instead of every 100. On the same 1,000 strokes the
retrained renderer reaches 24.16 dB against 24.38 dB for the original, a difference of 0.22 dB and
inside the 1 dB target of the research plan; the SSIM is the same for the foreground and slightly
higher for the opacity map (Table 4.3).

**Table 4.3.** The retrained light oil renderer against the original (experiment
`2026-10-02_fidelity_week4`).

| Renderer | Epochs | PSNR fg | PSNR $\alpha$ | PSNR mean | SSIM fg | SSIM $\alpha$ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original (Zou et al. 2021) | 400 | 27.38 | 21.38 | 24.38 | 0.944 | 0.920 |
| Retrained here | 100 | 27.17 | 21.15 | 24.16 | 0.944 | 0.929 |
| Difference | | -0.21 | -0.23 | -0.22 | 0.000 | +0.009 |

The learning curve (Figure 4.1) shows where the gain comes from. At the base learning rate the PSNR
climbs to 23.4 dB in 25 epochs and then stalls. The first drop adds 0.52 dB, the second 0.06 dB and
the third 0.01 dB, and the last 40 epochs stay between 24.15 and 24.16 dB, so the remaining 0.22 dB
is not a matter of the final epochs. Whether the original's 400-epoch schedule would close it was
not tested: the target is met, and the extra training would cost about three more hours of the
desktop.

**Figure 4.1.** Training loss (left, log scale) and validation PSNR (right) of the retrained light
oil renderer. Dotted lines mark the learning-rate drops; the dashed line is the original renderer's
24.38 dB (`report/figures/renderer_training_curve_oil_light.png`).

Figure 4.2 shows eight held-out strokes rendered by the rasterizer, by the original renderer and by
ours. The two networks produce the same smooth silhouettes and colour gradients, and both miss what
a network of this size does not reproduce: the texture streaks inside an oil stroke and the
pixel-level edge of the rasterizer. In these eight strokes the original and the retrained renderer
cannot be told apart by eye.

**Figure 4.2.** Eight held-out strokes. Rows: foreground of the rasterizer, of the original light
renderer and of ours, then the same three rows for the opacity map
(`report/figures/renderer_comparison_oil_light.png`).

### 4.1.7 Cost

The quantity that matters for painting is the cost of the optimizer's inner loop: a forward and a
backward pass to the stroke parameters through the frozen renderer. `benchmark_renderer` measures it
for 64 strokes, with the clock read only after the device has finished (Table 4.4). The weights are
21 MB (light) and 72 MB (full) in float32, which is all a laptop has to hold.

**Table 4.4.** Cost of 64 strokes through the oil renderers. The `cuda` rows come from experiment
`2026-10-02_fidelity_week4`; the desktop `cpu` rows were measured separately on the idle desktop with 10
threads; the Apple M4 rows come from one run of `scripts/device_check.py`
(`2026-10-02_check_darwin_arm64_mps`, 4 CPU threads). Peak memory is the largest amount of tensor memory
allocated on the device. † On `mps` PyTorch has no peak counter: the figure is the memory the Metal driver
had allocated to the process when the run ended, so it is not comparable with the CUDA column.

| Device | Renderer | Parameters (M) | Forward (ms) | Forward + backward (ms) | Peak memory (GB) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `cuda` (RTX 3070) | light | 5.37 | 1.2 | 2.6 | 0.09 |
| `cuda` (RTX 3070) | full | 18.09 | 7.6 | 15.6 | 0.50 |
| `cpu` (i5-13400F) | light | 5.37 | 20.8 | 43.8 | - |
| `cpu` (i5-13400F) | full | 18.09 | 155.8 | 291.9 | - |
| `mps` (Apple M4) | light | 5.37 | 17.3 | 25.8 | 0.11 † |
| `cpu` (Apple M4) | light | 5.37 | 22.8 | 126.0 | - |
| `mps`, `cpu` (Apple M4) | full | 18.09 | | | *pending: the full checkpoint is not on the MacBook* |

The light renderer is six to seven times cheaper than the full one on both devices, and the GPU is
17 to 19 times faster than the CPU for either. The second number is worth setting beside the Week 1
baseline, in which the whole original pipeline, light renderer included, ran only 2.6 times faster
on the GPU than on the CPU. Taken together they suggest that the renderer itself is not where most
of the wall time goes; the work around it is. On the M4 the GPU's advantage is smaller: for the light
renderer it is 4.9 times in forward plus backward and only 1.3 times in the forward pass, against 2.1
times for the whole pipeline in the Week 1 baseline on the same machine. The M4 GPU is also 10 to 14
times slower than the RTX 3070 on this renderer (forward plus backward 25.8 against 2.6 ms, forward
17.3 against 1.2 ms).

## 4.2 Stroke optimization

Section 2.8 describes what the reference code optimizes. This section describes how the package
implements it (`pipeline/painter_engine.py`, `pipeline/stroke_sampler.py`, `losses/sinkhorn.py`,
`pipeline/metrics.py`), how each part was checked against the 2021 code, and then measures the
result. The aim of the implementation was to reproduce what the reference optimizes, in a form that
can be tested and run on `cuda`, `mps` and `cpu`, so that the experiments of Section 4.2.6 onwards
change one thing at a time against a faithful baseline.

### 4.2.1 The painting loop

`PainterEngine.paint` takes a batch of $B$ target blocks of shape $(B, 3, S, S)$, where $S$ is the
renderer's output size (32 for the light renderer), and places $N$ strokes on each block, one at a time.
For the $k$-th stroke of every block (the *anchor*):

1. a new stroke is drawn from the error map of the canvas that the previous anchor left behind
   (Section 4.2.3) and written into slot $k$ of the parameter tensors;
2. $T$ optimizer steps follow, each in the order of the reference: clamp the parameters, render strokes
   $0, \dots, k$ of every block with the neural renderer as one batch of $B(k+1)$ strokes, dilate the
   foreground and erode the opacity map with a $3 \times 3$ window (the neural stand-in for the
   rasterizer's inference step, Section 2.3.2), composite the strokes in order onto the starting canvas
   with (2.1), evaluate the loss (2.10), back-propagate, clamp again and take an optimizer step.

All strokes placed so far are optimized jointly, which is why the cost of a step grows with $k$.
Shape parameters are clamped to $[0.1, 0.9]$ so that a stroke never collapses onto the border of its
block, colour and opacity to $[0, 1]$. The optimizer is RMSprop with learning rate $0.002$, and the number
of steps per stroke is $T = \lfloor 500 / N \rfloor$, so every setting uses about 500 optimizer steps
whatever the stroke budget: 25 steps per stroke for $N = 20$ strokes per block, a single step for
$N = 500$. The defaults of `PainterConfig` are those of the reference run with the pixel loss alone; the
Sinkhorn term (Section 4.2.2), Adam and the centred form of RMSprop are options.

`paint_fixed_grid` turns this into a whole-image painter. The image is stretched to $mS$ pixels (by area
averaging; the reference interpolates bilinearly, see Section 4.2.11) and cut into an $m \times m$ grid of blocks, which are painted as one batch; the strokes of all blocks are mapped
to image coordinates with (2.13) and rasterized by the *procedural* rasterizer at 512 pixels, stroke $j$
of every block (the blocks in a random order drawn from the seeded generator) before stroke $j + 1$ of
any block, as in the reference. The final image is therefore the real brush texture and not the neural
approximation. The grid $m = 1$ is the full-image mode of the reference. The function returns the image,
the strokes in drawing order, and the loss and the PSNR of the neural canvas at every step, which the
experiments below use. Strokes are written in the layout of the reference `.npz` files (`x_ctt`,
`x_color`, `x_alpha`).

What was changed is how the loop is written, not what it computes (Table 4.5).

**Table 4.5.** Differences between the reference painting code and `PainterEngine`.

| Reference | Here | Reason |
| :--- | :--- | :--- |
| strokes, tensors and optimizer are attributes of one `Painter` class that reads module-level globals (device, paths) | one object owns them; the device is an argument | the engine can be tested on a toy renderer, and the same code runs on three devices |
| random numbers from the global state of NumPy, PyTorch and Python | one seed drives the initial parameters, the stroke sampler and the block shuffle; the Sinkhorn colour channel has its own seeded generator | a run is reproducible and two configurations can be compared on the same seed |
| loss and PSNR are printed at every step | the history is kept on the device and read once per anchor, and a callback replaces `print` | no device-to-host synchronisation inside the loop |
| the error map and the draw are computed block by block on the CPU | the error map is computed for the whole batch on the device; the draw is an inverse-CDF lookup | see Section 4.2.3 |
| the image is shrunk to the blocks' resolution with `cv2.resize`'s default, bilinear interpolation | area averaging (`cv2.INTER_AREA`); `PainterConfig.resize = "linear"` restores the original | the average is the better target: $+0.32$ dB (Section 4.2.11) |

### 4.2.2 The Sinkhorn loss

`SinkhornLoss` follows Section 2.6.4 step by step: area-downsample canvas and target to $24 \times 24$,
use one colour channel as the mass, run $L = 5$ log-domain iterations (2.8) at $\varepsilon = 0.01$ and
return the transport cost (2.9). Three details are worth stating because they decide whether the
loss can be trusted.

*It equals the reference.* On random masses over a $12 \times 12$ grid the value agrees with the
reference function to a relative error of $10^{-5}$ and the gradient with respect to the canvas mass to
$10^{-3}$ (float32; the reference builds the cost matrix from differences and this code from a product, so
the two differ in the last bits). The whole module, with its resizing and its channel choice, agrees to
$10^{-4}$ on $32 \times 32$ canvases for each of the three channels and for both the plain cost and the
debiased divergence. The reference also clamps the mass *in place* through `.data`, which silently
modifies the canvas it was given; here the clamp is out of place.

*Two numerical modes.* In the `reference` mode (the default) the log-sum-exp is
$\log(\sum e^{a} + 10^{-6})$ as in the reference, so results are comparable with the 2021 code; the
$10^{-6}$ keeps the logarithm finite when every term has underflowed, at the price of a small bias there.
The `exact` mode uses `torch.logsumexp`. On dense masses the two agree to $10^{-3}$, and on a measure that
puts all its mass in one corner pixel the exact mode stays finite in value and gradient.

*It runs on Apple GPUs.* The reference resizes with `F.interpolate(mode="area")`, which on `mps` becomes
an adaptive average pool that rejects sizes the output does not divide, and $32 \to 24$ is such a size.
`area_resize` computes the same result as two matrix products with averaging matrices (checked against
PyTorch for input sizes 32, 48, 100 and 128).

A final test checks the reason the loss exists (Section 2.6.1). Two Gaussian blobs that do not overlap
give the pixel loss a gradient with respect to the position of the first one that is smaller than
$10^{-3}$ of the transport cost's gradient, and the transport gradient points from the blob towards the
target; a small step against it lowers the transport cost.

### 4.2.3 Error-map sampling

`StrokeSampler` implements (2.12) for a whole batch of blocks. `box_blur` is a normalized box filter that
equals `cv2.blur` for every kernel size, including the even sizes that arise here ($32 / 8 = 4$ pixels),
for which OpenCV's window is asymmetric and its border reflects without repeating the edge pixel; this is
checked pixel by pixel against OpenCV in the tests, because a one-pixel shift in the error map moves every
stroke. The draw is an inverse-CDF lookup on a float64 cumulative sum of the 512 $\times$ 512 map; the
reference calls `numpy.random.choice` over 262,144 categories for every block and every stroke. Over 500
draws on the same error map the mean and the standard deviation of the stroke centres agree with the
reference sampler to within 0.04, and strokes land where the canvas is wrong and take the target colour at
their centre. A perfect canvas, whose error map is zero everywhere, falls back to a uniform map.

### 4.2.4 Verification

`tests/test_optimizer.py` runs the engine on a toy renderer and on the pretrained oil renderer. The loss
falls over 50 steps on the toy target, parameters stay inside their ranges, the same seed gives the same
strokes and a different seed does not, only the strokes placed so far are drawn, a given starting canvas
is the base of the composite, and a full-image painting is the one-block case of the grid painter. Further tests cover the probe hook, the uniform start, the resize option and, on a CUDA machine, the exact repeatability under `cudnn.deterministic`. The
Sinkhorn and sampler comparisons above are in `tests/test_sinkhorn.py` and `tests/test_stroke_sampler.py`
and need the original repository only for the reference side of each comparison.

### 4.2.5 Experimental protocol

All experiments of this section use the frozen evaluation set of Section 5.2 (30 images of $512 \times 512$
pixels, `data/eval_set`), the oil brush, the authors' pretrained light oil renderer (the one retrained in
Section 4.1.6 is within 0.22 dB of it and is not used here, so that every number can be compared with the
2021 code), a white canvas, and the fixed grid of Section 4.2.1: $5 \times 5$ blocks of 32 pixels, 20 strokes
per block (500 in all) and 25 optimizer steps per stroke, 500 steps with RMSprop at learning rate $0.002$.
Every configuration is run with the seeds 0, 1 and 2, and two configurations that are compared use the same
seeds, so they start from the same initial parameters and the same random numbers.

*Scores.* A run is scored on the *finished* painting: the strokes of all blocks are rasterized procedurally at
512 pixels and compared with the 512-pixel target by PSNR (4.1) and by SSIM (same definition as in
Section 4.1.6). Three further numbers are recorded: the PSNR of the *neural* canvas (the $160 \times 160$
mosaic of the 25 blocks as the optimizer sees it), the first optimizer step at which that PSNR reaches 22 dB
and 24 dB, and the time of the optimization loop. The neural canvas scores higher than the finished painting
because it is a smaller image and is drawn by the surrogate; it is the only number available during the
optimization, which is why steps to a threshold are read from it. LPIPS needs the AlexNet weights, a
230 MB download, and is not part of the tables yet.

*Summaries.* A table cell is the mean over the three seeds of the average over the images, plus or minus the
standard deviation of the three seed means. With only three seeds that spread says how repeatable a number
is; it is not an interval. A comparison of two configurations is made *per image*. The three seeds of an
image are repeated measurements of one photograph, not independent photographs, so they are averaged first,
and the sample is the $n$ images:

$$
\bar{d} = \frac{1}{n} \sum_{i=1}^{n} d_i, \qquad
d_i = \frac{1}{3} \sum_{s=0}^{2} \big( m^{\text{other}}_{i,s} - m^{\text{base}}_{i,s} \big), \tag{4.2}
$$

where $m_{i,s}$ is a score of image $i$ and seed $s$. The tables give $\bar{d}$, its 95 % confidence interval
$\bar{d} \pm t_{0.975,\,n-1}\, s_d / \sqrt{n}$, the share of images with $d_i > 0$ (a tie is not a win), and the
two-sided $p$ value of Wilcoxon's signed-rank test, which does not assume that the $d_i$ are normal.

*Repeating a run.* A seed fixes the initial parameters, the draws of the stroke sampler and the order of the
blocks, but not the last bits of the GPU computation. cuDNN may pick different convolution algorithms from one
call to the next, and a painting is a chaotic computation (500 optimizer steps through clamps and max-pools), so two
runs of one seed on the RTX 3070 end in different paintings. Sixty (image, configuration, seed) cells were run
twice, because Ablation A and the weight sweep both contain the pixel-only loss and the weight 0.1 on the same 10
images: none of the 60 pairs has the same PSNR, the difference has no bias ($-0.004$ dB on average), its median
size is 0.10 dB, its 90th percentile 0.29 dB and its largest value 0.85 dB, which makes the standard deviation of
one run 0.16 dB. That is as large as the spread between seeds (median 0.10 dB within an image), so single runs
carry little information and the conclusions rest on averages over images and seeds. Setting
`torch.backends.cudnn.deterministic` (`core.device.set_deterministic`, the option `--deterministic` of the
experiment scripts) makes the strokes of two runs identical, which was checked on one image; the experiments of
this section were run without it, so they can be reproduced statistically but not bit for bit. On the CPU a run
is exactly repeatable.

### 4.2.6 Ablation A: pixel loss against pixel loss plus Sinkhorn

Research question RQ1 asks whether the optimal-transport term improves the painting. Ablation A answers it
for the setting of the reference code: the pixel loss alone, which is the reference's default, against the
pixel loss plus the Sinkhorn term at its default weight $\beta_{\text{OT}} = 0.1$, on the 30 evaluation images
with three seeds each, 180 paintings in all (experiment `2026-10-03_ablation_a_pixel_vs_sinkhorn`).

**Table 4.6.** Ablation A: mean over the three seeds of the image average, plus or minus the standard
deviation of the three seed means. "Steps to 22 dB" is the median over the runs whose neural canvas reaches
22 dB, with the number of runs that do in brackets.

| Loss | PSNR (dB) | SSIM | Neural-canvas PSNR (dB) | Steps to 22 dB | Optimization time (s) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Pixel ($L_1$) | 18.67 $\pm$ 0.03 | 0.490 $\pm$ 0.001 | 23.65 $\pm$ 0.05 | 335 (75 of 90) | 6.17 $\pm$ 0.03 |
| Pixel + Sinkhorn ($\beta_{\text{OT}} = 0.1$) | 18.69 $\pm$ 0.01 | 0.490 $\pm$ 0.001 | 23.69 $\pm$ 0.05 | 335 (75 of 90) | 15.99 $\pm$ 0.05 |

*The term made no measurable difference to the finished painting.* The two losses give the same mean PSNR to
within 0.02 dB and the same SSIM (Table 4.6). The paired comparison per image (Table 4.7) puts the mean
difference at $+0.011$ dB with a 95 % confidence interval of $[-0.027, +0.049]$, so a gain above 0.05 dB or a
loss above 0.03 dB is excluded; the term wins on 14 of the 30 images and loses on 16, and Wilcoxon's test
gives $p = 0.98$. The SSIM difference is $+0.001$ ($[-0.000, +0.002]$, $p = 0.16$). For scale, the baseline
paintings of different images range from 13.5 to 25.7 dB, and the same image painted with three seeds
varies by 0.085 dB (median standard deviation over images and losses), so differences of the size seen here are
about the size of the seed noise.

**Table 4.7.** Ablation A, paired differences (Sinkhorn minus pixel), one difference per image with its three
seeds averaged, $n = 30$.

| Metric | Mean difference | 95 % CI | Images where Sinkhorn is better | Wilcoxon $p$ |
| :--- | ---: | :--- | ---: | ---: |
| PSNR (dB) | $+0.011$ | $[-0.027, +0.049]$ | 14 of 30 | 0.98 |
| SSIM | $+0.001$ | $[-0.000, +0.002]$ | 18 of 30 | 0.16 |

*Nor did it make the optimization converge faster.* Figure 4.3 plots the PSNR of the neural canvas against
the optimizer step. The two curves coincide; their paired difference stays between $-0.02$ and $+0.08$ dB, and
its 95 % band contains zero at 499 of the 500 steps. The step at which the neural canvas first reaches 22 dB is
335 for both losses, and 24 dB is reached at step 416 (42 runs of 90) without the term and 400 (41 runs) with it.
These thresholds mostly record how many strokes have been placed, because the canvas improves as strokes are
added, so they would be a weak test of a faster optimizer even if the curves differed.

**Figure 4.3.** PSNR of the neural canvas against the optimizer step, mean over 30 images and three seeds
with a 95 % confidence band (left), and the paired difference between the two losses (right)
(`report/figures/ablation_a_pixel_vs_sinkhorn_convergence.png`).

*It cost 2.6 times as much.* The optimization loop takes 6.17 s per painting with the pixel loss and 15.99 s
with the Sinkhorn term (a per-run ratio with median 2.61, range 2.37 to 2.69, on the RTX 3070 with the process at
below-normal priority; the final 512-pixel rasterization adds 1.4 s to both). Each step evaluates, for each of
the 25 blocks, a $576 \times 576$ cost matrix and ten log-sum-exps, forward and backward, which are
memory-bound operations on tensors of 8.3 million entries.

**Figure 4.4.** Target, pixel-only painting and pixel + Sinkhorn painting (seed 0) for three images chosen by
a rule and not by eye: the image where the Sinkhorn term gains most in PSNR, the median image and the image
where it loses most (`report/figures/ablation_a_pixel_vs_sinkhorn_qualitative.png`).

The paintings in Figure 4.4 look alike: the two losses produce the same composition and the same level of
detail, and differ only in where individual strokes fall. Under this protocol there is therefore no evidence
that the term helps, and the data exclude an average gain above 0.05 dB. That is a statement about this
setting (oil brush, light renderer, 500 strokes, a $5 \times 5$ grid, the weight 0.1 of the reference), not
about the loss in general; the next sections ask whether another weight, another regularization or a
worse initialization changes the picture.

### 4.2.7 The weight and the regularization of the Sinkhorn term

*Weight.* The weight 0.1 of the reference is one point on a scale, so Ablation A was repeated at the weights 1,
10 and 100 on 10 images (every third image of the sorted evaluation set) with three seeds, 150 paintings with
the pixel-only baseline (experiment `2026-10-03_ablation_sinkhorn_weight_sweep`).

**Table 4.8.** Weight sweep: mean over the three seeds of the image average, plus or minus the standard
deviation of the three seed means, and the paired difference in PSNR to the pixel loss per image ($n = 10$).

| Loss | PSNR (dB) | SSIM | PSNR difference, 95 % CI | Images better | Wilcoxon $p$ |
| :--- | ---: | ---: | :--- | ---: | ---: |
| Pixel only | 19.84 $\pm$ 0.06 | 0.556 $\pm$ 0.001 | | | |
| + Sinkhorn, $\beta_{\text{OT}} = 0.1$ | 19.83 $\pm$ 0.01 | 0.557 $\pm$ 0.001 | $-0.011$, $[-0.076, +0.054]$ | 5 of 10 | 0.85 |
| + Sinkhorn, $\beta_{\text{OT}} = 1$ | 19.85 $\pm$ 0.02 | 0.557 $\pm$ 0.001 | $+0.016$, $[-0.087, +0.118]$ | 6 of 10 | 1.0 |
| + Sinkhorn, $\beta_{\text{OT}} = 10$ | 19.84 $\pm$ 0.03 | 0.556 $\pm$ 0.000 | $+0.004$, $[-0.096, +0.105]$ | 4 of 10 | 1.0 |
| + Sinkhorn, $\beta_{\text{OT}} = 100$ | 19.73 $\pm$ 0.02 | 0.552 $\pm$ 0.002 | $-0.104$, $[-0.178, -0.030]$ | 2 of 10 | 0.02 |

From 0.1 to 10 the term changes nothing that the experiment can see, and each interval excludes an effect
larger than 0.12 dB in either direction. At 100 it does harm: the PSNR falls by 0.10 dB and the SSIM by 0.004,
and 8 of the 10 images get worse (Figure 4.5). No weight in this range helps.

**Figure 4.5.** Paired difference to the pixel loss against the Sinkhorn weight, in PSNR (left) and SSIM
(right): the mean with its 95 % confidence interval, and one dot per image
(`report/figures/ablation_sinkhorn_weight_sweep.png`).

*Why the weight of the reference does nothing.* `scripts/loss_gradients.py` measures it directly. It paints three
images with the pixel loss alone and, at the first and the last optimizer step of each of the 20 strokes of a
block (120 probes), takes the gradient of the pixel loss and of the unweighted Sinkhorn cost with respect to the
stroke parameters (experiment `2026-10-03_diagnostic_gradient_ratio`). Table 4.9 gives the median ratio of the
two gradient norms.

**Table 4.9.** Size of the Sinkhorn gradient relative to the pixel-loss gradient, median over 120 probes,
for the unweighted cost; multiply by $\beta_{\text{OT}}$.

| Moment | All strokes placed so far | Newest stroke | Position of the newest stroke |
| :--- | ---: | ---: | ---: |
| First step of a new stroke | 0.0072 | 0.0034 | 0.0032 |
| Last step of a stroke | 0.0087 | 0.0043 | 0.0042 |

The transport gradient is below one per cent of the pixel gradient, and in none of the 120 probes does the ratio
exceed 0.07. At the reference weight of 0.1 the term therefore supplies about 0.03 to 0.09 % of the gradient:
it is practically switched off, which is why Ablation A found no effect, and the finding is not that the term
is harmless and useless but that it is too small to act. The weight has to be about 100 for the transport
gradient to reach a third of the pixel gradient on the newest stroke (0.34) and 0.7 of it over all strokes, and
that is the weight at which Table 4.8 shows harm. The sweep therefore brackets the range in which the term could
help: below 100 it is too small to act, and at 100, where it begins to act, it hurts.

*Regularization.* The entropic regularization $\varepsilon$ sets how far the transport plan spreads the mass
(Section 2.6.5: a standard deviation of about $\sqrt{\varepsilon}$ on the unit square, 1.3 cells of the
$24 \times 24$ grid for 0.003 and 7.6 cells for 0.1). At the reference weight the term is too small to act
(Table 4.9), so this sweep was run at the weight 100, where it is large enough to be modulated: $\varepsilon \in
\{0.003, 0.01, 0.03, 0.1\}$ on 5 images (every sixth image of the sorted evaluation set) with three seeds, 75
paintings (experiment `2026-10-03_ablation_sinkhorn_epsilon_sweep`).

**Table 4.10.** Epsilon sweep at the weight 100: mean over the three seeds of the image average, plus or minus the
standard deviation of the three seed means, and the paired difference in PSNR to the pixel loss per image
($n = 5$; with five images the smallest possible Wilcoxon $p$ is 0.0625).

| Loss | PSNR (dB) | SSIM | PSNR difference, 95 % CI | Images better |
| :--- | ---: | ---: | :--- | ---: |
| Pixel only | 20.04 $\pm$ 0.15 | 0.559 $\pm$ 0.002 | | |
| + Sinkhorn, $\varepsilon = 0.003$ | 20.13 $\pm$ 0.11 | 0.563 $\pm$ 0.000 | $+0.093$, $[-0.015, +0.201]$ | 4 of 5 |
| + Sinkhorn, $\varepsilon = 0.01$ (reference) | 19.90 $\pm$ 0.08 | 0.554 $\pm$ 0.001 | $-0.141$, $[-0.468, +0.186]$ | 1 of 5 |
| + Sinkhorn, $\varepsilon = 0.03$ | 19.56 $\pm$ 0.09 | 0.532 $\pm$ 0.001 | $-0.481$, $[-0.882, -0.079]$ | 0 of 5 |
| + Sinkhorn, $\varepsilon = 0.1$ | 16.57 $\pm$ 0.05 | 0.400 $\pm$ 0.002 | $-3.474$, $[-4.347, -2.600]$ | 0 of 5 |

The harm grows with $\varepsilon$ (Figure 4.6): 0.14 dB at the reference value (an interval that includes
zero), 0.48 dB at 0.03, and 3.5 dB at 0.1, where the PSNR of the neural canvas itself falls from 25.4 to 15.5 dB,
that is, the blurred transport cost dominates and the strokes no longer reproduce the target. At the smallest value,
0.003, the difference is $+0.09$ dB ($[-0.02, +0.20]$, better on 4 of 5 images), the largest positive value of the
two sweeps. It is too small and too uncertain to count as a gain: five images cannot separate it from zero, and
it would come with the same 2.6 times longer optimization. Smaller values of $\varepsilon$ were not tried.

**Figure 4.6.** Paired difference to the pixel loss against $\varepsilon$ at the weight 100, in PSNR (left) and
SSIM (right) (`report/figures/ablation_sinkhorn_epsilon_sweep.png`).

### 4.2.8 The morphological step

Section 2.3.2 explains why the engine dilates the predicted foreground and erodes the predicted opacity with
$3 \times 3$ windows before compositing: the reference rasterizer does the same at inference time, so that
the strokes of the finished painting have no dark halo at their edges, while the neural renderer imitates the
rasterizer in *training* mode, halo included. The reference applies the step but does not report what it
contributes, so Ablation A3 measures it: the pixel loss alone, with and without the step, on 10 of the 30 images
(every third image of the sorted evaluation set) and three seeds, 60 paintings
(experiment `2026-10-03_ablation_a3_morphology`).

**Table 4.11.** Ablation A3: mean over the three seeds of the image average, plus or minus the standard
deviation of the three seed means, and the paired difference per image ($n = 10$).

| Dilate and erode | PSNR (dB) | SSIM | Neural-canvas PSNR (dB) |
| :--- | ---: | ---: | ---: |
| With (default) | 19.84 $\pm$ 0.04 | 0.556 $\pm$ 0.001 | 24.93 $\pm$ 0.12 |
| Without | 17.13 $\pm$ 0.02 | 0.534 $\pm$ 0.002 | 23.18 $\pm$ 0.03 |
| Difference (without minus with) | $-2.707$, 95 % CI $[-3.945, -1.470]$ | $-0.022$, $[-0.068, +0.023]$ | |

Without the step the finished painting loses 2.7 dB of PSNR, more than any effect found in the loss ablations,
and none of the 10 images improves (Wilcoxon $p = 0.002$, the smallest value possible for ten images). The
SSIM difference is small and its interval contains zero, so what is lost is mostly fidelity of colour and
tone and not structure. The neural canvas loses less (1.75 dB) than the finished painting (2.7 dB), which is
consistent with a mismatch between the surrogate and the rasterizer: the optimizer fits strokes whose edges carry
the training-mode halo, and the final rasterization does not have it. The step costs nothing but two
pooling operations per rendered batch, and of the details of the reference painting loop that were tested
here it is the most valuable.

### 4.2.9 Does the transport term make up for a worse start?

The Sinkhorn term exists for strokes that sit where the pixel loss gives them no gradient (Section 2.6.1). The
error-map sampler of Section 4.2.3 starts every stroke on a place where the canvas is wrong, and may already
spare the optimizer that situation, which would explain why the term has nothing left to do in Ablation A.
Ablation A2 tests this. It removes the sampler's guidance (`PainterConfig.init = "uniform"`: the centre of a new
stroke is drawn uniformly over the block, and its colour is still read from the target at that point) and runs
the pixel loss and the pixel loss plus the Sinkhorn term at the weight 100, where the term is large enough to
act (Table 4.9), from both starts, on 10 images (every third image of the sorted evaluation set) with three seeds,
120 paintings (experiment `2026-10-03_ablation_a2_initialization`).

**Table 4.12.** Ablation A2: mean over the three seeds of the image average, plus or minus the standard deviation
of the three seed means, and the paired difference to the pixel loss from the same start ($n = 10$) and to the
pixel loss from the error-map start.

| Start | Loss | PSNR (dB) | SSIM | Difference to the same start with the pixel loss, 95 % CI | Difference to error-map start with the pixel loss, 95 % CI |
| :--- | :--- | ---: | ---: | :--- | :--- |
| Error map | Pixel | 19.81 $\pm$ 0.06 | 0.556 $\pm$ 0.002 | | |
| Error map | Pixel + Sinkhorn 100 | 19.75 $\pm$ 0.06 | 0.550 $\pm$ 0.002 | $-0.056$, $[-0.177, +0.064]$ | $-0.056$, $[-0.177, +0.064]$ |
| Uniform | Pixel | 18.94 $\pm$ 0.06 | 0.533 $\pm$ 0.002 | | $-0.866$, $[-1.249, -0.482]$ |
| Uniform | Pixel + Sinkhorn 100 | 18.68 $\pm$ 0.12 | 0.526 $\pm$ 0.002 | $-0.264$, $[-0.485, -0.044]$ | $-1.130$, $[-1.608, -0.652]$ |

The error-map start is worth 0.87 dB with the pixel loss, and better on every image (Wilcoxon $p = 0.002$). The
hypothesis fails: from the uniform start the transport term does not make up for the poorer placement, it makes the
painting 0.26 dB worse ($[-0.49, -0.04]$, worse on 8 of 10 images, $p = 0.03$; SSIM $-0.007$), and from the error-map
start the change is $-0.06$ dB with an interval that includes zero (SSIM $-0.006$, $[-0.009, -0.003]$). A
probable reason is that a blank canvas leaves the pixel loss with gradient everywhere, because a stroke placed
at random still overlaps regions that are wrong, so the situation the term was designed for hardly arises while
a painting is being built up; this was not tested separately. Whatever the reason, no tested combination of
weight, regularization and start (Tables 4.6, 4.8, 4.10 and 4.12) shows the term improving the finished
painting by a margin that these samples can resolve, which is about 0.1 dB.

### 4.2.10 Full image against fixed grid

The first half of research question RQ2 compares painting the whole image in one block (*full-image mode*,
one $32 \times 32$ canvas for the light renderer, all 500 strokes in it) with the fixed $5 \times 5$ grid (25
blocks of 20 strokes) at the same stroke budget. The oil brush, the pixel loss, the light renderer, 10 images
(every third image of the sorted evaluation set) and three seeds were used, 60 paintings in all (experiment
`2026-10-03_ablation_b_full_vs_grid`). Both modes follow the reference rule of about 500 optimizer steps: the
grid takes 25 steps per stroke, the single block one step per stroke.

**Table 4.13.** Ablation B, first half: mean over the three seeds of the image average, plus or minus the
standard deviation of the three seed means, and the paired difference per image ($n = 10$). The PSNR of the
neural canvas is left out: the canvases have different sizes ($32 \times 32$ against $160 \times 160$), so it
does not compare the two modes.

| Mode | PSNR (dB) | SSIM |
| :--- | ---: | ---: |
| $5 \times 5$ grid, 20 strokes per block | 19.78 $\pm$ 0.03 | 0.556 $\pm$ 0.001 |
| Full image, 500 strokes in one block | 14.08 $\pm$ 0.11 | 0.435 $\pm$ 0.002 |
| Difference (full image minus grid) | $-5.701$, 95 % CI $[-7.015, -4.388]$ | $-0.121$, $[-0.165, -0.078]$ |

The grid is better by 5.7 dB of PSNR and 0.12 of SSIM, and it is better on every one of the ten images (Wilcoxon
$p = 0.002$ for both metrics, the smallest value possible with ten images). The smallest gap, 1.6 dB, is on a
dark photograph with a thin bright subject; the largest, 8.1 dB, is on a pale image with a small saturated
object, where the grid places small strokes on the object and the single block cannot (Figure 4.7).

**Figure 4.7.** Target, full-image painting and $5 \times 5$ grid painting (seed 0) for the images where the
grid gains least, the median image and the image where it gains most
(`report/figures/ablation_b_full_vs_grid_qualitative.png`).

Three properties of full-image mode explain the gap, and the experiment does not separate them. The size of a
stroke is a fraction of its block, so 500 strokes in one block start at 10 to 25 % of the image side (51 to 128
pixels at 512), five times larger than in the grid, and the painting is a coarse mosaic of such strokes. The
optimizer also sees only a $32 \times 32$ image of the target, so detail finer than a few pixels of that image
is not in the objective at all. And with one optimizer step per stroke a new stroke is hardly adjusted after it
is drawn from the error map, against 25 steps in the grid. The comparison is therefore a statement about the
reference's two modes at 500 strokes with the light renderer, not about full-image painting in general: the
full-size renderer, whose $128 \times 128$ canvas would raise the first limit, was not tested. The progressive
mode, which paints at several grid sizes in turn, is the second half of RQ2 (Week 7).

### 4.2.11 Agreement with the original code

The tests of Section 4.2.4 check the parts of the engine one at a time. Whether the parts add up to the same
painter is a question about whole paintings, so `scripts/compare_with_original.py` runs the unmodified
`demo.py` of the 2021 code (fixed grid, pixel loss, the pretrained light oil renderer, a white canvas, 500
strokes on a $5 \times 5$ grid) and this package on 5 images (every sixth image of the sorted evaluation set) with
three repeats each, and scores all 45 finished paintings with the same code (experiment
`2026-10-03_check_oil_fixed_grid`). The original has no seed option in `demo.py`, so its repeats are unseeded.

The first comparison showed this package ahead of the original by 0.30 dB (95 % CI $[+0.06, +0.54]$, better on all
five images). Reading the two codes side by side found one difference in the data path, not in the
optimization: the original shrinks the 512-pixel image to the blocks' resolution (160 pixels for the light
renderer on a $5 \times 5$ grid) with `cv2.resize` at its default, bilinear interpolation, which takes point samples
and so aliases fine texture into the target that the strokes are fitted to, whereas this package averages
(`cv2.INTER_AREA`). A third configuration, `ours_linear`, runs this package with the original's bilinear
shrinking (the option `PainterConfig.resize = "linear"`).

**Table 4.14.** The original code against this package on 5 images and three repeats: mean over the repeats of
the image average, plus or minus the standard deviation of the three repeat means, and the paired difference
to the original per image ($n = 5$).

| Implementation | PSNR (dB) | SSIM | PSNR difference to the original, 95 % CI |
| :--- | ---: | ---: | :--- |
| Original 2021 `demo.py` | 19.75 $\pm$ 0.14 | 0.552 $\pm$ 0.001 | |
| This package, bilinear shrinking as in the original | 19.73 $\pm$ 0.13 | 0.555 $\pm$ 0.001 | $-0.021$, $[-0.215, +0.173]$ |
| This package, area-averaged target (default) | 20.04 $\pm$ 0.04 | 0.558 $\pm$ 0.001 | $+0.298$, $[+0.058, +0.537]$ |

With the original's resizing the two implementations are indistinguishable: the mean difference is $-0.02$ dB
and its interval, $\pm 0.2$ dB, includes zero (Wilcoxon $p = 0.81$), the SSIM differs by $+0.003$
($[-0.002, +0.009]$), and within one image the three repeats of either code differ by 0.1 to 0.16 dB. The
re-implementation therefore reproduces the quality of the reference to within the noise that five images
and three repeats can resolve. The default of this package, which averages when it shrinks, is better than
its bilinear variant by 0.32 dB ($[+0.02, +0.62]$, all five images, $p = 0.06$, the smallest value possible with
five images), and that single choice accounts for the whole gap to the original. All the other experiments in this
section use the default.

### 4.2.12 Summary of the section

**Table 4.15.** What each design choice of the painting loop is worth, as the paired difference in PSNR per image
between the choice and its alternative. All numbers are for the oil brush, the light renderer and 500 strokes.

| Choice | Alternative | Difference in PSNR (dB) | Where |
| :--- | :--- | ---: | :--- |
| $5 \times 5$ grid | full image, one block | $+5.70$ | Table 4.13 |
| $3 \times 3$ dilate and erode of the renderer's output | none | $+2.71$ | Table 4.11 |
| Error-map start of new strokes | uniform start | $+0.87$ | Table 4.12 |
| Target shrunk by area averaging | bilinear shrinking (the original's) | $+0.32$ | Table 4.14 |
| Sinkhorn term, weight 0.1 to 10 | pixel loss alone | between $-0.01$ and $+0.02$ (no effect) | Tables 4.6, 4.8 |
| Sinkhorn term, weight 100 | pixel loss alone | $-0.10$ | Table 4.8 |

The answer to research question RQ1 under this protocol is therefore no: the optimal-transport term does not
improve the finished painting, at the reference weight or at any weight from 0.1 to 100, at the reference
regularization or at three others, or from either of two starts, and every difference found is within about
0.1 dB or negative. At the reference weight the term is practically switched off, since its gradient is about
0.3 % of the pixel loss's. It also costs time: the optimization loop took 2.3 to 2.6 times as long with the
term (6.2 against 16.0 s per painting in Ablation A, and 6.8 against 15.8 s in a later run of three images
on the same machine). Full-image mode took about 5 to 11 times as long as the grid (36 to 74 s against 6.5 s for
one painting of the same image, depending on the state of the machine: its loop launches up to 500 small
operations per step, so it depends on the speed of the processor as much as on the GPU).

The limits of these statements are the following. Everything was measured with one brush, one renderer, one
stroke budget and the fixed grid; the progressive mode of Week 7 and the other three brushes may behave
differently, and so may a style loss for which the transport term might matter more. The weight and
regularization sweeps used 10 and 5 of the 30 images, so their intervals are about $\pm 0.1$ and $\pm 0.2$ dB wide
and cannot show a smaller effect. LPIPS is missing from all tables until the AlexNet weights are downloaded (the
column can be filled afterwards from the saved paintings). The timings are those of one desktop and moved by
a factor of two between runs made hours apart on the same machine, so only ratios measured together mean much;
the same loop on the MacBook is measured in Week 7.
