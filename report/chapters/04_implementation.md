# 4. Implementation

This chapter describes how the method of Chapter 2 is implemented in the `neural_painter`
package. It grows week by week; this version covers the neural renderer (Week 4), the
optimization of strokes against a target image (Week 5), the four brush materials (Week 6) and the automatic pipeline
(Week 7). Unless a
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
Sinkhorn term (Section 4.2.2), Adam and the centred form of RMSprop are options. (Since Week 6 the ranges new strokes start
from come from each brush's YAML file and differ from the reference's; Section 4.3.3. Every experiment of this section used
the reference ranges.)

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
optimization, which is why steps to a threshold are read from it. LPIPS (the `lpips` package with the AlexNet backbone, images scaled to $[-1, 1]$, lower is better) was added
afterwards from the saved paintings, once the 233 MB weights had been downloaded (`scripts/score_lpips.py`); it
is large for every painting here (0.54 to 0.69) because a painting is not meant to look like its photograph, so
only its differences between configurations are of use.

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

| Loss | PSNR (dB) | SSIM | LPIPS | Neural-canvas PSNR (dB) | Steps to 22 dB | Optimization time (s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pixel ($L_1$) | 18.67 $\pm$ 0.03 | 0.490 $\pm$ 0.001 | 0.536 $\pm$ 0.000 | 23.65 $\pm$ 0.05 | 335 (75 of 90) | 6.17 $\pm$ 0.03 |
| Pixel + Sinkhorn ($\beta_{\text{OT}} = 0.1$) | 18.69 $\pm$ 0.01 | 0.490 $\pm$ 0.001 | 0.536 $\pm$ 0.001 | 23.69 $\pm$ 0.05 | 335 (75 of 90) | 15.99 $\pm$ 0.05 |

*The term made no measurable difference to the finished painting.* The two losses give the same mean PSNR to
within 0.02 dB and the same SSIM (Table 4.6). The paired comparison per image (Table 4.7) puts the mean
difference at $+0.011$ dB with a 95 % confidence interval of $[-0.027, +0.049]$, so a gain above 0.05 dB or a
loss above 0.03 dB is excluded; the term wins on 14 of the 30 images and loses on 16, and Wilcoxon's test
gives $p = 0.98$. The SSIM difference is $+0.001$ ($[-0.000, +0.002]$, $p = 0.16$). LPIPS, which asks whether the two paintings
look alike to a trained network, does not separate them either: $-0.000$ ($[-0.002, +0.002]$, $p = 0.90$). For scale, the baseline
paintings of different images range from 13.5 to 25.7 dB, and the same image painted with three seeds
varies by 0.085 dB (median standard deviation over images and losses), so differences of the size seen here are
about the size of the seed noise.

**Table 4.7.** Ablation A, paired differences (Sinkhorn minus pixel), one difference per image with its three
seeds averaged, $n = 30$.

| Metric | Mean difference | 95 % CI | Images where Sinkhorn is better | Wilcoxon $p$ |
| :--- | ---: | :--- | ---: | ---: |
| PSNR (dB) | $+0.011$ | $[-0.027, +0.049]$ | 14 of 30 | 0.98 |
| SSIM | $+0.001$ | $[-0.000, +0.002]$ | 18 of 30 | 0.16 |
| LPIPS (lower is better) | $-0.000$ | $[-0.002, +0.002]$ | 12 of 30 | 0.90 |

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

| Loss | PSNR (dB) | SSIM | LPIPS | PSNR difference, 95 % CI | Images better | Wilcoxon $p$ |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: |
| Pixel only | 19.84 $\pm$ 0.06 | 0.556 $\pm$ 0.001 | 0.563 $\pm$ 0.003 | | | |
| + Sinkhorn, $\beta_{\text{OT}} = 0.1$ | 19.83 $\pm$ 0.01 | 0.557 $\pm$ 0.001 | 0.563 $\pm$ 0.002 | $-0.011$, $[-0.076, +0.054]$ | 5 of 10 | 0.85 |
| + Sinkhorn, $\beta_{\text{OT}} = 1$ | 19.85 $\pm$ 0.02 | 0.557 $\pm$ 0.001 | 0.563 $\pm$ 0.001 | $+0.016$, $[-0.087, +0.118]$ | 6 of 10 | 1.0 |
| + Sinkhorn, $\beta_{\text{OT}} = 10$ | 19.84 $\pm$ 0.03 | 0.556 $\pm$ 0.000 | 0.562 $\pm$ 0.001 | $+0.004$, $[-0.096, +0.105]$ | 4 of 10 | 1.0 |
| + Sinkhorn, $\beta_{\text{OT}} = 100$ | 19.73 $\pm$ 0.02 | 0.552 $\pm$ 0.002 | 0.568 $\pm$ 0.001 | $-0.104$, $[-0.178, -0.030]$ | 2 of 10 | 0.02 |

From 0.1 to 10 the term changes nothing that the experiment can see, and each interval excludes an effect
larger than 0.12 dB in either direction. At 100 it does harm: the PSNR falls by 0.10 dB and the SSIM by 0.004,
and 8 of the 10 images get worse (Figure 4.5). No weight in this range helps. LPIPS agrees: its differences are $+0.001$, $+0.001$ and $-0.001$ for the weights
0.1, 1 and 10 (every interval within $\pm 0.003$) and $+0.005$ at 100 ($[-0.001, +0.010]$, $p = 0.08$), the same
direction as PSNR at 100 but not significant.

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

| Loss | PSNR (dB) | SSIM | LPIPS | PSNR difference, 95 % CI | Images better |
| :--- | ---: | ---: | ---: | :--- | ---: |
| Pixel only | 20.04 $\pm$ 0.15 | 0.559 $\pm$ 0.002 | 0.565 $\pm$ 0.001 | | |
| + Sinkhorn, $\varepsilon = 0.003$ | 20.13 $\pm$ 0.11 | 0.563 $\pm$ 0.000 | 0.569 $\pm$ 0.001 | $+0.093$, $[-0.015, +0.201]$ | 4 of 5 |
| + Sinkhorn, $\varepsilon = 0.01$ (reference) | 19.90 $\pm$ 0.08 | 0.554 $\pm$ 0.001 | 0.574 $\pm$ 0.002 | $-0.141$, $[-0.468, +0.186]$ | 1 of 5 |
| + Sinkhorn, $\varepsilon = 0.03$ | 19.56 $\pm$ 0.09 | 0.532 $\pm$ 0.001 | 0.587 $\pm$ 0.001 | $-0.481$, $[-0.882, -0.079]$ | 0 of 5 |
| + Sinkhorn, $\varepsilon = 0.1$ | 16.57 $\pm$ 0.05 | 0.400 $\pm$ 0.002 | 0.642 $\pm$ 0.002 | $-3.474$, $[-4.347, -2.600]$ | 0 of 5 |

The harm grows with $\varepsilon$ (Figure 4.6): 0.14 dB at the reference value (an interval that includes
zero), 0.48 dB at 0.03, and 3.5 dB at 0.1, where the PSNR of the neural canvas itself falls from 25.4 to 15.5 dB,
that is, the blurred transport cost dominates and the strokes no longer reproduce the target. At the smallest value,
0.003, the difference is $+0.09$ dB ($[-0.02, +0.20]$, better on 4 of 5 images), the largest positive value of the
two sweeps. It is too small and too uncertain to count as a gain: five images cannot separate it from zero, and
it would come with the same 2.6 times longer optimization. LPIPS does not confirm it: at 0.003 it is $+0.004$ ($[-0.001, +0.010]$) and none of the five images improves; it
rises with $\varepsilon$ ($+0.009$, $+0.022$ and $+0.077$ at 0.01, 0.03 and 0.1) as the PSNR falls. Smaller values of
$\varepsilon$ were not tried.

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

| Dilate and erode | PSNR (dB) | SSIM | LPIPS | Neural-canvas PSNR (dB) |
| :--- | ---: | ---: | ---: | ---: |
| With (default) | 19.84 $\pm$ 0.04 | 0.556 $\pm$ 0.001 | 0.565 $\pm$ 0.002 | 24.93 $\pm$ 0.12 |
| Without | 17.13 $\pm$ 0.02 | 0.534 $\pm$ 0.002 | 0.582 $\pm$ 0.001 | 23.18 $\pm$ 0.03 |
| Difference (without minus with) | $-2.707$, 95 % CI $[-3.945, -1.470]$ | $-0.022$, $[-0.068, +0.023]$ | $+0.017$, $[-0.029, +0.062]$ | |

Without the step the finished painting loses 2.7 dB of PSNR, more than any effect found in the loss ablations,
and none of the 10 images improves (Wilcoxon $p = 0.002$, the smallest value possible for ten images). The
SSIM difference is small and its interval contains zero, so what is lost is mostly fidelity of colour and
tone and not structure. LPIPS moves the same way, less clearly: $+0.017$ without the step ($[-0.029, +0.062]$, 8 of
the 10 images worse, $p = 0.23$). The neural canvas loses less (1.75 dB) than the finished painting (2.7 dB), which is
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

| Start | Loss | PSNR (dB) | SSIM | LPIPS | PSNR difference to the same start with the pixel loss, 95 % CI | PSNR difference to the error-map start with the pixel loss, 95 % CI |
| :--- | :--- | ---: | ---: | ---: | :--- | :--- |
| Error map | Pixel | 19.81 $\pm$ 0.06 | 0.556 $\pm$ 0.002 | 0.565 $\pm$ 0.001 | | |
| Error map | Pixel + Sinkhorn 100 | 19.75 $\pm$ 0.06 | 0.550 $\pm$ 0.002 | 0.573 $\pm$ 0.003 | $-0.056$, $[-0.177, +0.064]$ | $-0.056$, $[-0.177, +0.064]$ |
| Uniform | Pixel | 18.94 $\pm$ 0.06 | 0.533 $\pm$ 0.002 | 0.597 $\pm$ 0.001 | | $-0.866$, $[-1.249, -0.482]$ |
| Uniform | Pixel + Sinkhorn 100 | 18.68 $\pm$ 0.12 | 0.526 $\pm$ 0.002 | 0.601 $\pm$ 0.002 | $-0.264$, $[-0.485, -0.044]$ | $-1.130$, $[-1.608, -0.652]$ |

The error-map start is worth 0.87 dB with the pixel loss, and better on every image (Wilcoxon $p = 0.002$). The
hypothesis fails: from the uniform start the transport term does not make up for the poorer placement, it makes the
painting 0.26 dB worse ($[-0.49, -0.04]$, worse on 8 of 10 images, $p = 0.03$; SSIM $-0.007$), and from the error-map
start the change is $-0.06$ dB with an interval that includes zero (SSIM $-0.006$, $[-0.009, -0.003]$). LPIPS gives the same ordering: the error-map start is better by 0.033 with the pixel loss (all
10 images, $p = 0.002$), and the transport term raises LPIPS by 0.008 from the error-map start ($[+0.003, +0.014]$,
$p = 0.004$) and by 0.004 from the uniform start ($[-0.001, +0.008]$, $p = 0.08$). A
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

| Mode | PSNR (dB) | SSIM | LPIPS |
| :--- | ---: | ---: | ---: |
| $5 \times 5$ grid, 20 strokes per block | 19.78 $\pm$ 0.03 | 0.556 $\pm$ 0.001 | 0.565 $\pm$ 0.000 |
| Full image, 500 strokes in one block | 14.08 $\pm$ 0.11 | 0.435 $\pm$ 0.002 | 0.690 $\pm$ 0.002 |
| Difference (full image minus grid) | $-5.701$, 95 % CI $[-7.015, -4.388]$ | $-0.121$, $[-0.165, -0.078]$ | $+0.125$, $[+0.081, +0.170]$ |

The grid is better by 5.7 dB of PSNR and 0.12 of SSIM, and it is better on every one of the ten images (Wilcoxon
$p = 0.002$ for both metrics, the smallest value possible with ten images). LPIPS agrees: 0.565 for the grid
against 0.690 for the full image, a difference of 0.125 ($[+0.081, +0.170]$, all ten images, $p = 0.002$). The smallest gap, 1.6 dB, is on a
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

| Implementation | PSNR (dB) | SSIM | LPIPS | PSNR difference to the original, 95 % CI |
| :--- | ---: | ---: | ---: | :--- |
| Original 2021 `demo.py` | 19.75 $\pm$ 0.14 | 0.552 $\pm$ 0.001 | 0.570 $\pm$ 0.005 | |
| This package, bilinear shrinking as in the original | 19.73 $\pm$ 0.13 | 0.555 $\pm$ 0.001 | 0.567 $\pm$ 0.001 | $-0.021$, $[-0.215, +0.173]$ |
| This package, area-averaged target (default) | 20.04 $\pm$ 0.04 | 0.558 $\pm$ 0.001 | 0.566 $\pm$ 0.004 | $+0.298$, $[+0.058, +0.537]$ |

With the original's resizing the two implementations are indistinguishable: the mean difference is $-0.02$ dB
and its interval, $\pm 0.2$ dB, includes zero (Wilcoxon $p = 0.81$), the SSIM differs by $+0.003$
($[-0.002, +0.009]$), and within one image the three repeats of either code differ by 0.1 to 0.16 dB. The
re-implementation therefore reproduces the quality of the reference to within the noise that five images
and three repeats can resolve. LPIPS agrees: the three configurations are within 0.004 of each other (0.570, 0.567 and
0.566), and its paired differences to the original, $-0.003$ and $-0.004$, have intervals of about $\pm 0.01$. The default of this package, which averages when it shrinks, is better than
its bilinear variant by 0.32 dB ($[+0.02, +0.62]$, all five images, $p = 0.06$, the smallest value possible with
five images), and that single choice accounts for the whole gap to the original in PSNR. LPIPS does not see the difference
between the two filters ($-0.001$, $[-0.007, +0.004]$), so the 0.3 dB is a gain in pixel fidelity that a perceptual
metric does not register. All the other experiments in this
section use the default.

### 4.2.12 Summary of the section

**Table 4.15.** What each design choice of the painting loop is worth, as the paired difference in PSNR per image
between the choice and its alternative. All numbers are for the oil brush, the light renderer and 500 strokes.

| Choice | Alternative | Difference in PSNR (dB) | Difference in LPIPS (lower is better) | Where |
| :--- | :--- | ---: | ---: | :--- |
| $5 \times 5$ grid | full image, one block | $+5.70$ | $-0.125$ | Table 4.13 |
| $3 \times 3$ dilate and erode of the renderer's output | none | $+2.71$ | $-0.017$ (interval includes zero) | Table 4.11 |
| Error-map start of new strokes | uniform start | $+0.87$ | $-0.033$ | Table 4.12 |
| Target shrunk by area averaging | bilinear shrinking (the original's) | $+0.32$ | $-0.001$ (no difference) | Table 4.14 |
| Sinkhorn term, weight 0.1 to 10 | pixel loss alone | between $-0.01$ and $+0.02$ (no effect) | between $-0.001$ and $+0.001$ | Tables 4.6, 4.8 |
| Sinkhorn term, weight 100 | pixel loss alone | $-0.10$ | $+0.005$ (interval includes zero) | Table 4.8 |

The answer to research question RQ1 under this protocol is therefore no: the optimal-transport term does not
improve the finished painting, at the reference weight or at any weight from 0.1 to 100, at the reference
regularization or at three others, or from either of two starts, and every difference found is within about
0.1 dB or negative. LPIPS agrees wherever it can resolve a difference: it finds no effect of the term from 0.1 to 10,
a small and not significant rise at 100, a rise with $\varepsilon$ that parallels the fall in PSNR, and it does not
confirm the small PSNR gain at $\varepsilon = 0.003$ ($+0.004$). At the reference weight the term is practically switched off, since its gradient is about
0.3 % of the pixel loss's. It also costs time: the optimization loop took 2.3 to 2.6 times as long with the
term (6.2 against 16.0 s per painting in Ablation A, and 6.8 against 15.8 s in a later run of three images
on the same machine). Full-image mode took about 5 to 11 times as long as the grid (36 to 74 s against 6.5 s for
one painting of the same image, depending on the state of the machine: its loop launches up to 500 small
operations per step, so it depends on the speed of the processor as much as on the GPU).

The limits of these statements are the following. Everything was measured with one brush, one renderer, one
stroke budget and the fixed grid; the progressive mode of Week 7 and the other three brushes may behave
differently (for the brushes, Section 4.3.5 finds that they do not), and so may a style loss for which the transport term might matter more. The weight and
regularization sweeps used 10 and 5 of the 30 images, so their intervals are about $\pm 0.1$ and $\pm 0.2$ dB wide
and cannot show a smaller effect. The timings are those of one desktop and moved by
a factor of two between runs made hours apart on the same machine, so only ratios measured together mean much
(Week 7 traced this to Windows' power throttling, Section 4.4.7); the same loop on the MacBook is still to be measured.

## 4.3 Brush materials

Section 4.1 checked and scored the pretrained renderers of all four brushes, but its training and cost measurements and
all of Section 4.2 used the oil brush only. This section extends the painter to the other three materials of the
reference, watercolour, marker pen and coloured tape, and asks how each should start its strokes, how the four compare
on the same photographs, whether the conclusion about the Sinkhorn term holds for all of them, and whether the renderer
training of Section 4.1.5 works for every brush (Section 4.3.6). The technical reference for the brushes (files, parameters, how to
load and add one) is `report/notes/render_module.md`.

### 4.3.1 The four materials

Every brush shares the pipeline: a parameter vector in $[0, 1]^d$, a procedural rasterizer that is the ground truth,
a neural renderer that imitates it, and the compositing equation. What differs is summarized in Table 4.16.

**Table 4.16.** The four materials as the rasterizer draws them ($W$ is the canvas side in pixels).

| | Oil paint | Watercolour | Marker pen | Coloured tape |
| :--- | :--- | :--- | :--- | :--- |
| Parameters (shape / colour / opacity) | 12 (5 / 6 / 1) | 15 (8 / 6 / 1) | 12 (8 / 3 / 1) | 9 (5 / 3 / 1) |
| Geometry | textured patch: centre, size, angle | quadratic Bezier | quadratic Bezier | rotated rectangle |
| Footprint | grey-scale brush texture scaled to $(1 + wW) \times (1 + hW)$ px | 100 discs, radius interpolated between the ends | 100 squares turned with the tangent, one width | one rectangle |
| Colour | gradient along the brush | gradient along the curve | one colour | one colour |
| Opacity | always 1 | parameter, uniform | parameter, uniform | parameter, uniform |
| Default canvas | white | white | black | black |

The reference rasterizer has no bleeding, wet edge or paper texture for the watercolour. Such effects could be drawn,
but every pretrained renderer was trained on the reference rasterizer, so an effect would also need a renderer trained
on it; it is left as future work, and "blur and bleed", which the research plan lists among the properties to tune, is
therefore not tuned here. What can be tuned without retraining is how a new stroke starts: the ranges its size and
opacity are drawn from, and, for the Bezier brushes, its initial shape (Sections 4.3.2 and 4.3.3).

*Protocol.* Unless a sentence says otherwise, the experiments of this section follow Section 4.2.5 (fixed $5 \times 5$
grid, 500 strokes, 25 steps per stroke, RMSprop at $0.002$, pixel loss only, the authors' pretrained light renderer of
each brush, seeds 0, 1 and 2) with each brush's default canvas. Settings were tuned on three images chosen to differ in
content (`cube2`: flat coloured regions; `sunflowers`: a saturated still life; `ice-cream-cone`: a photograph with a
smooth background). The best setting of each sweep, whenever it improved PSNR on all three tuning images, was then
checked on ten other images, spread evenly over the remaining 27 of the evaluation set in name order, so that no default
rests on the images it was chosen on. The four brushes are compared on three images not used for tuning (`apple`, the
image of the Week 1 baseline; `iceland`, a landscape; `jay`, a portrait); `apple` and `iceland` are also outside the
held-out set, while `jay` is one of its ten images, so its paintings in Table 4.21 are the held-out runs of the adopted
settings. All Week 6 runs set `--deterministic`: the 27 runs of three marker settings that were painted again in a
second experiment folder (for Figure 4.9) gave identical PSNR to every printed digit.

### 4.3.2 The marker pen could not start

With the reference initialization the marker pen hardly paints. On the three tuning images, 500 strokes leave the
canvas black except for a few specks (PSNR 4.2 dB, Table 4.18, first row; Figure 4.9). The cause is in how a new stroke starts and in what
the renderer learnt from the rasterizer. The reference places all three Bezier points of a new marker stroke on the
pixel the error map chose. The rasterizer does not draw a marker stroke whose end points are less than 4 pixels apart
(in L1 distance, at the canvas size it draws at), so in the training data of the light 32-pixel renderer every short
stroke is blank, and the renderer learnt that. A stroke that renders almost nothing gets almost no gradient with respect
to its shape, so almost none starts to grow.

Table 4.17 measures this directly (`scripts/marker_chord.py`, experiment `2026-10-10_diagnostic_marker_chord`): 512
straight strokes per chord length, in 16 directions, scored by the share whose largest opacity exceeds 0.5. The light
renderer draws none of the strokes up to a chord of 0.05 of the block and all of them from 0.1; the gradient of its
summed opacity with respect to the shape parameters is about 8,600 times smaller for a dot (0.69) than for a chord of
0.1 (5,900). The full 128-pixel renderer, by contrast, draws 83 % of the dots, although its rasterizer draws none at
either size. A plausible reason, not tested, is that at 128 pixels the blank region is much narrower (chords below about
0.03 instead of 0.1), so the network smooths over it; painting with the full renderer was not tried, so the diagnostic
suggests, but does not show, that the problem is specific to the light renderer. The Week 1 baseline (the original
`demo_prog.py` with the light renderer, a progressive grid and centred RMSprop) fails in the same way at its $1 \times 1$
level, where the PSNR of its canvas stays at 6.9 dB for all 500 steps, but there strokes did start to grow from the
$2 \times 2$ level on and the painting ended at 14.0 dB with large unpainted areas; which difference lets them grow there
was not tested.

**Table 4.17.** Share of straight marker strokes of a given chord length (block units) that each renderer draws.

| Chord | Rasterizer, 32 px | Rasterizer, 128 px | Neural, light (32 px) | Neural, full (128 px) |
| ---: | ---: | ---: | ---: | ---: |
| 0 (the reference start) | 0.00 | 0.00 | 0.00 | 0.83 |
| 0.02 | 0.00 | 0.33 | 0.00 | 1.00 |
| 0.05 | 0.00 | 1.00 | 0.00 | 1.00 |
| 0.075 | 0.21 | 1.00 | 0.26 | 1.00 |
| 0.1 | 0.71 | 1.00 | 1.00 | 1.00 |
| 0.125 | 0.98 | 1.00 | 1.00 | 1.00 |
| 0.15 | 1.00 | 1.00 | 1.00 | 1.00 |

*The fix.* A new Bezier stroke can now start as a straight segment of a given length (`chord` in the brush's YAML,
`PainterConfig.init_chord`), in a random direction, with its control point in the middle, centred on the point the error
map chose; near a border the centre is moved inwards, axis by axis, only as far as both ends need to stay inside the
shape range $[0.1, 0.9]$. A length of 0 makes no extra random draw, so it gives exactly the strokes of the Week 5 painter
(`tests/test_brush_init.py`). Table 4.18 sweeps the length on the tuning images. Any length from 0.1 makes the brush
paint (17.1 dB against 4.2 dB). Longer starts are better up to 0.4 and then level off: 0.4, 0.5 and 0.6 are within
0.1 dB of each other (18.08, 18.14 and 18.17 dB) and of the noise between seeds, while 0.3 is 0.28 dB below 0.5 on
every image. The shortest start on that plateau, 0.4, was chosen, since a shorter stroke can reach finer detail. On the
ten held-out images 0.4 beats 0.1 by 1.18 dB (95 % CI 0.95 to 1.42), on every image, and SSIM (+0.042) and LPIPS
($-0.037$) agree on every image too (experiment `2026-10-10_ablation_confirm_start_markerpen_fixed`). The marker pen
therefore starts its strokes with a length of 0.4 by default (`configs/marker_pen.yaml`). The PSNR of the *neural* canvas
peaks earlier, at 0.3 (24.4 dB), and falls for longer starts (23.2 dB at 0.6): long starts help the finished painting
more than the surrogate's view of it.

**Table 4.18.** Marker pen, length of a new stroke: mean over the three seeds of the average over the three tuning
images, plus or minus the standard deviation over seeds (experiment `2026-10-10_ablation_tuning_marker_chord_fixed`; the
row of length 0 is from `2026-10-10_ablation_tuning_marker_chord`, which has the same images and seeds).

| Start length | PSNR (dB) | SSIM | LPIPS | Neural-canvas PSNR (dB) |
| ---: | ---: | ---: | ---: | ---: |
| 0 (the reference) | 4.20 ± 0.00 | 0.008 ± 0.000 | 0.927 ± 0.012 | 4.18 ± 0.00 |
| 0.1 | 17.11 ± 0.16 | 0.617 ± 0.003 | 0.555 ± 0.004 | 23.14 ± 0.14 |
| 0.15 | 17.28 ± 0.05 | 0.623 ± 0.005 | 0.546 ± 0.010 | 23.74 ± 0.07 |
| 0.2 | 17.49 ± 0.16 | 0.628 ± 0.003 | 0.545 ± 0.004 | 24.10 ± 0.17 |
| 0.3 | 17.86 ± 0.13 | 0.633 ± 0.004 | 0.538 ± 0.008 | 24.36 ± 0.13 |
| **0.4** | **18.08 ± 0.14** | **0.631 ± 0.004** | **0.530 ± 0.001** | **24.19 ± 0.24** |
| 0.5 | 18.14 ± 0.17 | 0.625 ± 0.003 | 0.527 ± 0.002 | 23.77 ± 0.13 |
| 0.6 | 18.17 ± 0.14 | 0.624 ± 0.003 | 0.527 ± 0.001 | 23.20 ± 0.17 |

**Figure 4.8.** Left: share of marker strokes drawn against their chord length, for the rasterizer and the neural
renderer at both sizes (Table 4.17). Right: PSNR and LPIPS of the finished marker paintings against the start length
(Table 4.18) (`report/figures/marker_chord.png`).

**Figure 4.9.** Marker paintings of the three tuning images (seed 0) with the reference start (a dot) and with start
lengths of 0.1 and 0.4 (`report/figures/marker_start_qualitative.png`). With the reference start the 500 strokes leave
a few specks on the black canvas. The thin black seams along the block borders in the other two columns are the canvas
showing between blocks: a stroke's end points (or centre) are clamped to $[0.1, 0.9]$ of its block, so a stroke narrower
than that margin cannot reach the border. On the white canvas of the oil brush the same seams are white and much less visible; the progressive mode of
Week 7, whose blocks overlap from one level to the next, hides them (Section 4.4.5).

### 4.3.3 How large a new stroke starts

The reference draws the width and height (oil, tape) or the two radii (watercolour, marker) of a new stroke uniformly
from $[0.1, 0.25]$ for every brush. Table 4.19 compares this with a smaller range, $[0.05, 0.15]$, and a larger one,
$[0.2, 0.4]$, on the tuning images. For all three brushes tested the smaller range is worse or no better and the larger
one has the highest PSNR (by 0.26 to 0.35 dB); for tape and watercolour SSIM and LPIPS agree or do not disagree, while
for the oil brush they are slightly worse, so on three images the oil result was ambiguous.

**Table 4.19.** Start size range on the three tuning images: mean over the three seeds of the image average, plus or
minus the standard deviation over seeds (experiments `2026-10-10_ablation_tuning_size_<brush>`). The marker pen was
tuned for its start length instead (Section 4.3.2).

| Brush | Start sizes | PSNR (dB) | SSIM | LPIPS |
| :--- | :--- | ---: | ---: | ---: |
| Oil paint | $[0.05, 0.15]$ | 18.74 ± 0.04 | 0.607 ± 0.001 | 0.530 ± 0.002 |
| | $[0.1, 0.25]$ (reference) | 19.37 ± 0.09 | 0.606 ± 0.003 | 0.525 ± 0.002 |
| | $[0.2, 0.4]$ | 19.72 ± 0.06 | 0.602 ± 0.000 | 0.535 ± 0.003 |
| Watercolour | $[0.05, 0.15]$ | 18.58 ± 0.08 | 0.638 ± 0.003 | 0.470 ± 0.007 |
| | $[0.1, 0.25]$ (reference) | 18.69 ± 0.04 | 0.638 ± 0.004 | 0.473 ± 0.005 |
| | $[0.2, 0.4]$ | 19.04 ± 0.04 | 0.646 ± 0.002 | 0.474 ± 0.002 |
| Coloured tape | $[0.05, 0.15]$ | 19.50 ± 0.05 | 0.647 ± 0.002 | 0.499 ± 0.004 |
| | $[0.1, 0.25]$ (reference) | 19.74 ± 0.04 | 0.660 ± 0.003 | 0.486 ± 0.002 |
| | $[0.2, 0.4]$ | 20.00 ± 0.04 | 0.674 ± 0.001 | 0.467 ± 0.003 |

A setting that reached the held-out check was adopted when its PSNR gain on the ten held-out images has a 95 %
confidence interval that excludes zero and neither SSIM nor LPIPS is significantly worse; the same rule was applied to
every candidate. Table 4.20 gives the result. The larger range gains 0.35 to 0.39 dB on average and improves every one of
the ten images for all three brushes; SSIM is better for all three; LPIPS is better for tape and watercolour and unchanged
for oil. All three adopt $[0.2, 0.4]$ (`configs/*.yaml`, `sampler.size_range`). With the larger start the neural canvas also
reaches 22 dB earlier (median step 319 to 264 for oil, 362 to 306 for watercolour, 342 to 298 for tape); as Section 4.2.6
notes for such thresholds, this mostly says that fewer strokes are needed to cover a block, not that the optimizer
converges faster.

**Table 4.20.** The larger start range against the reference on ten held-out images: paired difference per image
(seeds averaged), its 95 % confidence interval and the share of images that improve (experiments
`2026-10-10_ablation_confirm_start_<brush>`; for the marker pen `..._confirm_start_markerpen_fixed`; for the
watercolour's opacity `..._confirm_alpha_watercolor`).

| Brush | PSNR (dB) | SSIM | LPIPS (lower is better) |
| :--- | :--- | :--- | :--- |
| Oil paint | +0.39 [+0.14, +0.64], 10 of 10 | +0.005 [+0.001, +0.008], 8 of 10 | $-0.003$ [$-0.009$, +0.004], 7 of 10 |
| Watercolour | +0.35 [+0.23, +0.48], 10 of 10 | +0.013 [+0.000, +0.026], 9 of 10 | $-0.012$ [$-0.021$, $-0.004$], 9 of 10 |
| Coloured tape | +0.36 [+0.14, +0.58], 10 of 10 | +0.022 [+0.015, +0.030], 10 of 10 | $-0.017$ [$-0.027$, $-0.006$], 10 of 10 |
| Marker pen (start length 0.4 against 0.1) | +1.18 [+0.95, +1.42], 10 of 10 | +0.042 [+0.020, +0.063], 10 of 10 | $-0.037$ [$-0.044$, $-0.029$], 10 of 10 |
| Watercolour (start opacity $[0.5, 0.7]$ against $[0.98, 1]$) | +0.30 [+0.20, +0.40], 10 of 10 | +0.006 [+0.002, +0.010], 9 of 10 | $-0.009$ [$-0.019$, +0.002], 9 of 10 |

*Starting opacity.* The three translucent brushes were also tuned for the range their starting opacity is drawn from
(experiments `2026-10-10_ablation_tuning_alpha_<brush>`). For the marker pen and tape neither a nearly opaque start
($[0.98, 1]$) nor a translucent one ($[0.5, 0.7]$) differs from the reference $[0.8, 0.98]$ by more than the noise: the
PSNR differences lie between $-0.07$ and $+0.20$ dB with intervals that include zero, and neither SSIM nor LPIPS changes
significantly. The marker pen's translucent start, which had improved PSNR on all three tuning images (+0.20 dB, interval
$-0.18$ to $+0.59$), was checked on the held-out images and rejected: $+0.09$ dB (95 % CI $-0.21$ to $+0.39$) with a worse
LPIPS on 9 of the 10 images ($+0.013$, Wilcoxon $p = 0.049$; experiment `..._confirm_alpha_markerpen`). The
optimizer adjusts the opacity of every stroke from its first step, so it is not surprising that where it starts matters
little. The watercolour, which the reference starts almost opaque ($[0.98, 1]$), is the exception: it improves when it
starts translucent, by 0.11 dB at $[0.8, 0.98]$ and 0.36 dB at $[0.5, 0.7]$, with SSIM and LPIPS better on all three
tuning images at $[0.5, 0.7]$ (at $[0.8, 0.98]$ LPIPS improves on two of the three), while lower ranges are no better than $[0.5, 0.7]$ ($+0.01$ dB at $[0.3, 0.5]$, $+0.05$ dB at $[0.1, 0.3]$,
experiment `..._tuning_alpha_watercolor_low`). On the held-out images $[0.5, 0.7]$ gains 0.30 dB on average and improves every image, with
SSIM better and LPIPS not worse (last row of Table 4.20), and it is the watercolour's default. One reading, not tested
further, is that a translucent first stroke mixes with the canvas instead of covering it, so the colours of a block are
built up in layers, which suits the soft discs of this brush.

These defaults change the oil painter of Section 4.2, whose experiments all used the reference range: they are
reproduced by `init_size_range=0.1:0.25`. The comparison with the 2021 code, `scripts/compare_with_original.py`, keeps the
reference range in its `ours_linear` configuration; its `ours` configuration now uses $[0.2, 0.4]$, so a rerun would not
reproduce the default row of Table 4.14.

### 4.3.4 The four brushes on the same images

Table 4.21 and Figure 4.10 paint the three comparison images with each brush and its tuned defaults.

**Table 4.21.** The four brushes on `apple`, `iceland` and `jay`: mean over the three seeds of the average over the
three images, plus or minus the standard deviation over seeds (experiment `2026-10-10_brushes_brush_comparison_final`,
which summarizes the four folders `2026-10-10_ablation_brushes_<brush>`, with `..._markerpen_fixed` and
`..._watercolor_final` for the marker pen and the watercolour).

| Brush | PSNR (dB) | SSIM | LPIPS | PSNR, apple / iceland / jay |
| :--- | ---: | ---: | ---: | :--- |
| Oil paint | 20.86 ± 0.06 | 0.505 ± 0.002 | 0.548 ± 0.006 | 20.70 / 20.40 / 21.48 |
| Watercolour | 20.36 ± 0.10 | 0.535 ± 0.002 | 0.517 ± 0.001 | 20.16 / 19.73 / 21.18 |
| Marker pen | 19.41 ± 0.27 | 0.539 ± 0.002 | 0.565 ± 0.003 | 19.14 / 18.55 / 20.53 |
| Coloured tape | 21.53 ± 0.10 | 0.581 ± 0.002 | 0.504 ± 0.003 | 21.01 / 20.83 / 22.77 |

**Figure 4.10.** The three comparison images painted with each brush (seed 0, 500 strokes)
(`report/figures/brush_comparison.png`).

The order by PSNR is the same on all three images: tape, oil, watercolour, marker pen. The two metrics of structure
and perception put tape first as well (SSIM 0.581, LPIPS 0.504), but they rank the other three differently from PSNR:
watercolour has a better LPIPS than oil (0.517 against 0.548) at a lower PSNR, and the marker pen, last by PSNR and LPIPS,
has the second-best SSIM. A likely reason, not tested here, is that tape is a solid rectangle whose five shape parameters map almost directly
onto the area it covers, which suits a fixed grid of small square blocks, while the textured oil stroke and the gradients
of the Bezier brushes spend part of their parameters on detail the pixel loss rewards less. The marker pen, like the tape,
keeps one colour along a stroke, and its width is fixed along the curve. These differences range from 0.5 dB (oil against
watercolour) to 2.1 dB (tape against marker pen) and describe the brushes under one protocol (500 strokes on a fixed grid); they say which material fits the photograph
more closely, not which painting looks better, a question for the user study of Week 9.

### 4.3.5 Does the Sinkhorn term help any brush?

Section 4.2.6 found that the optimal-transport term does not improve oil paintings. The reference paper motivates the
term with strokes that do not overlap their target, which could matter more for thin Bezier strokes or for flat tape
than for the broad oil brush, so Ablation A was repeated for the other three brushes: pixel loss alone against pixel
loss plus the Sinkhorn term at the reference weight 0.1, each brush with its Week 6 defaults, on the same ten images as
the weight sweep of Section 4.2.7 (every third image of the evaluation set) and three seeds (60 paintings per brush).
Table 4.22 sets them next to the oil result of Section 4.2.6.

**Table 4.22.** Ablation A for every brush: mean PSNR of the pixel-only and the pixel + Sinkhorn paintings (mean over
seeds of the image average), the paired difference per image with its 95 % confidence interval and the share of images
where the term wins, the paired difference in LPIPS, and the ratio of the optimization times (experiments
`2026-10-03_ablation_a_pixel_vs_sinkhorn` for oil, `2026-10-10_ablation_ablation_a_<brush>` for the others, with
`..._watercolor_final` for the watercolour).

| Brush | Images | Pixel loss (dB) | + Sinkhorn (dB) | Difference in PSNR (dB) | Difference in LPIPS | Time |
| :--- | ---: | ---: | ---: | :--- | :--- | ---: |
| Oil paint (Section 4.2.6) | 30 | 18.67 | 18.69 | +0.011 [$-0.027$, +0.049], 47 % | $-0.000$ [$-0.002$, +0.002] | 2.6 x |
| Watercolour | 10 | 19.59 | 19.63 | +0.038 [$-0.055$, +0.131], 60 % | $-0.000$ [$-0.004$, +0.004] | 2.7 x |
| Marker pen | 10 | 18.82 | 18.80 | $-0.016$ [$-0.121$, +0.088], 40 % | +0.002 [$-0.001$, +0.006] | 2.6 x |
| Coloured tape | 10 | 20.88 | 20.92 | +0.047 [$-0.001$, +0.094], 80 % | +0.002 [$-0.003$, +0.008] | 2.6 x |

No brush gains. For watercolour and the marker pen the difference is zero within about 0.1 dB. For tape the
term wins on 8 of the 10 images by 0.047 dB on average, but the interval just reaches zero (Wilcoxon $p = 0.08$) and LPIPS
moves, insignificantly, the other way; a gain of a few hundredths of a decibel at 2.6 times the cost would not be worth
having even if it were real. At the reference weight, the answer to RQ1 found for the oil brush therefore holds for all
four materials under this protocol; the other weights, regularizations and starts of Sections 4.2.7 to 4.2.9 were tested
with the oil brush only. The term is not expected to rescue the marker pen's reference start either (this was not run: the
marker rows above use the 0.4 start): a stroke that renders almost nothing barely changes either loss, so both gradients
are close to zero, and for the oil brush at the weight 0.1 the transport gradient is below 0.1 % of the pixel gradient in
any case (Table 4.9). That is why the fix of Section 4.3.2 changes the start and not the loss. The times of this table were
measured while renderers were training on the same machine, so only the ratio, measured on interleaved runs, is
meaningful; at 2.6 to 2.7 it is close to the 2.6 of Section 4.2.6 (2.3 to 2.6 over the Week 5 runs, Section 4.2.12).

### 4.3.6 Renderers trained for every brush

The light renderers of watercolour, marker pen and tape were trained with the recipe of Section 4.1.5 (100 epochs of
50,000 strokes, Adam at $2 \times 10^{-4}$ divided by 10 every 25 epochs, $100 \cdot \mathrm{MSE}$), as the oil renderer was in
Week 4, and scored on the same 1,000 held-out strokes as the pretrained ones (`scripts/renderer_fidelity.py`, experiment
`2026-10-10_fidelity_week6`; training runs `2026-10-10_train_<brush>_light_100ep`). Table 4.23 gives the result.

**Table 4.23.** The retrained light renderers against the authors' (same 1,000 held-out strokes per brush). "Epochs" is
the length of the schedule; the checkpoint used is the best one on the validation strokes, which are these same 1,000
strokes (the last-epoch checkpoint scores less than 0.01 dB lower in every case).

| Brush | Original: PSNR mean (SSIM fg / $\alpha$) | Retrained: PSNR mean (SSIM fg / $\alpha$) | Difference | Epochs | Training time |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Oil paint (Week 4) | 24.38 (0.944 / 0.920) | 24.16 (0.944 / 0.929) | $-0.22$ | 100 | 51 min |
| Watercolour | 25.76 (0.925 / 0.917) | 25.31 (0.923 / 0.934) | $-0.45$ | 100 | 132 min |
| Marker pen | 24.59 (0.914 / 0.903) | 23.95 (0.910 / 0.913) | $-0.64$ | 100 | 197 min |
| Coloured tape | 28.33 (0.959 / 0.957) | 27.53 (0.957 / 0.953) | $-0.80$ | 100 | 14 min |
| Coloured tape, reference schedule | 28.33 (0.959 / 0.957) | 28.52 (0.965 / 0.964) | $+0.19$ | 400 | 50 min |

Every retrained renderer is within the 1 dB target of the research plan, and the opacity SSIM is higher than the
original's for all of them except the 100-epoch tape renderer. The cost differs by a factor of 14 between brushes because training is limited by the
rasterizer on the CPU, not by the GPU: a tape stroke is one polygon (about 6,000 strokes per second), an oil stroke one
texture warp (1,600), and the Bezier brushes draw 100 discs or squares per stroke from a Python loop (about 630 for the
watercolour, 420 for the marker pen).

*For tape, the gap is the schedule.* The tape renderer's learning curve (Figure 4.11) was still rising steeply when the learning rate
was first divided by 10 at epoch 25, and flat afterwards. Because tape trains in minutes, it was also trained with the
reference's schedule (400 epochs, drops every 100; run `2026-10-10_train_rectangle_light_400ep`): it reaches 28.32 dB just
after the first drop, at epoch 102, and 28.52 dB at the end, 0.19 dB *above* the authors' renderer. So, for tape, the
training code reproduces the reference renderer once given the same schedule, and the 0.80 dB deficit of its 100-epoch run
is what a quarter of the training costs; that the same holds for the other three brushes is plausible but untested. The
test would take about 3.4 (oil), 9 (watercolour) and 13 hours (marker pen) and was not run.

**Figure 4.11.** Learning curves of the retrained light renderers: watercolour, marker pen, tape with 100 epochs and tape
with the reference's 400 (`report/figures/renderer_training_curve_<brush>_light.png` and
`renderer_training_curve_rectangle_light_400ep.png`). Dotted lines mark the learning-rate drops, the dashed line the
authors' renderer.

*Painting with the retrained renderers.* A renderer is only a means to paint, so each retrained renderer was used to paint
the ten held-out images with the brush's defaults and compared, image by image, with the same images painted by the
authors' renderer (`scripts/paired_folders.py`; experiments `2026-10-10_ablation_heldout_ours_<brush>` and, for the 400-epoch tape renderer,
`2026-10-10_ablation_heldout_ours_rectangle_400ep`, against the held-out runs of Section 4.3.3). Table 4.24 gives the result.

**Table 4.24.** Painting with the retrained renderer instead of the authors': paired difference per image (seeds averaged)
on the ten held-out images, 95 % confidence interval and the share of images that improve.

| Renderer | PSNR (dB) | SSIM | LPIPS (lower is better) | Neural-canvas PSNR (dB) |
| :--- | :--- | :--- | :--- | :--- |
| Oil paint | $-0.15$ [$-0.29$, $-0.01$], 2 of 10 | $-0.026$ [$-0.044$, $-0.008$], 1 of 10 | $-0.009$ [$-0.023$, +0.004], 7 of 10 | $-0.04$ |
| Watercolour | +1.21 [+0.04, +2.39], 9 of 10 (median +0.78) | +0.023 [$-0.006$, +0.051], 8 of 10 | $-0.034$ [$-0.058$, $-0.010$], 10 of 10 | +1.57 |
| Marker pen | +1.20 [$-0.04$, +2.45], 10 of 10 (median +0.71) | +0.026 [+0.011, +0.042], 10 of 10 | $-0.065$ [$-0.104$, $-0.026$], 10 of 10 | +1.04 |
| Coloured tape | +0.02 [$-0.34$, +0.38], 3 of 10 | $-0.005$ [$-0.021$, +0.012], 4 of 10 | $-0.024$ [$-0.046$, $-0.001$], 9 of 10 | +0.54 |
| Coloured tape, 400 epochs | +0.28 [$-0.19$, +0.76], 8 of 10 | +0.013 [+0.005, +0.020], 10 of 10 | $-0.032$ [$-0.052$, $-0.013$], 10 of 10 | +0.76 |

Only the oil renderer paints in line with its fidelity: 0.22 dB less faithful, it paints 0.15 dB worse (and its SSIM is
lower on 9 of the 10 images). The 100-epoch tape renderer, the least faithful of all ($-0.80$ dB), is even in PSNR and better
in LPIPS and in neural-canvas PSNR, and the 400-epoch one is better in SSIM and LPIPS on every image. The two Bezier brushes
depart from their fidelity most:
their retrained renderers are 0.45 and 0.64 dB *less* faithful on random strokes, yet paint 1.2 dB better on average,
better on 9 and 10 of the 10 images, with LPIPS better on every image. The mean is inflated by one image, the balloon on a
flat pale background, which gains 5.8 (watercolour) and 6.0 dB (marker pen); on the other nine images the gains are $-0.1$
to $+1.2$ dB (watercolour) and $+0.1$ to $+1.6$ dB (marker pen), with medians of 0.78 and 0.71 dB over all ten. Figure 4.12 shows the effect: with the authors' watercolour
renderer the optimized strokes leave dark rings on the flat background once they are rasterized, with ours far fewer. The
neural canvas improves as much as the finished painting (+1.57 and +1.04 dB), so the retrained renderers do not only agree
better with the rasterizer; the optimizer also reaches a better canvas through them.

Why was not established. Fidelity is measured on strokes drawn uniformly from the whole parameter space, while the painter
uses a narrow part of it (strokes of a given size and opacity, overlapping on a smooth canvas); a renderer can be more
faithful on average and less faithful, or less smooth, where the painter works. The retrained renderers also differ from
the authors' in their output layer (a sigmoid that bounds colour and mask to $[0, 1]$, Table 4.1), which may give better
behaved gradients; the oil renderer, which has the same layer and paints no better, argues against that being the whole
story. A study of the renderers' error on the strokes the painter actually produces would answer it. In practice the
result means that for the watercolour and the marker pen the retrained light renderers are the better choice for painting;
they are not yet the default (the experiments of this chapter all use the authors' renderers, so that they can be compared
with the 2021 code), and `run_ablation.py --renderer checkpoints/<brush>_light_100ep/best.pt` selects them.

**Figure 4.12.** Watercolour paintings of three held-out images with the authors' renderer and with the retrained one
(seed 0), for the images where the retrained renderer gains most, the median and the least (rule as in Figure 4.4)
(`report/figures/watercolor_renderers.png`, experiment `2026-10-10_brushes_watercolor_renderers`).

### 4.3.7 Summary of the section

**Table 4.25.** What the Week 6 choices are worth: paired PSNR difference per image (Sections 4.3.2 to 4.3.6), on the ten
held-out images unless the row says otherwise (the first row is on the tuning images, the Sinkhorn row on the ten images of
Table 4.22), with the brushes concerned.

| Choice | Alternative | Difference in PSNR (dB) | Where |
| :--- | :--- | ---: | :--- |
| Marker stroke starting as a segment of length 0.4 | the reference's dot | $+13.9$ (tuning images; the dot paints almost nothing) | Table 4.18 |
| Marker start length 0.4 | 0.1 | $+1.18$ | Table 4.20 |
| Start sizes $[0.2, 0.4]$ | $[0.1, 0.25]$ | $+0.35$ to $+0.39$ (oil, watercolour, tape) | Table 4.20 |
| Watercolour start opacity $[0.5, 0.7]$ | $[0.98, 1]$ | $+0.30$ | Table 4.20 |
| Sinkhorn term, weight 0.1 | pixel loss alone | $-0.02$ to $+0.05$, no effect (all brushes) | Table 4.22 |
| Retrained light renderer | the authors' | $+1.2$ (watercolour, marker; median $+0.7$ to $+0.8$), $+0.0$ to $+0.3$ (tape), $-0.15$ (oil) | Table 4.24 |

With its tuned defaults each brush paints the comparison images at 19.4 (marker pen) to 21.5 dB (tape) in 5.8 s of
optimization on an idle desktop, the same for every brush because the light renderers have the same size, plus 0.9 (tape)
to 1.4 s (marker pen) to rasterize the final 512-pixel painting (experiments `2026-10-10_ablation_timing_clean_<brush>`,
three images, seed 0). The limits are those of Section 4.2.12: one stroke budget, the fixed grid and the light renderers;
the tuning used three images and the checks ten, so effects below about 0.1 to 0.2 dB cannot be resolved; and every
choice was tuned with the authors' renderers, so the defaults may not be the best ones for the retrained renderers of
Section 4.3.6.

## 4.4 The automatic pipeline

Weeks 4 to 6 built the parts of the painter and measured them with the fixed grid of the 2021 `demo.py`. This section joins
the parts into one program that turns a photograph into a painting with one command (Section 4.4.1), adds the coarse-to-fine
mode of the reference's `demo_prog.py` (Section 4.4.2) and two options of its own (Section 4.4.3), measures which grid mode
paints best (the second half of RQ2, Sections 4.4.4 and 4.4.5), how quality and time grow with the number of strokes (RQ3,
Section 4.4.6), and what the program costs in time and memory on each device (RQ4, Section 4.4.7), and tests it end to end on
photographs of every shape and format (Section 4.4.8).

### 4.4.1 From photograph to painting in one command

`scripts/run_paint.py` takes a photograph and writes a painting:

```
venv/Scripts/python.exe scripts/run_paint.py photo.jpg --brush oil --strokes 500 --grid progressive
```

Figure 4.13 shows what happens in between. The photograph is loaded (JPG, PNG, BMP, WebP or TIFF; EXIF rotation applied)
and fitted to a $512 \times 512$ square by `image_io.preprocess` (Section 5.1). Three fits are offered: `stretch`, the
reference's behaviour, distorts the photograph to a square and stretches the painting back to the photograph's aspect
ratio when it is saved; `crop` and `pad` keep the strokes undistorted and give a square painting. The fitted image is then
painted by `pipeline/progressive_painter.paint_image` at one or several grid sizes in turn: for each level the target and the
canvas so far are cut into $m \times m$ blocks, every block gets its strokes, the strokes of all blocks are optimized as one
batch by the engine of Section 4.2.1, mapped to image coordinates (2.13) and drawn by the procedural rasterizer on top of the
painting, which becomes the starting canvas of the next level. While it runs, one line on the terminal shows the level, the
strokes done out of the total, the loss, the PSNR of the neural canvas and the elapsed time; the same information, with the
neural canvas itself for a live preview, reaches any other caller as a `Progress` event (the web interface of Week 8 will be
such a caller). At the end a new folder receives the painting (`final.png`), the fitted photograph it is scored against
(`input.png`), the strokes in the reference's `.npz` layout, every setting with the device and the commit (`config.yaml`) and
one row of results: PSNR and SSIM, optimization and rasterization time, number of strokes and optimizer steps, and the peak
working memory (`results.csv`). Folders are never reused.

**Figure 4.13.** The end-to-end pipeline (`report/figures/pipeline_end_to_end.png`).

The peak memory is measured by `core.device.PeakMemory` in the same way on every backend as far as the backends allow: on
`cuda` it is the allocator's exact peak of tensor memory (`torch.cuda.max_memory_allocated` after a reset); `mps` keeps no
peak counter, so a thread samples its allocated tensor memory every 5 ms; on the CPU the same thread samples the process's
resident memory, which counts every allocation of the process and not only tensors. All three report the rise above the
level at the start of the painting, so the renderer's weights and the CUDA context, loaded before, are not counted.

### 4.4.2 Coarse-to-fine painting

The reference's `demo_prog.py` paints at the grids $1 \times 1, 2 \times 2, \dots, M \times M$ in turn ($M = 5$ by default).
Every block of every level gets the same number of strokes, $\lfloor 500 / 55 \rfloor = 9$ for 500 strokes and $M = 5$
($55 = 1 + 4 + 9 + 16 + 25$ blocks), so 495 strokes are painted, and each stroke gets $\lfloor 500 / 9 \rfloor = 55$
optimizer steps: about 500 steps per level and 2,475 in all, five times as many as the fixed grid's 500. The first strokes
are therefore as large as a fifth to two fifths of the image (the start sizes of Section 4.3.3 are fractions of the block)
and lay in the background colours; each finer level paints over the previous one in smaller strokes, starting from the
canvas that the earlier strokes, rasterized with the real brush, produce. After every level the reference re-renders all
strokes so far from a blank canvas; `paint_image` draws each level's strokes on top of the canvas it already has, which is
the same sequence of compositing operations and so the same image (checked to the bit in the tests), with one pass over the
strokes instead of one per level. Two further differences are in the data path: the target of every level is shrunk from
the source image (the reference shrinks it from a copy already shrunk to $M S$ pixels), by area averaging unless
`resize = "linear"` asks for the reference's bilinear filter (Section 4.2.11), and so is the canvas handed to the next
level. The progressive demo uses the centred form of RMSprop where `demo.py` uses the plain one; `PainterConfig.centered`
selects either, and Section 4.4.4 measures both.

`paint_image` paints the reference's other two modes as single levels: `fixed` is the one level $M \times M$ and gives the
same strokes as `paint_fixed_grid` of Section 4.2.1 (the same random draws in the same order; checked to the bit in the
tests), and `full` is the one level $1 \times 1$. The three modes are thus one code path, which is what makes their
comparison clean.

*Agreement with the reference.* As in Section 4.2.11, `scripts/compare_with_original.py --mode progressive` runs the
reference's `demo_prog.py` (oil brush, light renderer, white canvas, 500 strokes, $M = 5$, pixel loss) and this package on
the five images that are every sixth of the evaluation set, three repeats each, and scores all 45 paintings with the same
code (experiment `2026-10-15_check_oil_progressive`). Our patch of the reference added a `--seed` option to `demo_prog.py`,
so its repeats are seeded with the repeat number; that fixes its random draws, but they are not ours. Like ours, the
reference process is opted out of Windows' power throttling (Section 4.4.7), so that the two run under the same conditions.
The configuration `ours_linear` is this package with the reference's settings: bilinear shrinking, the start sizes
$[0.1, 0.25]$ and centred RMSprop.

**Table 4.26.** The reference's `demo_prog.py` against this package on 5 images and three repeats: mean over the repeats of the
image average, plus or minus the standard deviation of the three repeat means, and the paired difference to the reference
per image ($n = 5$).

| Implementation | PSNR (dB) | SSIM | LPIPS | PSNR difference to the reference, 95 % CI |
| :--- | ---: | ---: | ---: | :--- |
| Reference `demo_prog.py` | 20.30 $\pm$ 0.09 | 0.578 $\pm$ 0.002 | 0.553 $\pm$ 0.001 | |
| This package, the reference's settings | 20.25 $\pm$ 0.04 | 0.577 $\pm$ 0.001 | 0.551 $\pm$ 0.004 | $-0.05$, $[-0.15, +0.05]$ |
| This package, its defaults | 20.95 $\pm$ 0.05 | 0.584 $\pm$ 0.003 | 0.549 $\pm$ 0.002 | $+0.65$, $[+0.21, +1.08]$, 5 of 5 |

With the reference's settings the two implementations are indistinguishable in all three metrics (the SSIM and LPIPS
differences are below 0.003), so the progressive mode reproduces the reference as the fixed grid did. With the package's
defaults (area-averaged targets, the Week 6 start sizes, plain RMSprop at a constant 0.002, the settings of Table 4.29) the
progressive painting is better by 0.65 dB on every image, and by 0.70 dB than the package run with the reference's
settings. That difference combines the shrinking filter (0.32 dB with the fixed grid, Table 4.14), the start sizes (0.39 dB
with the fixed grid, Table 4.20) and the form of RMSprop (0.01 dB, Table 4.29), which this experiment does not separate.
LPIPS sees no difference ($-0.003$, $[-0.017, +0.010]$).

### 4.4.3 Two options: strokes where they are needed, and a learning-rate schedule

The plan for this week names two modernizations of the loop, dynamic stroke allocation and learning-rate scheduling. Both
are options of `PainterConfig` and are off by default until the measurements below say otherwise.

*Stroke allocation* (`allocation = "error"`). The reference gives every block the same number of strokes, so a block of
flat sky that the canvas colour already matches gets as many strokes as a block of foliage, and the equal split drops the
remainder of the budget (5 of 500 strokes for $M = 5$; 45 of 100). With the option, each level first gets its share of the
budget in proportion to its number of blocks, rounded so that the shares add up to exactly the budget
(`core.grid.level_budgets`: 9, 36, 82, 146 and 227 strokes for 500 and $M = 5$), and the share is then split among the blocks
in proportion to their mean absolute error $\frac{1}{3S^2}\sum |I - C|$ on the canvas the level starts from, by D'Hondt's
method (strokes are handed out one at a time to the block with the largest $e_b / (n_b + 1)$), with at most twice the level's
mean per block so that one block cannot take the whole share. Blocks with different counts are still optimized as one batch:
each has as many slots as the largest count, and the slots a block does not use are made transparent by multiplying their
opacity by zero, so they neither paint nor receive a gradient. The number of steps per stroke follows the largest count,
$\lfloor 500 / n_{\max} \rfloor$, which keeps about 500 steps per level as in the reference; the price is that a block's
strokes get fewer steps each when another block gets many.

*Learning-rate schedule* (`lr_schedule = "cosine"`). The reference keeps the learning rate at 0.002 throughout. With the
option the rate falls along half a cosine from its peak to zero over the $T$ steps that follow each new stroke and starts
again at the next one, a warm restart at every stroke [Loshchilov and Hutter, SGDR, 2017]: the strokes move far just after a
new one arrives and settle before the error map is read for the next. With the same peak the mean rate is halved, so the
schedule is tried both at the reference's peak and at twice it (0.004, the same mean rate as the reference).

Both options were tuned with the protocol of Section 4.3.3: three tuning images (cube2, sunflowers, ice-cream-cone) and three
seeds first, then every setting that improved PSNR on all three images was checked on the ten held-out images and adopted if
the 95 % interval of its PSNR gain excludes zero and neither SSIM nor LPIPS is significantly worse. The tuning also varied the
number of steps per stroke of the progressive mode, which Section 4.4.4 shows to matter (experiments
`2026-10-15_ablation_tuning_pipeline_oil` and `..._tuning_pipeline_iters`, oil, 126 paintings).

**Table 4.27.** Tuning on three images (seeds averaged per image): PSNR difference to the progressive mode with the reference's
settings (20.01 dB, 2,475 steps), the number of images improved, and the optimization time of runs made two at a time.

| Setting | PSNR difference (dB) | Images improved | LPIPS difference | Time (s) | Peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Reference (constant rate 0.002, equal allocation, 55 steps per stroke) | | | | 21.4 | 0.18 |
| Allocation by error | $+0.21$ | 3 of 3 | $-0.027$ | 28.8 | 0.32 |
| Cosine schedule, peak 0.002 | $-0.11$ | 1 of 3 | $+0.001$ | 21.2 | 0.17 |
| Cosine schedule, peak 0.004 | $+0.12$ | 3 of 3 | $-0.005$ | 20.7 | 0.17 |
| Allocation by error and cosine, peak 0.004 | $+0.21$ | 3 of 3 | $-0.027$ | 30.2 | 0.33 |
| 22 steps per stroke (990 steps) | $-0.08$ | 0 of 3 | $+0.009$ | 8.7 | 0.17 |
| 11 steps per stroke (495 steps) | $-0.28$ | 0 of 3 | $+0.015$ | 4.4 | 0.17 |
| 5 steps per stroke (225 steps) | $-0.49$ | 0 of 3 | $+0.021$ | 2.6 | 0.17 |
| 11 steps per stroke with allocation by error | $+0.11$ | 3 of 3 | $-0.014$ | 10.9 | 0.33 |
| Fixed grid (reference) | $-0.29$ | 0 of 3 | $+0.001$ | 14.3 | 0.38 |

The cosine schedule helps only at the doubled peak; at the reference's peak, which halves the mean rate, it loses 0.11 dB.
The number of steps per stroke trades quality for time smoothly: the reference's 55 steps per stroke are the best, 22 cost
0.08 dB in 40 % of the time, and 11 paint as well as the fixed grid in a third of its time. Allocation by error helps in every
combination, most of all in LPIPS, and costs about a third more time and almost twice the memory, because a block may hold up to
twice the mean number of strokes and the batch is padded to the largest.

**Table 4.28.** The candidates on the ten held-out images (`2026-10-15_ablation_confirm_pipeline_oil`, 150 paintings): paired
difference per image to the progressive mode with the reference's settings (20.56 dB), with 95 % intervals.

| Setting | PSNR (dB) | Images won | SSIM | LPIPS | Time (s) |
| :--- | :--- | ---: | :--- | :--- | ---: |
| Reference settings | 20.56 | | 0.576 | 0.521 | 21.1 |
| Cosine schedule, peak 0.004 | $+0.070$ $[+0.026, +0.114]$ | 8 of 10 | $+0.005$ $[+0.002, +0.007]$ | $-0.004$ $[-0.008, +0.001]$ | 20.8 |
| Allocation by error | $+0.082$ $[-0.004, +0.167]$ | 8 of 10 | $+0.004$ $[-0.000, +0.008]$ | $-0.007$ $[-0.014, +0.000]$ | 28.2 |
| Both | $+0.106$ $[+0.034, +0.178]$ | 9 of 10 | $+0.006$ $[+0.002, +0.009]$ | $-0.010$ $[-0.015, -0.005]$ | 29.1 |
| 11 steps per stroke with allocation by error | $-0.007$ $[-0.085, +0.070]$ | 4 of 10 | $-0.006$ $[-0.011, -0.002]$ | $+0.008$ $[+0.003, +0.013]$ | 10.3 |

The cosine schedule at a peak of 0.004 passes the check: $+0.07$ dB, on eight of the ten images, with SSIM better too and at no
cost in time. It is the default of `run_paint.py` and of the rest of this section (`PainterConfig` keeps the reference's
constant rate, so that the experiments of Sections 4.2 and 4.3 remain reproducible from their recorded settings). Allocation
by error does not pass on its own (its interval just includes zero), and on top of the schedule it adds $+0.036$ dB
($[-0.014, +0.086]$, not significant) for a fifth more time when run alone (40 % in these runs, made two at a time) and almost
twice the memory, so it stays an option. Its one clear use is speed: with 11 steps per stroke it paints as well as the
reference settings by PSNR ($-0.01$ dB) in less than half the time (5.2 against 12.1 s alone), with a
small loss in SSIM and LPIPS, which makes it a candidate for a "fast" setting of the interface of Week 8.

### 4.4.4 Ablation B, second half: progressive against fixed grid and full image

The second half of research question RQ2 compares the three grid modes at the same stroke budget. The protocol is that of
Section 4.2.10 (the oil brush, the pixel loss, the authors' light renderer, the ten images that are every third of the sorted
evaluation set, seeds 0 to 2), with the brush defaults of Week 6, so the fixed grid and the full image are measured again
rather than taken from Table 4.13 (experiment `2026-10-15_ablation_ablation_b_progressive`, 180 paintings). Two controls
separate the grid schedule from the number of optimizer steps, which the progressive mode multiplies by five: the
progressive mode with 11 steps per stroke (495 steps, as many as the fixed grid's 500), and the fixed grid with 125 steps per
stroke (2,500 steps, as many as the progressive mode's 2,475). A sixth configuration uses the centred RMSprop of the
reference's `demo_prog.py`. All six use the reference's constant learning rate; the schedule of Section 4.4.3 comes on top.

**Table 4.29.** Ablation B, second half: mean over the three seeds of the image average, plus or minus the standard deviation
of the three seed means, and the paired difference per image to the fixed grid ($n = 10$). The time is the optimization time
of runs made alone on the desktop (`cuda`; apple, iceland and jay, seed 0; experiment `2026-10-15_ablation_timing_clean_modes`);
the peak memory is the working memory of the painting on `cuda` (Section 4.4.1).

| Mode | Strokes | Steps | PSNR (dB) | SSIM | LPIPS | Difference in PSNR to the fixed grid, 95 % CI, images won | Time (s) | Peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| Fixed $5 \times 5$ grid | 500 | 500 | 20.31 $\pm$ 0.02 | 0.558 $\pm$ 0.001 | 0.564 $\pm$ 0.000 | | 5.7 | 0.38 |
| Full image | 500 | 500 | 13.82 $\pm$ 0.25 | 0.444 $\pm$ 0.002 | 0.681 $\pm$ 0.002 | $-6.49$ $[-7.85, -5.12]$, 0 of 10 | 30.8 | 0.38 |
| Progressive $1 \to 5$ | 495 | 2,475 | 20.55 $\pm$ 0.04 | 0.570 $\pm$ 0.002 | 0.558 $\pm$ 0.004 | $+0.25$ $[+0.09, +0.40]$, 9 of 10 | 12.1 | 0.18 |
| Progressive, centred RMSprop | 495 | 2,475 | 20.56 $\pm$ 0.05 | 0.571 $\pm$ 0.000 | 0.558 $\pm$ 0.002 | $+0.26$ $[+0.12, +0.40]$, 10 of 10 | | 0.18 |
| Progressive, 11 steps per stroke | 495 | 495 | 20.38 $\pm$ 0.02 | 0.559 $\pm$ 0.002 | 0.578 $\pm$ 0.001 | $+0.07$ $[-0.14, +0.28]$, 6 of 10 | 2.8 | 0.17 |
| Fixed grid, 125 steps per stroke | 500 | 2,500 | 20.21 $\pm$ 0.03 | 0.556 $\pm$ 0.001 | 0.556 $\pm$ 0.002 | $-0.10$ $[-0.19, -0.01]$, 3 of 10 | 26.0 | 0.39 |

The progressive mode paints better than the fixed grid by 0.25 dB (Wilcoxon $p = 0.01$), on nine of the ten images; SSIM and
LPIPS lean the same way but their differences ($+0.012$ and $-0.006$) are within their intervals. The gain ranges from
$-0.07$ dB on the dark cereal bowl to $+0.69$ dB on the pale photograph of a balloon and a chair (Figure 4.14). The full image
is far behind, as in Week 5 ($-6.5$ dB, all ten images), and the centred optimizer of the reference changes nothing
($+0.01$ dB against the plain one, $[-0.04, +0.06]$). The progressive mode takes twice the fixed grid's time (12.1 against
5.7 s), less than its five times the steps would suggest, because most of its steps are on the coarse levels, whose batches
are small.

**Figure 4.14.** Target, fixed grid, full image and progressive painting (seed 0) for the images where the progressive mode
gains most, the median image and the image where it loses most
(`report/figures/ablation_b_progressive_qualitative.png`).

*It is the schedule, not the steps.* Giving the fixed grid the progressive mode's 2,500 steps does not help it: it loses
0.10 dB in 4.5 times the time; its neural canvas improves (26.4 to 27.1 dB) while the rasterized painting does not, so the
extra steps fit the surrogate rather than the brush (LPIPS improves slightly, by 0.008). Cutting the progressive mode to the
fixed grid's 495 steps costs it 0.18 dB against its full length ($[-0.30, -0.05]$) and leaves it level with the fixed grid
($+0.07$, not significant) in half the fixed grid's time (2.8 against 5.7 s). The coarse levels are what helps. They lay the
background in large strokes, so the finest level starts from a canvas that is already close; and a block of the finest level
is painted on top of strokes that crossed its borders, so the thin seams of canvas that the fixed grid leaves along the block
borders (Section 4.3.2, Figure 4.9) are covered: they are visible as white flecks in the fixed-grid paintings of Figure 4.14
and absent from the progressive ones. The progressive mode also needs half the working memory, because its finest level
renders 9 strokes per block where the fixed grid renders 20.

The neural-canvas PSNR in the experiment folder is lower for the progressive mode (24.5 against 26.4 dB) although its
painting is better: the progressive mode's last level composites only its own 9 strokes per block over a rasterized canvas,
and its neural canvas is not comparable with the fixed grid's, whose 20 strokes per block are all drawn by the surrogate.

### 4.4.5 The four brushes in progressive mode

Ablation B and the tuning used the oil brush. The same three configurations (the fixed grid, the progressive mode with the
reference's settings, and the progressive mode with the cosine schedule) were run for every brush on the ten held-out images
of Section 4.3.3 with three seeds (experiments `2026-10-15_ablation_brushes_prog_<brush>`, 360 paintings; the 60 oil paintings
that repeat runs of Table 4.28 came out identical to the last digit).

**Table 4.30.** Every brush on the ten held-out images: mean over the three seeds of the image average, and the paired
difference per image ($n = 10$) to the fixed grid and, for the cosine schedule, to the progressive mode with the reference's
constant rate.

| Brush | Fixed grid (dB) | Progressive (dB) | Progressive minus fixed, 95 % CI, images won | Progressive with cosine (dB) | Cosine minus constant, 95 % CI, images won | LPIPS, fixed / progressive with cosine |
| :--- | ---: | ---: | :--- | ---: | :--- | :--- |
| Oil | 20.30 | 20.56 | $+0.26$ $[+0.10, +0.43]$, 9 of 10 | 20.63 | $+0.07$ $[+0.03, +0.11]$, 8 of 10 | 0.528 / 0.517 |
| Watercolour | 19.76 | 21.82 | $+2.06$ $[+1.18, +2.95]$, 10 of 10 | 22.06 | $+0.24$ $[+0.04, +0.45]$, 9 of 10 | 0.494 / 0.441 |
| Marker pen | 19.09 | 20.83 | $+1.74$ $[+0.67, +2.82]$, 10 of 10 | 21.16 | $+0.32$ $[+0.06, +0.59]$, 9 of 10 | 0.523 / 0.444 |
| Coloured tape | 21.07 | 22.22 | $+1.14$ $[+0.68, +1.61]$, 10 of 10 | 22.37 | $+0.15$ $[+0.06, +0.25]$, 9 of 10 | 0.466 / 0.416 |

For the other three brushes the progressive mode gains far more than for oil, 1.1 to 2.1 dB on every image, with SSIM and
LPIPS in agreement (the LPIPS of the marker pen falls from 0.523 to 0.444). The figures show why: on the fixed grid the marker
pen and the tape leave a regular lattice of black canvas along the block borders, and the watercolour, whose strokes are round
and start translucent, white gaps between its discs (Figure 4.15). The shape clamp of Section 4.2.1 keeps a stroke's centre
and end points a tenth of the block away from its border, so a thin stroke cannot reach the border, and with a fixed grid no
other stroke covers it (Section 4.3.2 met these seams with the marker pen); in the progressive mode the strokes of the coarse
levels cross the borders of the finer grids, and the canvas no longer shows. The oil strokes are wide and opaque, and reach
further past their centre, so their seams are smaller (the white flecks of Figure 4.14). The largest gains are on the balloon
image (6 to 7 dB for watercolour and marker pen), a flat pale background where every seam is an error. The cosine schedule
helps every brush (0.15 to 0.32 dB, nine of ten images each), more than oil's 0.07 dB on the same images, so it is the
default for all four.

**Figure 4.15.** Fixed grid and progressive painting with the cosine schedule (seed 0) for the marker pen: the images where the
progressive mode gains most and least and the median image (`report/figures/brushes_progressive_markerpen_qualitative.png`;
the same figure for the watercolour and the tape is `brushes_progressive_watercolor_qualitative.png` and
`brushes_progressive_rectangle_qualitative.png`).

### 4.4.6 Stroke budget (RQ3)

Research question RQ3 asks how quality and time grow with the stroke budget. Every brush painted the ten images of
Ablation B (every third of the evaluation set) with three seeds at budgets of 100, 300, 500 and 1,000 strokes, in the
progressive mode with the cosine schedule (experiments `2026-10-15_ablation_scaling_<brush>`, 480 paintings). The equal
split of the reference paints $\lfloor B / 55 \rfloor \cdot 55$ of a budget $B$, so the four budgets give 55, 275, 495 and 990
strokes, and the curves are drawn against the strokes actually painted. Every budget uses about 500 optimizer steps per level
(the steps per stroke are $\lfloor 500 / n \rfloor$ for $n$ strokes per block, as in the reference), so the budget changes
how many strokes the steps are spread over, not how many steps there are; the time grows because each step renders more
strokes. The times in Table 4.31 come from separate runs made alone on the desktop, on three images with one seed
(`2026-10-15_ablation_timing_clean_scaling_<brush>`).

**Table 4.31.** Quality against the number of strokes painted: mean over the three seeds of the image average (ten images),
and the optimization time of runs made alone (three images, seed 0).

| Brush | Strokes | PSNR (dB) | SSIM | LPIPS | Optimization (s) | Rasterization (s) | Peak memory (GB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Oil | 55 | 17.65 $\pm$ 0.05 | 0.551 | 0.606 | 8.6 | 0.2 | 0.04 |
|  | 275 | 20.34 $\pm$ 0.03 | 0.573 | 0.561 | 9.8 | 0.8 | 0.10 |
|  | 495 | 20.64 $\pm$ 0.05 | 0.572 | 0.554 | 11.7 | 1.4 | 0.17 |
|  | 990 | 20.73 $\pm$ 0.02 | 0.570 | 0.548 | 16.9 | 2.7 | 0.34 |
| Watercolour | 55 | 18.39 $\pm$ 0.11 | 0.617 | 0.546 | 9.1 | 0.2 | 0.04 |
|  | 275 | 21.42 $\pm$ 0.05 | 0.657 | 0.483 | 10.4 | 0.8 | 0.10 |
|  | 495 | 21.95 $\pm$ 0.08 | 0.665 | 0.469 | 12.5 | 1.4 | 0.17 |
|  | 990 | 22.07 $\pm$ 0.03 | 0.673 | 0.458 | 17.6 | 2.7 | 0.34 |
| Marker pen | 55 | 17.75 $\pm$ 0.22 | 0.575 | 0.558 | 8.9 | 0.2 | 0.04 |
|  | 275 | 20.80 $\pm$ 0.05 | 0.644 | 0.482 | 10.4 | 0.8 | 0.10 |
|  | 495 | 21.11 $\pm$ 0.04 | 0.653 | 0.464 | 12.4 | 1.4 | 0.17 |
|  | 990 | 21.07 $\pm$ 0.03 | 0.658 | 0.461 | 17.1 | 3.0 | 0.34 |
| Coloured tape | 55 | 13.77 $\pm$ 0.29 | 0.441 | 0.661 | 9.2 | 0.1 | 0.04 |
|  | 275 | 21.41 $\pm$ 0.02 | 0.653 | 0.477 | 10.8 | 0.5 | 0.10 |
|  | 495 | 22.22 $\pm$ 0.04 | 0.674 | 0.442 | 12.7 | 0.9 | 0.17 |
|  | 990 | 22.74 $\pm$ 0.04 | 0.691 | 0.407 | 17.7 | 1.8 | 0.34 |

**Figure 4.16.** PSNR, LPIPS and optimization time against the number of strokes painted, one curve per brush; the error bars
are the standard deviation over the three seeds (`report/figures/scaling_strokes.png`).

The quality saturates. From 55 to 275 strokes every brush gains 2.7 to 3.1 dB (7.6 dB for the tape, whose few strokes
leave most of its black canvas bare), from 275 to 495 a further 0.3 to 0.8 dB, and from 495 to 990, at 1.4 times the
optimization time and twice the rasterization time, at most 0.1 dB for oil, watercolour and marker pen ($+0.10$, $+0.12$ and
$-0.04$ dB, none significant over the ten images). Only the tape keeps improving: $+0.53$ dB ($[+0.24, +0.81]$, nine of ten
images) with SSIM and LPIPS in agreement ($+0.017$ and $-0.035$, both significant). With the light renderers and $M = 5$, 500
strokes are therefore close to the useful maximum for three of the four brushes, and the default of 500 is kept; the tape
gains from 1,000. A finer grid (a larger $M$)
or the full-size renderer might move the knee; neither was measured here.

### 4.4.7 Speed and memory

The plan for this week lists four kinds of performance work: batching the strokes of the grid cells, `torch.compile`, mixed
precision, and memory-aware batch sizes for `mps` and the CPU. Measuring where the time goes came first, and it found a
fifth, larger cause on the desktop.

*Windows' power throttling.* A painting that ran from a background shell got slower the longer it ran. On the CPU an
optimizer step of the light renderer with 512 strokes took 0.31 s in a fresh process but 1.24 s after the first large steps,
and from then on every step was about 3.5 times slower, small ones included. The count of page faults per step did not
change, so memory was not the cause. Windows 11 throttles a busy process that is not in the foreground ("EcoQoS"): it moves
its threads to the efficiency cores of the i5-13400F at a low clock. Opting the process out of that
(`core.device.disable_power_throttling`, one call to `SetProcessInformation`) removed the slowdown. `scripts/profile_step.py`
reproduces it by forcing the throttling on and off in fresh processes (experiment `2026-10-15_profile_step-4`): a CPU step
takes 3.5 to 3.9 times as long when throttled (0.09 against 0.33 s for 128 strokes, 0.32 against 1.24 s for 512), and on
`cuda`, where the time of a step is spent on the CPU launching small kernels (below), a fixed-grid painting of `apple` takes
7.0 instead of 5.8 s, a progressive one 21.4 instead of 12.1 s, and the rasterization of the 512-pixel painting 2.7 instead of
1.3 s (two paintings each). Every script that paints or times (`run_paint.py`,
`run_ablation.py`, `compare_with_original.py`, `benchmark.py`, `device_check.py` and the renderer training) now opts out first,
and so does the reference's demo when `compare_with_original.py` runs it. This is very likely the "factor of two over a day" by
which the desktop's speed was noted to vary in Weeks 5 and 6 (Sections 4.2.12 and 4.3.7): it depends on whether Windows
considered the process to be in the foreground. The timings of those weeks are therefore upper bounds, and only their ratios
within one experiment should be read. Whether the MacBook throttles in a similar way is not known; its benchmark runs will
show whether its timings drift.

*Where a step's time goes on `cuda`.* With the throttling out of the way, `scripts/profile_step.py` profiled one fixed-grid
painting of `apple` (500 strokes, 500 optimizer steps; experiment `2026-10-15_profile_step-4`): 143,571 kernel launches, about
290 per step, and the GPU ran kernels for 4.7 of the 7.3 s the painting took under the profiler, so it is busy about two thirds
of the time. The rest is the CPU launching small kernels: the light renderer works on $32 \times 32$ images, and a step
composites the strokes one at a time. Three ways to make a step cheaper were timed on the same painting (Table 4.32).

**Table 4.32.** Optimization time of one fixed-grid painting (`apple`, 500 strokes, light oil renderer, `cuda`, seed 0), mean of
two runs each, and the PSNR of the painting (experiment `2026-10-15_profile_step-4`; earlier runs of the script gave the same
times within 0.1 s).

| Variant | Optimization (s) | Change | PSNR (dB) |
| :--- | ---: | ---: | ---: |
| As built: strokes composited one at a time, float32 | 5.82 | | 20.73 |
| Strokes composited in one closed-form product | 5.23 | $-10$ % | 20.48 |
| Renderer under autocast to float16 | 4.36 | $-25$ % | 20.86 |
| Renderer under autocast to bfloat16 | 4.36 | $-25$ % | 20.63 |

*Batching.* The engine already renders the strokes of all blocks of a level as one batch, which is the batching the plan asked
for. Compositing the strokes in one closed-form product,
$C = \sum_i A_i F_i \prod_{j > i} (1 - A_j) + C_0 \prod_j (1 - A_j)$ with one cumulative product, saves 10 % (it is the same
function, checked in the tests, but its rounding differs, and the painting with it). *Mixed precision* saves a quarter: the
renderer's matrix products and convolutions run in 16-bit floating point while the stroke parameters, the loss and the
optimizer stay in float32 (`PainterConfig.precision`). bfloat16 keeps float32's exponent range, so the small gradients of a mean
pixel loss over a batch of blocks cannot underflow, as they can in float16 without a gradient scaler; it is the one the
package offers (`--precision bfloat16`). Whether it changes the paintings is measured below. *`torch.compile`* needs the Triton
compiler for GPU code generation, which is not available for Windows in this PyTorch build, so it could not be measured on the
desktop.

**Table 4.33.** The progressive mode with the cosine schedule in bfloat16 against float32 on the ten held-out images, three seeds
(experiments `2026-10-15_ablation_precision_<brush>`, 240 paintings): paired difference per image with its 95 % interval and
the number of images improved, and the change in optimization time (in these runs, made two at a time) and peak memory.

| Brush | PSNR (dB) | SSIM | LPIPS | Time | Peak memory (GB) |
| :--- | :--- | :--- | :--- | ---: | ---: |
| Oil | $+0.04$ $[-0.01, +0.09]$, 7 of 10 | $-0.002$ $[-0.003, -0.000]$ | $+0.001$ $[-0.002, +0.004]$ | $-12$ % | 0.18 to 0.10 |
| Watercolour | $-0.13$ $[-0.21, -0.04]$, 1 of 10 | $-0.001$ $[-0.005, +0.003]$ | $-0.001$ $[-0.005, +0.004]$ | $-15$ % | 0.17 to 0.10 |
| Marker pen | $+0.08$ $[-0.04, +0.20]$, 8 of 10 | $-0.002$ $[-0.006, +0.001]$ | $+0.004$ $[-0.001, +0.009]$ | $-13$ % | 0.18 to 0.10 |
| Coloured tape | $-0.02$ $[-0.10, +0.06]$, 5 of 10 | $-0.000$ $[-0.003, +0.002]$ | $-0.001$ $[-0.004, +0.003]$ | $-13$ % | 0.18 to 0.10 |

Three brushes paint as well in bfloat16, but the watercolour loses 0.13 dB, on nine of the ten images. The saving in time
depends on the load. In these runs, made two at a time so that the GPU was shared, bfloat16 was 12 to 15 % faster; run alone,
the progressive painting of three images took 13.4 s in bfloat16 against 12.3 s in float32
(`2026-10-15_ablation_timing_clean_modes`), 9 % slower: its levels render few strokes per step, so the step is limited by the
launches, and the conversions add launches. Only the large batches of the fixed grid (Table 4.32) gain on an idle GPU. The
working memory falls by about 40 % in every case. Reduced precision is therefore not adopted as a default; it remains the
option `--precision bfloat16` of `run_paint.py`, worth using when memory is short.

*Memory.* With the light renderers the working memory of a painting is small: the fixed grid peaks at 0.38 GB and the
progressive mode at 0.17 GB on `cuda` (Table 4.29), and it grows in proportion to the strokes per block, to 0.34 GB for 990
strokes in the progressive mode (Table 4.31); allocation by error almost doubles it, because the batch is padded to the
largest block's count. The full-size renderers ($128 \times 128$ pixels, 18.1 M parameters) need ten to twenty times more: painting
`apple` with the full oil renderer took 3.4 to 3.5 GB in the fixed grid and 1.5 to 1.6 GB in the progressive mode at 500
strokes, and 6.8 and 3.1 to 3.2 GB at 1,000 strokes (experiments `2026-10-15_ablation_timing_clean_full_renderer` and its
`_repeat`, the same values within 0.1 GB in both). The fixed grid with 1,000 strokes thus comes close to the 8 GB of the
RTX 3070, and more strokes would not fit; the progressive mode stays below half of it. The plan's "memory-aware batch sizing" is
therefore not needed for the light renderers on any of the three devices, and the progressive mode is the memory-aware choice
for the full ones. (The full renderer also paints better, 23.7 dB against 21.4 dB for the light one on this image in the
progressive mode with the same settings, but its optimization time varied from 46 to 252 s between two runs of the same
painting, and only one image was painted, so neither number measures the full renderers; that comparison is not part of this
week's experiments.)

*Hardware benchmark, first pass.* `scripts/benchmark.py` paints the same three images (every tenth of the evaluation set) with
every brush in the progressive mode with the pipeline's defaults and in the fixed grid, seed 0, on one device profile, and
writes one folder per brush (`2026-10-15_benchmark_<profile>_<brush>`). On the desktop it ran for `cuda` (the RTX 3070) and
`cpu` (the i5-13400F, 10 threads), each alone; the `mps` profile and the MacBook's CPU are to be run with the same commit on
the MacBook.

**Table 4.34.** The hardware benchmark on the desktop: means over three images, seed 0, the light renderers. The CPU's memory is
the rise of the process's resident memory (Section 4.4.1), so it is not the same quantity as the GPU's tensor memory. The PSNR
is that of the `cuda` paintings; the `cpu` paintings differ from them by $-0.01$ to $+0.35$ dB, the rounding of the two backends
taking the chaotic optimization to different strokes.

| Brush | Mode | `cuda`: optimize (s) | `cuda`: peak memory (GB) | `cpu`: optimize (s) | `cpu`: memory rise (GB) | Rasterize, both (s) | PSNR (dB) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oil | progressive | 12.0 | 0.18 | 130 | 0.13 | 1.5 | 22.74 |
|  | fixed | 5.6 | 0.38 | 108 | 0.19 | 1.4 | 22.31 |
| Watercolour | progressive | 12.4 | 0.18 | 130 | 0.11 | 1.5 | 23.75 |
|  | fixed | 5.7 | 0.38 | 108 | 0.22 | 1.4 | 20.15 |
| Marker pen | progressive | 12.4 | 0.18 | 131 | 0.10 | 1.6 | 23.32 |
|  | fixed | 5.7 | 0.38 | 109 | 0.18 | 1.5 | 19.11 |
| Coloured tape | progressive | 12.9 | 0.18 | 131 | 0.09 | 1.0 | 24.70 |
|  | fixed | 5.7 | 0.38 | 109 | 0.19 | 1.0 | 22.80 |

On the CPU a painting takes about two minutes: 130 s in the progressive mode and 108 s on the fixed grid, 10 and 19 times as
long as on the GPU. The ratio of the two modes differs between the devices (1.2 on the CPU, 2.2 on the GPU) because they are
limited by different things: on the CPU every step costs in proportion to the strokes it renders, and the two modes render
about as many strokes in all; on the GPU the progressive mode's five times as many steps cost launches whatever their size.
The time does not depend on the brush on either device (the four light renderers have the same size); only the
rasterization of the final painting does (0.9 to 1.7 s). Memory is no constraint for the light renderers on any device.

### 4.4.8 End-to-end test on many inputs

The evaluation set is uniform by construction: thirty PNG squares of 512 pixels (Section 5.2). A user's photographs are not,
so `scripts/check_end_to_end.py` runs `run_paint.py` with its defaults (progressive, 500 strokes, the cosine schedule) on
twenty inputs and checks each outcome (experiment `2026-10-15_check_end_to_end`). The brushes and the three fits are taken in
turn, so that every brush meets every fit.

**Table 4.35.** The end-to-end check: input, brush and fit, the size of the saved painting, and its PSNR against the fitted
photograph (`cuda`; optimization took 11.7 to 13.3 s for every painting but the first of the process, 23.5 s).

| Input | Size | Brush, fit | Saved painting | PSNR (dB) |
| :--- | :--- | :--- | :--- | ---: |
| alien.jpg (reference test image) | 494 x 496 | oil, stretch | 509 x 512 | 19.17 |
| apple.jpg | 1021 x 854 | watercolour, crop | 512 x 512 | 22.48 |
| cube2.jpg | 533 x 483 | marker pen, pad | 512 x 512 | 20.55 |
| diamond.jpg | 396 x 377 | tape, stretch | 512 x 487 | 16.07 |
| diamond2.jpg | 477 x 463 | oil, crop | 512 x 512 | 19.98 |
| fire.jpg | 748 x 709 | watercolour, pad | 512 x 512 | 18.67 |
| iceland.jpg | 568 x 452 | marker pen, stretch | 512 x 407 | 21.07 |
| jay.jpg | 729 x 640 | tape, crop | 512 x 512 | 24.35 |
| joker.jpg | 559 x 563 | oil, pad | 512 x 512 | 18.58 |
| sunflowers.jpg | 677 x 672 | watercolour, stretch | 512 x 508 | 19.89 |
| yosemite.jpg | 640 x 480 | marker pen, crop | 512 x 512 | 19.90 |
| grey-scale PNG (generated) | 400 x 300 | tape, pad | 512 x 512 | 27.04 |
| PNG with an alpha channel | 400 x 300 | oil, stretch | 512 x 384 | 28.98 |
| 4:1 panorama, JPG | 1600 x 200 | watercolour, crop | 512 x 512 | 29.42 |
| 1:3 portrait, JPG | 200 x 1200 | marker pen, pad | 512 x 512 | 23.22 |
| 40 x 30 thumbnail, PNG | 40 x 30 | tape, stretch | 512 x 384 | 30.10 |
| 16-bit grey TIFF | 400 x 300 | oil, crop | 512 x 512 | 31.20 |
| random bytes named `.png` | | | refused: "cannot decode" | |
| `.gif` file | | | refused: unsupported format, with the list of supported ones | |
| 6332 x 6332 PNG (40.1 M pixels) | | | refused: more than the limit of 40,000,000 pixels | |

All twenty behaved as expected: seventeen paintings, each saved in the photograph's aspect ratio under `stretch` and square
under `crop` and `pad`, and three refusals with a message and without an output folder. Preparing the check found one bug: a
16-bit grey image (PNG or TIFF), which Pillow opens in mode `I;16`, loaded almost white, because converting it to RGB clips its
values at 255 of 65,535; `image_io.load_image` now scales such images to $[0, 1]$ (tested), and the painting of the generated
16-bit TIFF in Table 4.35 was made after the fix. The generated images are smooth gradients, which is why they score 23 to
31 dB.

### 4.4.9 Summary of the section

**Table 4.36.** What the Week 7 choices are worth: paired PSNR difference per image, with the brushes and images concerned.

| Choice | Alternative | Difference in PSNR (dB) | Time on `cuda` | Where |
| :--- | :--- | ---: | :--- | :--- |
| Progressive grid $1 \to 5$ | fixed $5 \times 5$ grid | $+0.25$ (oil), $+1.1$ to $+2.1$ (watercolour, marker pen, tape), every image for the last three | $\times 2.1$ | Tables 4.29, 4.30 |
| Full image | fixed $5 \times 5$ grid | $-6.5$ (oil) | $\times 5.4$ | Table 4.29 |
| Cosine schedule, peak 0.004 | constant 0.002 | $+0.07$ (oil) to $+0.32$ (marker pen) | same | Tables 4.28, 4.30 |
| Allocation by error | equal split | $+0.08$ (oil; not significant), $+0.04$ on top of the schedule | $\times 1.2$, memory $\times 1.7$ | Table 4.28 |
| 11 steps per stroke with allocation by error | 55 steps, equal split | $-0.01$ (oil) | $\times 0.4$ | Table 4.28 |
| bfloat16 | float32 | $-0.13$ to $+0.08$ (brush-dependent) | $\times 1.09$ alone, memory $\times 0.6$ | Table 4.33 |
| 990 strokes | 495 | $+0.10$ to $+0.12$ (not significant), $-0.04$ (marker pen), $+0.53$ (tape) | $\times 1.4$ | Table 4.31 |
| Opting out of Windows power throttling | throttled | identical paintings | $\times 0.57$ (progressive), $\times 0.25$ to $0.29$ per CPU step | Section 4.4.7 |

The answer to the second half of RQ2 is that the progressive mode is the best of the three: better than the fixed grid for
every brush, by a quarter of a decibel for oil and by one to two decibels for the brushes whose thin strokes leave the fixed
grid's block borders bare, at twice the time and half the memory; the full image is far behind. The gain comes from the
coarse-to-fine schedule, not from its five-fold number of optimizer steps. To RQ3, quality grows quickly up to about 300
strokes and is close to saturation at 500 for every brush but the tape, while the time grows steadily (9 s at 100 strokes,
12 s at 500 and 17 s at 1,000 on `cuda`). To RQ4, a painting of 500 strokes in the progressive mode takes 12 s on the RTX 3070
and 130 s on the i5-13400F (the fixed grid 5.7 and 108 s); every setting with the light renderers stays below 0.4 GB of
working memory, and the full renderers need up to 6.8 GB.

The pipeline that `run_paint.py` runs by default is therefore the progressive mode with the cosine schedule at a peak of
0.004 and 500 strokes. The limits: the options were tuned on oil and checked on ten images; the brushes and budgets used the
authors' light renderers only; the timings are those of one desktop, measured alone with three images and one seed, and the
`mps` profile has still to be run on the MacBook.
