# 4. Implementation

This chapter describes how the method of Chapter 2 is implemented in the `neural_painter`
package. It grows week by week; this version covers the neural renderer (Week 4). Unless a
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
