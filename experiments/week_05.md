# Week 5: Stroke optimization against the target image

Status mirror of `ROADMAP.md`. Done on the desktop (`cuda`, RTX 3070) on 2026-10-03. The
tables are in the experiment folders named below (each has `results.csv`, `summary.md`
and `config.yaml`; the paintings are in the git-ignored `runs/` folders); the prose is in
`report/chapters/04_implementation.md`, section 4.2.

## Build

- [x] `losses/sinkhorn.py`: Sinkhorn cost and divergence, `SinkhornLoss`; value and gradient
  equal to the original's; `reference` and `exact` log-sum-exp modes; matrix-based
  `area_resize` because Apple MPS cannot pool 32 to 24.
- [x] `pipeline/stroke_sampler.py`: error map, `box_blur` identical to `cv2.blur`,
  `StrokeSampler` with draws that match the original's distribution; `uniform` option
  for the initialization ablation.
- [x] `pipeline/painter_engine.py`: `PainterEngine` (the original's loop), `paint_fixed_grid`
  (grid 1 is full image mode), stroke `.npz` in the original layout, `probe` hook,
  `PainterConfig.init` and `.resize` options; `pipeline/metrics.py` (PSNR, SSIM, optional
  LPIPS).
- [x] Experiment tools: `scripts/run_ablation.py` (per-config grid, config overrides,
  `--every`, `--deterministic`), `scripts/summarize_ablation.py` (per-image paired
  statistics), `scripts/plot_ablation.py`, `scripts/plot_sweep.py`,
  `scripts/compare_with_original.py`, `scripts/loss_gradients.py`, `scripts/score_lpips.py`;
  a painting step in `scripts/device_check.py` for the MacBook.
- [x] Tests for all of the above.

## Measure

| Experiment | Folder | Runs | Result |
| :--- | :--- | ---: | :--- |
| A: pixel against pixel + Sinkhorn (0.1), 30 images | `2026-10-03_ablation_a_pixel_vs_sinkhorn` | 180 | PSNR 18.67 against 18.69 dB; paired +0.011 dB (95 % CI -0.027 to +0.049); 2.6 times the time |
| Weight sweep 0.1, 1, 10, 100; 10 images | `2026-10-03_ablation_sinkhorn_weight_sweep` | 150 | no effect up to 10; -0.104 dB at 100 (CI -0.178 to -0.030) |
| Gradient size, 3 images | `2026-10-03_diagnostic_gradient_ratio` | 120 probes | Sinkhorn gradient 0.3 to 0.9 % of the pixel loss's |
| Epsilon sweep at weight 100; 5 images | `2026-10-03_ablation_sinkhorn_epsilon_sweep` | 75 | harm grows with epsilon (-0.14, -0.48, -3.47 dB); +0.09 dB at 0.003 (CI -0.02 to +0.20) |
| A2: error-map against uniform start; 10 images | `2026-10-03_ablation_a2_initialization` | 120 | error-map start +0.87 dB; Sinkhorn from the uniform start -0.26 dB |
| A3: dilate / erode on against off; 10 images | `2026-10-03_ablation_a3_morphology` | 60 | without it -2.71 dB (CI -3.95 to -1.47), every image worse |
| B: full image against 5x5 grid; 10 images | `2026-10-03_ablation_b_full_vs_grid` | 60 | full image -5.70 dB (CI -7.02 to -4.39), every image worse |
| Original 2021 code against this package; 5 images | `2026-10-03_check_oil_fixed_grid` | 45 | with the original's resizing -0.02 dB (CI -0.22 to +0.17); the default (area) +0.30 dB |
| Clean timing, 3 images | `2026-10-03_ablation_timing_clean` | 9 | grid 6.8 s, grid + Sinkhorn 15.8 s, full image 72 s (the machine was slower than in the morning) |

- [ ] LPIPS is missing from every table until the AlexNet weights are downloaded (the
  student's go-ahead is needed); `scripts/score_lpips.py` then fills it in.

## Write

- [x] Report sections 4.2.1 to 4.2.12 (loop, Sinkhorn, sampler, verification, protocol,
  Ablation A, sweeps and gradients, morphology, initialization, full image against grid,
  agreement with the original, summary), figures 4.3 to 4.7, tables 4.5 to 4.15; chapter 5
  rules 5 (paired per image) and the metric table updated; chapter 2 section 2.3.2 corrected.

## Exit criteria

- [x] Optimizer test passes.
- [x] Ablation A table complete with means and standard deviations (without LPIPS).

## Notes and deviations

See `ROADMAP.md`, Week 5, "Notes and deviations". The main ones: the unit of the paired
statistics is the image; items were added (comparison with the original, gradients, A2,
A3); sweeps use subsets of the images; CUDA runs are not bit-reproducible unless
`--deterministic` is given; timings moved by a factor of two over the day; the marker pen
cannot start with the original's initialization (Week 6).
