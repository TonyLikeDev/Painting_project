# Week 7: End-to-end automatic pipeline and performance

Status mirror of `ROADMAP.md`. Done on the desktop (`cuda`, RTX 3070, and `cpu`, i5-13400F) on 2026-10-15. Every painting run set
`--deterministic` and opted out of Windows' power throttling. The tables are in the experiment folders named below
(`results.csv`, `summary.md`, `config.yaml`; the paintings are in the git-ignored `runs/` folders); the prose is in
`report/chapters/04_implementation.md`, section 4.4.

## Build

- [x] Code check before new code (2026-10-15): four problems found and fixed: `paired_folders.py` silently overwrote runs when a
  folder held several configs and none was named (now refused); `summarize_ablation.py` failed on folders older than a new
  column (missing columns now read as empty); resuming an old `results.csv` would have misaligned new columns (the runner now
  keeps an existing file's header); `peak_memory_gb` on `mps` reported current, not peak, memory (documented; the new
  `PeakMemory` measures the peak on every backend). A fifth problem was found later by the end-to-end check: 16-bit grey
  PNG / TIFF images loaded almost white (`convert("RGB")` clips at 255 of 65535); fixed in `image_io.load_image`.
- [x] `pipeline/progressive_painter.paint_image`: grid modes `progressive` (1..M, the reference's `demo_prog.py`), `fixed`
  (identical strokes to `paint_fixed_grid`, tested to the bit) and `full`; incremental rasterization; `Progress` events with the
  neural canvas for a live preview.
- [x] Engine: per-block stroke counts with transparent unused slots (`paint(strokes=...)`), `allocation="error"` (D'Hondt split by
  block error, capped at twice the mean), `lr_schedule="cosine"` (warm restart at every stroke), `precision="bfloat16"|"float16"`
  (autocast), the canvas in the progress callback. Defaults unchanged (the reference's).
- [x] `scripts/run_paint.py`: one command, photo to painting, live status line, clean refusal of bad inputs, output folder with
  final / input PNG, strokes, config and results; defaults progressive + cosine schedule at 0.004.
- [x] `scripts/benchmark.py` (per device profile), `scripts/profile_step.py`, `scripts/check_end_to_end.py`,
  `scripts/plot_scaling.py`; `compare_with_original.py --mode progressive` (runs `demo_prog.py`); `run_ablation.py` configs can
  set `mode`, `strokes`, `allocation`, `lr_schedule`, `iters_per_stroke`, `precision`, and every row records optimizer steps and
  peak memory (`core.device.PeakMemory`).
- [x] `core.device.disable_power_throttling()` called by every painting, timing and training script (see Measure).
- [x] Tests: `tests/test_progressive.py` and new tests in the script, grid, image and comparison tests: 253 pass (235 before).

## Measure

| Experiment | Folder | Runs | Result |
| :--- | :--- | ---: | :--- |
| Progressive vs the reference `demo_prog.py` (oil, 5 images x 3 repeats) | `2026-10-15_check_oil_progressive` | 45 | reference settings: -0.05 dB [-0.15, +0.05] (same); package defaults +0.65 dB, 5 of 5 |
| Ablation B, second half (oil, 10 images x 3 seeds) | `2026-10-15_ablation_ablation_b_progressive` | 180 | progressive +0.25 dB over the fixed grid [+0.09, +0.40], 9 of 10; full image -6.5 dB; centred RMSprop +0.01; fixed grid with 5x steps -0.10; progressive with the fixed grid's steps +0.07 (n.s.) in half its time |
| Options, tuning (oil, 3 images) | `..._tuning_pipeline_oil`, `..._tuning_pipeline_iters` | 126 | allocation by error +0.21, cosine at 0.004 +0.12, cosine at 0.002 -0.11; steps per stroke 55 best, 22 -0.08, 11 -0.28 |
| Options, held-out check (oil, 10 images) | `2026-10-15_ablation_confirm_pipeline_oil` | 150 | cosine 0.004 +0.07 [+0.03, +0.11] adopted; allocation +0.08 [-0.00, +0.17] not adopted; both +0.11; allocation with 11 steps per stroke equal quality in half the time |
| Every brush, fixed vs progressive vs progressive + cosine (10 held-out images) | `2026-10-15_ablation_brushes_prog_<brush>` | 360 | progressive over fixed: oil +0.26, watercolour +2.06, marker +1.74, tape +1.14 dB (every image for the last three); cosine adds +0.07 to +0.32 |
| Stroke budget 100 / 300 / 500 / 1000 (4 brushes, 10 images) | `2026-10-15_ablation_scaling_<brush>` | 480 | 55 / 275 / 495 / 990 strokes painted; saturates near 500 except tape (+0.53 dB at 990) |
| bfloat16 vs float32 (4 brushes, 10 held-out images) | `2026-10-15_ablation_precision_<brush>` | 240 | oil +0.04, watercolour -0.13 (9 of 10 worse), marker +0.08, tape -0.02 dB; memory -40 %; not adopted |
| Clean timing (alone, 3 images, seed 0) | `2026-10-15_ablation_timing_clean_modes`, `..._timing_clean_scaling_<brush>` | 27 + 48 | fixed 5.7 s, progressive 12.1 s, progressive with 11 steps 2.8 s, full image 30.8 s; 100 to 1000 strokes 9 to 17 s; bfloat16 progressive 13.4 s (slower alone) |
| Full-size renderer, memory (apple, seed 0) | `..._timing_clean_full_renderer`, `..._repeat` | 4 + 3 | fixed 3.4 GB at 500 strokes, 6.8 GB at 1000; progressive 1.6 / 3.1 GB; times varied up to 5x between repeats |
| Step profile and throttling (cuda, cpu) | `2026-10-15_profile_step-4` | | 143,571 launches per fixed-grid painting, GPU busy 4.7 of 7.3 s; float16 and bfloat16 -25 %, closed-form composite -10 %; throttled: CPU step 3.5 to 3.9x, fixed grid 7.0 vs 5.8 s, progressive 21.4 vs 12.1 s, rasterization 2.7 vs 1.3 s |
| Hardware benchmark, first pass (4 brushes x 2 modes x 3 images) | `2026-10-15_benchmark_cuda_<brush>`, `2026-10-15_benchmark_cpu_<brush>` | 24 + 24 | `cuda`: progressive 12.0 to 12.9 s, fixed 5.6 to 5.7 s, 0.18 / 0.38 GB; `cpu` (10 threads): progressive 130 s, fixed 108 s, memory rise 0.09 to 0.22 GB; rasterization 0.9 to 1.7 s |
| End-to-end check of `run_paint.py` (20 inputs) | `2026-10-15_check_end_to_end` | 20 | 20 of 20 as expected: 11 reference photos of 377 to 1021 px, grey, alpha, 4:1, 1:3, 40 x 30, 16-bit; three refused with a message |

Superseded: `2026-10-15_profile_step` (its GPU time summed the profiler's per-operator totals, which count a kernel twice),
`..._profile_step-2` (correct, but without the whole-painting throttling rows) and `..._profile_step-3` (its mixed-precision rows
ran in float32: the engine's new `precision` option, at `float32`, switches off the script's outer autocast). `-4` uses the
engine option and is the one the report cites; the timings of all four agree within 0.1 s where they measured the same thing.

## Write

- [x] Report section 4.4 (4.4.1 to 4.4.9): pipeline and end-to-end diagram (Figure 4.13), progressive mode and agreement with
  the reference (Table 4.26), the two options (Tables 4.27, 4.28), Ablation B second half (Table 4.29, Figure 4.14), the four
  brushes (Table 4.30, Figure 4.15), stroke budget (Table 4.31, Figure 4.16), speed, memory and the hardware benchmark
  (Tables 4.32 to 4.34), end-to-end check (Table 4.35), summary (Table 4.36). Bibliography: Loshchilov and Hutter 2017 (SGDR).

## Not done / handed over

- `mps` and the MacBook's `cpu` rows of the benchmark: `python scripts/benchmark.py --profile mps` and `--profile cpu` on the
  MacBook (about 15 and 60 minutes).
- `torch.compile`: Triton is not available for Windows in this build; not measured.
