# Week 6: Multi-material brushes

Status mirror of `ROADMAP.md`. Done on the desktop (`cuda`, RTX 3070) on 2026-10-10. Every painting run set
`--deterministic`. The tables are in the experiment folders named below (`results.csv`, `summary.md`, `config.yaml`; the
paintings are in the git-ignored `runs/` folders); the prose is in `report/chapters/04_implementation.md`, section 4.3, and
the technical reference in `report/notes/render_module.md`.

## Build

- [x] Marker-pen start fixed: `ProceduralRasterizer.initial_params_at(..., chord=)` starts a Bezier stroke as a straight
  segment of a given length (random direction, control point mid-chord, centre moved inwards per axis only as far as both ends
  need to stay in the painter's shape range). `chord = 0` gives exactly the Week 5 draws (the generator state is checked).
  Invalid values (negative, longer than the shape range, given for oil or tape) are refused.
- [x] Per-call overrides of the start ranges (`size_range`, `alpha_range`, `chord`, `bounds`), passed through
  `StrokeSampler.sample(**init)` and `PainterConfig.init_size_range / init_alpha_range / init_chord`; `run_ablation.py`
  config fields `init_chord=`, `init_size_range=LO:HI`, `init_alpha_range=LO:HI`.
- [x] Tuned defaults in `configs/*.yaml` (each checked on ten held-out images): start sizes [0.2, 0.4] for oil, watercolour
  and tape; watercolour start opacity [0.5, 0.7]; marker start length 0.4. Reference starts: `init_size_range=0.1:0.25`,
  `init_alpha_range=0.98:1.0` (watercolour), `init_chord=0` (marker).
- [x] Experiment hygiene: `run_ablation.py` and `compare_with_original.py` record the brush's start ranges in `config.yaml`
  and refuse to resume a folder painted under other ones; `compare_with_original.py`'s `ours_linear` pins the reference
  start sizes.
- [x] New scripts: `scripts/marker_chord.py` (which renderers draw a marker stroke of a given length),
  `scripts/compare_brushes.py` (one table and image grid over several folders, only complete common runs),
  `scripts/paired_folders.py` (paired per-image statistics across two folders, e.g. two renderers),
  `scripts/plot_marker_chord.py`.
- [x] Renderers trained for watercolour, marker pen and tape (light, 100 epochs, the Week 4 recipe), plus tape with the
  reference's 400-epoch schedule: `checkpoints/<brush>_light_100ep/`, `checkpoints/rectangle_light_400ep/` (git-ignored).
- [x] Tests: `tests/test_brush_init.py` (14 tests), `tests/test_rasterizer.py`, `tests/test_optimizer.py` and
  `tests/test_compare_with_original.py` updated. 235 tests pass (221 before).

## Measure

| Experiment | Folder | Runs | Result |
| :--- | :--- | ---: | :--- |
| Marker stroke length vs renderer | `2026-10-10_diagnostic_marker_chord` | 4 x 8 x 512 strokes | light renderer draws nothing below a chord of 0.075, everything from 0.1; full renderer draws 83 % of dots |
| Marker start length, tuning (3 images) | `2026-10-10_ablation_tuning_marker_chord_fixed` (+ dot row of `..._tuning_marker_chord`) | 63 (+9) | dot 4.2 dB, 0.1: 17.1, 0.4: 18.1, plateau 0.4 to 0.6 |
| Marker start length, held-out (10 images) | `2026-10-10_ablation_confirm_start_markerpen_fixed` | 60 | 0.4 vs 0.1: +1.18 dB [+0.95, +1.42], 10 of 10 |
| Start sizes, tuning | `2026-10-10_ablation_tuning_size_<brush>` (oil, watercolour, tape) | 3 x 27 | larger range best by PSNR for all three |
| Start sizes, held-out | `2026-10-10_ablation_confirm_start_<brush>` | 3 x 60 | [0.2, 0.4]: +0.39 (oil), +0.35 (watercolour), +0.36 dB (tape), every image |
| Start opacity, tuning | `2026-10-10_ablation_tuning_alpha_<brush>`, `..._tuning_alpha_watercolor_low` | 108 | watercolour gains (+0.36 dB at [0.5, 0.7]); the marker's [0.5, 0.7] (+0.20 dB on all three images, interval includes zero) went to the held-out check; tape no gain |
| Start opacity, held-out | `2026-10-10_ablation_confirm_alpha_watercolor`, `..._confirm_alpha_markerpen` | 120 | watercolour [0.5, 0.7]: +0.30 dB, adopted; marker [0.5, 0.7]: +0.09 dB n.s., LPIPS worse, rejected |
| Four brushes, same 3 images | `2026-10-10_brushes_brush_comparison_final` (summary of `..._brushes_oilpaintbrush`, `_watercolor_final`, `_markerpen_fixed`, `_rectangle`) | 36 | tape 21.53, oil 20.86, watercolour 20.36, marker 19.41 dB |
| Ablation A per brush (10 images) | `2026-10-10_ablation_ablation_a_watercolor_final`, `_markerpen`, `_rectangle` | 3 x 60 | Sinkhorn: +0.04, -0.02, +0.05 dB, none significant; 2.6 to 2.7 x the time |
| Renderer fidelity | `2026-10-10_fidelity_week6` | 13 renderers | ours: oil -0.22, watercolour -0.45, marker -0.64, tape -0.80 dB; tape at 400 epochs +0.19 dB (so for tape the gap is the schedule) |
| Painting with our renderers, held-out | `2026-10-10_ablation_heldout_ours_<brush>`, `..._rectangle_400ep` | 5 x 30 | watercolour +1.21, marker +1.20 (medians +0.78, +0.71), tape +0.02 / +0.28, oil -0.15 dB |
| Painting with our renderers, 3 images | `2026-10-10_ablation_brushes_ours_<brush>`, figure `2026-10-10_brushes_watercolor_renderers` | 5 x 9 | same picture as the held-out set |
| Clean timing (idle machine) | `2026-10-10_ablation_timing_clean_<brush>` | 12 | optimize 5.8 s for every brush; render 0.9 (tape) to 1.4 s (marker) |
| Figure runs | `2026-10-10_ablation_marker_start_figure` | 27 | repeats of earlier runs, identical to every digit |

Superseded folders (kept, not cited in the report):

- `..._tuning_marker_chord` (configs 0.1 to 0.3), `..._tuning_marker_chord_long`, `..._confirm_start_markerpen`,
  `..._brushes_markerpen`, `..._brushes_brush_comparison`: made before a bug in the marker start was fixed (the centre was
  moved inwards by half the chord on both axes whatever the direction, so long starts lost part of the error-map placement).
  The dot row of `..._tuning_marker_chord` is unaffected and is cited. The corrected runs changed no conclusion.
- `..._brushes_watercolor`, `..._ablation_a_watercolor`, `..._brushes_brush_comparison_fixed`: the watercolour with its
  earlier start opacity [0.98, 1.0], before the opacity tuning.

## Write

- [x] Report section 4.3 (4.3.1 to 4.3.7): materials table, marker start (diagnostic, sweep, held-out check), start sizes
  and opacity, the four brushes on the same images, Sinkhorn per brush, retrained renderers and painting with them, summary;
  Tables 4.16 to 4.25, Figures 4.8 to 4.12.
- [x] Technical documentation of the render module: `report/notes/render_module.md` (files, the four brushes, start options
  and tuned defaults, renderers, how to add a brush).

## Exit criteria

- [x] All four renderer checkpoints exist (ours, light; the authors' eight are used as the reference).
- [x] Comparison figure and table produced (Table 4.21, Figure 4.10).

## Notes and deviations

- **The marker pen could not paint with the reference start** (Week 5 finding): fixed by starting strokes as segments; the
  light renderer is the reason (Table 4.17). Not a change of the rasterizer, so the pixel-exact port is untouched.
- **"Blur and bleed" were not tuned**: the reference rasterizer has no such effect, and adding one would need retrained
  renderers. What was tuned is how strokes start (size, opacity, marker length).
- **Tuning protocol added**: three tuning images, then a check on ten held-out images for every sweep winner that improved
  PSNR on all three tuning images; adopted when the held-out PSNR interval excludes zero and SSIM and LPIPS are not
  significantly worse. Applied the same way to every brush.
- **Defaults changed**: Week 5 runs used the reference starts; see the Build list for how to reproduce them.
- **Unexpected result**: the retrained light renderers of the two Bezier brushes are less faithful on random strokes but
  paint about 1.2 dB better (median 0.7 to 0.8 dB, about 6 dB on a flat-background image). Cause not established. Using them
  as the painter's default is a decision for the student (they are selected with `--renderer`).
- **Renderer training time depends on the brush** (tape 14 min, oil 51 min, watercolour 2.2 h, marker 3.3 h for 100
  epochs): the rasterizer of the Bezier brushes is a Python loop over 100 samples. A reference-length schedule for them
  (9 and 13 h) was not run.
- **Machine limits met again**: with Brave and Steam open, two PyTorch processes exhausted Windows commit memory and both
  died (resumed without loss); after other programs were closed (the commit charge fell from 34.7 to 21.3 GB), three ran
  together.
