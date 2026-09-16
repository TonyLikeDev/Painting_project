# Week 1: Literature review and proposal finalization

Status mirror of ROADMAP.md (the roadmap is authoritative). Dates 2026-09-05 (MacBook) and 2026-09-16 (desktop).

## Build
- [x] venv on both machines (MacBook: torch 2.14 / mps; desktop: torch 2.11.0+cu128 / RTX 3070) — `experiments/hardware.md`
- [x] `stylized-neural-painting` cloned, 8 checkpoints downloaded, `demo_prog.py` runs on apple + oil
- [x] git repository with remote (`origin`), both machines on the same commit
- [x] device details recorded — `experiments/hardware.md`
- [x] original repo patched for `mps`, torch>=2.6 (`weights_only`), opencv-python>=5 (numpy scalar colours) and a `--seed` flag — `experiments/original_repo_modern_torch.patch`

## Measure
- [x] MacBook `mps` and `cpu` rows (apple, oil, 500 strokes) — `experiments/week1_baseline.md`
- [ ] desktop `cuda` and `cpu` rows, 4 brushes, seed 0 — `experiments/2026-09-16_week1_baseline_desktop/results_{cuda,cpu}.csv` (running)

## Write
- [x] annotated bibliography — `report/bibliography.md`
- [x] Related Work outline — `report/related_work_outline.md`
- [x] proposal Section 4.1 correction drafted — `report/proposal_corrections.md`
- [ ] proposal corrected and submitted (user)

## Exit criteria
- [x] baseline demo runs end to end
- [ ] proposal submitted
- [x] bibliography file exists

## Notes and deviations
- The original code needs one more patch than the plan anticipated: OpenCV 5.0 rejects NumPy float32
  scalars as colour arguments in `cv2.circle` / `cv2.fillPoly`, which breaks the watercolour, marker and
  tape rasterizers (oil uses `warpAffine` and was unaffected, which is why the MacBook oil run passed).
  Found by the pixel-parity test in `tests/test_rasterizer.py`; fixed in `renderer.py` by casting to `float`.
- Baseline wall times include model load and the final 500-frame PNG/MP4 render, as on the MacBook.
