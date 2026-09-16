# Week 3: Image input and preprocessing module

Status mirror of ROADMAP.md (the roadmap is authoritative). Date 2026-09-16.

## Build
- [x] `core/image_io.py` — load JPG/PNG (EXIF-aware), sample picker, crop, resize (stretch / crop / pad), normalize, tensor, `preprocess`
- [x] `core/stroke_models.py` — `BrushSpec` from YAML, typed strokes, validation, clipping
- [x] `core/procedural_rasterizer.py` — pixel-exact port of the original rasterizer, four brushes, seeded sampling
- [x] `core/grid.py` — split / merge, `img2patches` / `patches2img`, `block_to_global`, progressive schedule helper
- [x] `tests/test_rasterizer.py`, `test_image_io.py`, `test_grid.py`, `test_stroke_models.py`, `test_package_imports.py` — 81 tests pass

## Measure
- [ ] evaluation set frozen — `data/eval_set/`, `experiments/dataset.md` (`scripts/freeze_dataset.py`)

## Write
- [ ] dataset and preprocessing subsection — `report/chapters/05_experiments_setup.md`, figure `report/figures/preprocessing_pipeline.png`

## Exit criteria
- [x] tests pass
- [x] a random stroke of each brush renders through the procedural rasterizer, identical to the original code at 32 and 128 px, train and inference modes
- [ ] dataset frozen

## Notes and deviations
- The rasterizer port is verified pixel-for-pixel against the original `renderer.py` (25 random strokes per
  brush / size / mode, plus a 40-stroke canvas sequence and the oil texture warp). This is what lets the
  pretrained renderers be reused unchanged in Week 4.
- Self-collected photos are not yet in the set; the freeze script re-runs with `--force` to add them and
  keeps the repo images byte-identical. Deadline: before the Week 5 ablations.
