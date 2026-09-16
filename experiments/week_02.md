# Week 2: Theory and system architecture

Status mirror of ROADMAP.md (the roadmap is authoritative). Date 2026-09-16.

## Build
- [x] `neural_painter/` package skeleton, `pyproject.toml`, four YAML brush configs — `neural_painter/configs/`
- [x] stub modules with documented planned interfaces for `models/`, `losses/`, `pipeline/`, `export/`, `app/`, `scripts/`
- [ ] system architecture diagram — `report/figures/system_architecture.svg`
- [ ] renderer architecture diagram — `report/figures/renderer_architecture.svg`

## Measure
- [x] one-page notes on `renderer.py`, `networks.py`, `loss.py`, `pytorch_batch_sinkhorn.py`, `painter.py`, `utils.py` — `report/notes/original_code/`

## Write
- [x] theory chapter draft (about 7,200 words) — `report/chapters/02_theory.md`
- [x] Method section skeleton — `report/paper/method_skeleton.md`

## Exit criteria
- [x] package skeleton imports cleanly (`tests/test_package_imports.py`)
- [ ] two diagrams saved as SVG/PNG
- [x] theory draft of at least 8 pages

## Notes and deviations
- Layout deviation from PLAN.md section 3: `pyproject.toml`, `tests/` and `scripts/` sit at the repository
  root and `neural_painter/` is the importable package (standard Python layout, `pip install -e .` works).
  `configs/` and `assets/brushes/` live inside the package so they ship with it.
- `losses/pixel_loss.py`, `core/morphology.py` and `core/differentiable_canvas.py` were implemented rather
  than stubbed because each is under 40 lines; they are still tested in the week that uses them.
