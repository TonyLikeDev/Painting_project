# Week 8: User interface

Status mirror of `ROADMAP.md`. Built on the desktop on 2026-10-15, ahead of the calendar week (19 to 23 Oct), so that the usability
test can be held inside the week. The prose is in `report/chapters/04_implementation.md`, section 4.5.

## Build

- [x] `neural_painter/app/gradio_app.py`: the web app. Input (upload, paste, six sample photos), brush, strokes, grid mode, speed,
  canvas colour, photo fit, seed; Paint and Stop; status line with progress bar; live preview, before / after slider (shown by
  itself when the painting is done), the painting, a replay slider in strokes; PNG and `.npz` downloads; results saved under
  `output/<date>_app_<photo>_<brush>/`. Vietnamese / English switch.
- [x] `pipeline/progressive_painter.py`: `PIPELINE_SETTINGS` and `FAST_SETTINGS` shared by the app and `run_paint.py` (new `--fast`),
  `paint_image(on_stroke=...)` for replay (and Week 9 time-lapse) frames.
- [x] `Start_App.bat`, `start_app.command`, `README.md`.
- [x] `scripts/app_screenshots.py` (headless Chrome over the DevTools protocol, both languages), `scripts/usability_summary.py`.
- [x] Tests: `tests/test_gradio_app.py` (5) and a usability-summary test in `tests/test_scripts.py`; 259 pass (253 before).

## Measure

| Check | Where | Result |
| :--- | :--- | :--- |
| The page in a real browser, English and Vietnamese | `scripts/app_screenshots.py`; screenshots `report/figures/app_*.png` | every step passes: language switch both ways, upload, Paint, live progress line, finished painting, automatic switch to the comparison, replay slider, damaged file refused with a message |
| Painting time in the app (watercolour, `apple`, 500 strokes, GPU) | status line of the runs above | 16 to 19 s of optimization and rasterization (about 14 s without the page, Table 4.31) |
| Usability test, 5 to 8 users | `experiments/<date>_usability/` (to come) | prepared: protocol, Vietnamese sheet and PDF, CSV templates in `experiments/usability_test_template/`, summary script |

The paintings made by these checks are in `output/` (git-ignored).

## Write

- [x] Report section 4.5 (4.5.1 to 4.5.4), Figures 4.17 to 4.19; `report/notes/usability_test.md`, `report/notes/usability_test_phieu_vi.md`
  (and `.pdf`); bibliography: Brooke 1996 (SUS), Sauro 2011.
- [ ] The user test summary and the fixes (after the sessions).
