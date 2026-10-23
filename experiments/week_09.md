# Week 9: Export, evaluation, and stretch goals

Status mirror of `ROADMAP.md`. Started on the desktop on 2026-10-16, ahead of the calendar week (26 to 30 Oct). The prose is in
`report/chapters/04_implementation.md`, section 4.6.

## Build

- [x] `neural_painter/export/video_recorder.py`: `frames`, `write_timelapse` (`.mp4` H.264 / `.gif` one palette, stride, fps, hold,
  size, shape), `write_timelapses` (the app's and `run_paint.py`'s settings `MP4_SETTINGS`, `GIF_SETTINGS`).
- [x] `pipeline/painter_engine.load_strokes_npz` (the package's and the original's layout); stroke files record `canvas`.
- [x] `run_paint.py --timelapse`; `scripts/make_timelapse.py` (`--strip` for print); `scripts/timelapse_cost.py`.
- [x] Web page: time-lapse tab (plays the MP4), downloads PNG / JPG / MP4 / GIF / `.npz`, replay from the blank canvas.
- [x] Tests: `tests/test_video_recorder.py` (4), the app test and the `run_paint` test extended; 263 pass (259 before).

## Measure

| Check | Where | Result |
| :--- | :--- | :--- |
| Time and size of the time-lapses, 4 brushes x 10 paintings (495 strokes, progressive pipeline, seed 0) | `2026-10-16_measure_timelapse_cost` | about 2 s per file (1.7 to 2.3 s); MP4 0.22 to 0.41 MB, GIF 0.25 to 0.65 MB |
| Last frame against the painting (oil, `apple`, 495 strokes) | ad hoc check, 2026-10-16 | mean absolute difference 2.3 (MP4) and 3.8 (GIF) grey levels of 255 |
| The page in Chrome, both languages | `scripts/app_screenshots.py` | every step passes; the MP4 plays in the time-lapse tab; the five download buttons are enabled; the Painting tab shows the painting (fixed 16 Oct: it was empty, see ROADMAP Week 9 notes) |
| Free 2-CPU hosting estimate (optional plan) | 2026-10-15, 2 threads of the i5-13400F | oil, 500 strokes: 260 s best, 110 s fast |

## Aesthetic survey

- [x] Material: `scripts/make_survey.py` -> `report/notes/aesthetic_survey/` (20 paintings, `paintings.pdf`, `answer_sheet_vi.pdf`,
  `key.csv`); protocol `report/notes/aesthetic_survey.md`; `scripts/survey_summary.py`; `tests/test_survey.py`.
- [ ] Ratings (student, at least 10 raters, from 21 Oct), then `survey_summary.py` and report section 6.6.

## Write

- [x] Report section 4.6 (Figures 4.20, 4.21, Table 4.37); section 4.5 updated; screenshots retaken.
- [~] Experiments chapter drafted (`report/chapters/06_results.md`, pending tables marked); Limitations draft
  (`report/chapters/07_limitations.md`).

## Open

- [ ] Full evaluation against the original at equal stroke budgets; final hardware benchmark (`mps` on the MacBook); aesthetic
  survey (10 raters); fixes from the usability test; stretch goals.
