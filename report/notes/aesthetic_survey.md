# Aesthetic survey (Week 9): protocol

**Question.** Which brush material do people find most beautiful, and do the fidelity metrics (PSNR, SSIM, LPIPS) predict it?
The metrics measure closeness to the photograph; a painting is not meant to be a copy, so only people can answer this
(RESEARCH_PLAN 3.3: aesthetic score 1 to 5, at least 10 raters, four brushes on five images).

**Material** (`scripts/make_survey.py`, in `report/notes/aesthetic_survey/`):

- 20 paintings: four brushes (oil, watercolour, marker pen, coloured tape) on five images of the evaluation set (portrait `jay`,
  landscape `yosemite`, still life with a cereal bowl, yellow daisies, balloon and chair), painted with the pipeline's settings
  (progressive, cosine learning rate from 0.004, 495 strokes, seed 0; the brush comparison of Week 7). The photographs are not shown.
- One fixed random order (seed 9) in which no two neighbouring paintings share an image or a brush; the paintings carry only
  their numbers, so the rater does not know which brush made which.
- `paintings.pdf` (four paintings per A4 page) and `paintings/01.png` to `20.png`; `answer_sheet_vi.pdf`, one page per rater;
  `key.csv`, the number of each painting with its image, brush and metrics (keep it away from the raters).

**Procedure** (about 5 minutes per rater).

1. Show the paintings in order, on a screen or one colour print of `paintings.pdf` that every rater reuses.
2. The rater circles one score per painting on the answer sheet: "how beautiful do you find this painting", 1 (not beautiful at
   all) to 5 (very beautiful), judged as a painting, not by its likeness to a photograph; and ticks whether they have studied art
   or work in it.
3. No names: the sheet carries a rater code (R1, R2, ...). Raters see no scores or brush names until they are done.

The participants of the usability test (21 to 22 Oct) can rate the paintings after their session; they have just used the app,
which the report should mention. Further raters can answer online: a Google Form with the 20 PNG files, one linear-scale
question (1 to 5) per painting titled `Tranh 1` to `Tranh 20`, one yes / no question titled `art`; its CSV export is read as is.

**Results.** Copy `experiments/aesthetic_survey_template/ratings.csv` into a new folder `experiments/<date>_aesthetic_survey/`,
one row per rater (`rater`, `art` as `co` / `khong`, the 20 scores), then run

```
venv/Scripts/python.exe scripts/survey_summary.py experiments/<date>_aesthetic_survey/ratings.csv
```

which writes `summary.md` beside it: the mean score per brush with its 95 % confidence interval, the Friedman test over the brushes
and Holm-corrected pairwise Wilcoxon tests, the mean per image and brush, and the Spearman correlation of the paintings' mean
scores with PSNR, SSIM and LPIPS. With fewer than 10 raters the report calls it a pilot study (RESEARCH_PLAN section 6).
