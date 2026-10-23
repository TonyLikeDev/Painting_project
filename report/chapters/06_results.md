# 6. Experiments: results

Chapter 4 reports every experiment where the part of the system it tests is built, with its protocol, its tables and its
limits. This chapter collects the answers: first the evaluation of the finished system against the 2021 implementation on the
whole evaluation set (Week 9), then one section per research question of Section 1, and finally what people think of the
paintings and of the application. Unless a table says otherwise, every painting uses the authors' light renderers, the
evaluation set of Section 5.2 at 512 pixels, 500 strokes (495 painted: the reference's equal split of $\lfloor 500/55 \rfloor$
strokes per block over the grids $1 \times 1$ to $5 \times 5$) and the pixel loss, and every difference is paired per image.

## 6.1 The finished system against the original implementation

*Protocol.* `scripts/compare_with_original.py --mode progressive` runs the reference's unmodified `demo_prog.py` (its best mode,
coarse to fine up to $5 \times 5$) and this package's pipeline as `run_paint.py` and the web page run it (progressive, cosine
learning rate from 0.004, the Week 6 start ranges, area-averaged targets), with the same renderer, canvas colour (white for oil
and watercolour, black for marker pen and tape), stroke budget and images, and scores both paintings with the same code
(experiments `2026-10-17_check_full_eval_<brush>`, 30 images, 4 brushes, seed 0, 240 paintings). Both processes are opted out
of Windows' power throttling and run one at a time on the desktop's RTX 3070. The spread over seeds is small next to the
differences between images (0.04 to 0.09 dB between repeat means for oil, Table 4.26), so one seed per image was painted and the
statistics are over the 30 images.

*pending: Table 6.1 (per brush: PSNR, SSIM, LPIPS and wall time of the original and of ours, paired difference with its 95 %
interval and the share of images won) and Figure 6.1 (one image per brush, original against ours).*

## 6.2 RQ1: does the Sinkhorn loss help?

No. Under the protocol of Section 4.2.5 the optimal-transport term changes the finished painting by $-0.02$ to $+0.05$ dB at the
reference weight 0.1 for all four brushes (Tables 4.6 and 4.22), by at most $+0.02$ dB at any weight from 0.1 to 10 and by
$-0.10$ dB at 100 (Table 4.8), and LPIPS sees no effect wherever it can resolve one; it does not make up for a worse start either
(Section 4.2.9). At the reference weight its gradient is 0.03 to 0.09 % of the pixel loss's (Table 4.9), so the term is practically
switched off, and it costs 2.3 to 2.7 times the optimization time. It is therefore not part of the pipeline. The design choices
that do matter are elsewhere (Table 4.15): the grid ($+5.7$ dB over the full image), the dilate-and-erode step of the reference
($+2.7$ dB), starting new strokes from the error map ($+0.9$ dB) and how the target is shrunk ($+0.3$ dB).

## 6.3 RQ2: progressive grid, fixed grid or full image?

The progressive grid. Painting the whole image at once is far behind the fixed $5 \times 5$ grid ($-5.7$ dB with the fixed
protocol, Table 4.13; $-6.5$ dB in Week 7, Table 4.29) and takes 5 to 11 times as long. The progressive mode beats the fixed grid
for every brush (Table 4.30): by 0.26 dB for oil and by 1.1 to 2.1 dB, on every image, for watercolour, marker pen and tape, whose
thin or translucent strokes leave the fixed grid's block borders bare. It takes about twice the time of the fixed grid on the GPU
(12 against 5.7 s) and 1.2 times on the CPU, with half the memory. The gain comes from the coarse-to-fine schedule, not from its
larger number of optimizer steps (Section 4.4.4). A cosine learning-rate schedule adds 0.07 to 0.32 dB for every brush at no cost.

## 6.4 RQ3: how do quality and time grow with the stroke budget?

Quality saturates and time does not (Table 4.31, Figure 4.16). From 55 to 275 strokes every brush gains about 3 dB, from 275 to
495 a further 0.3 to 0.8 dB, and from 495 to 990, at 1.4 times the optimization time, at most 0.1 dB (not significant) for oil,
watercolour and marker pen; only the tape still gains 0.5 dB. The optimization time grows from 9 s at 55 strokes to 12 s at
495 and 17 s at 990 on the RTX 3070, the memory in proportion to the strokes per block (0.04 to 0.34 GB). With the light
renderers and $M = 5$, 500 strokes are close to the useful maximum, which is why the application defaults to them.

## 6.5 RQ4: renderer fidelity and consumer hardware

*Renderers.* The renderers retrained with this package's code are within the plan's 1 dB of the authors' on 1,000 held-out
strokes: $-0.22$ (oil), $-0.45$ (watercolour), $-0.64$ (marker pen) and $-0.80$ dB (tape) after 100 epochs, a quarter of the
reference's training, and $+0.19$ dB for the tape with the reference's 400 epochs (Tables 4.3 and 4.23). Painting with them gives
an unexpected result: the less faithful watercolour and marker-pen renderers paint 1.2 dB better than the authors' (medians
0.7 to 0.8 dB, LPIPS better on every image), the oil renderer paints 0.15 dB worse, and the tape renderer the same (Table 4.24);
why was not established (Section 4.3.6).

*Hardware.* *pending: the final benchmark table (Table 6.2): `scripts/benchmark.py` on the final code for `cuda`, `cpu` and
`mps`.* The first pass (Table 4.34) measured a progressive painting of 500 strokes at 12 s on the RTX 3070 and 130 s on the
i5-13400F with 10 threads, every setting with the light renderers below 0.4 GB, and the full renderers at up to 6.8 GB.

## 6.6 Brush materials: closeness to the photograph and what people like

By PSNR the brushes rank tape, oil, watercolour, marker pen on the fixed grid (Table 4.21) and tape, watercolour, marker pen, oil
in the progressive mode, where the three thin or translucent brushes gain most (Table 4.30); SSIM and LPIPS order them
differently again. These metrics measure closeness to the photograph, which a painting is not meant to maximize. The aesthetic
survey (`report/notes/aesthetic_survey.md`) therefore asks people to rate 20 paintings, the four brushes on five images, from 1
(not beautiful at all) to 5 (very beautiful), without the photographs and without knowing which brush made which.

*pending: Table 6.3 (mean score per brush with its 95 % interval, Friedman test, pairwise tests) and the correlation of the
paintings' mean scores with their PSNR, SSIM and LPIPS (`scripts/survey_summary.py`).*

## 6.7 Usability of the application

*pending: the usability test with 5 to 8 participants (21 to 22 Oct): task completion and times, the SUS score, and the problems
found and fixed (Section 4.5.4, `scripts/usability_summary.py`).*

## 6.8 Summary

*pending: one table, one row per research question and evaluation, with the answer and the evidence.*
