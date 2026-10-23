# 7. Limitations and future work

*Draft (Week 9). Each limit names the section where it arises.*

## 7.1 What the measurements cannot show

**Small effects.** Most comparisons paint ten images with three seeds, so their 95 % intervals are about $\pm 0.1$ to
$\pm 0.2$ dB wide: an effect smaller than that, in either direction, cannot be told from zero (Sections 4.2.12 and 4.3.7). The
weight and regularization sweeps of the Sinkhorn term used ten and five images, and the brush defaults were tuned on three.

**Fidelity is not beauty.** PSNR, SSIM and LPIPS measure how close a painting is to the photograph. A painting is not meant to
be a copy, and the three metrics do not even agree on the order of the brushes (Section 4.3.4). The aesthetic survey is the only
measure of what people like, and it is small: at most a few dozen raters, five images, one fixed order of the paintings, and
some raters who have just used the application (Section 6.6).

**One machine for the timings.** Every time in this report was measured on one Windows desktop (RTX 3070, i5-13400F). Its
timings moved by a factor of two until Windows' power throttling was found and switched off in Week 7 (Section 4.4.7), so the
times of Weeks 5 and 6 are upper bounds and only their ratios should be read. The MacBook's `mps` and CPU profiles rest on one
device check (Table 4.4) until its benchmark runs. `torch.compile` could not be measured, because the Triton compiler is not
available for Windows in this PyTorch build. CUDA runs are reproducible to the last digit only with `--deterministic`.

**The light renderers.** Almost every experiment uses the authors' light renderers ($32 \times 32$ pixels). The full renderers
paint better in the one image tried (23.7 against 21.4 dB) but were not evaluated, and their optimization time varied by a
factor of five between two runs of the same painting (Section 4.4.7). The conclusions on stroke budgets, grids and brushes may
move with them, and so may the knee at 500 strokes, which also depends on the grid size $M = 5$ (Section 4.4.6).

**The evaluation set.** Thirty images, eleven from the reference repository and nineteen collected for this project, chosen to
cover portraits, landscapes, still lifes and textures (Section 5.2). It is enough to rank methods image by image, but not to
claim how the system behaves on photographs in general.

## 7.2 Open questions

**Why the retrained renderers paint better.** The watercolour and marker-pen renderers trained with this package's code are
less faithful to the rasterizer on random strokes than the authors', yet paint 1.2 dB better (Section 4.3.6). The likely
explanation, that fidelity on strokes drawn from the whole parameter space says little about the strokes the painter actually
uses, was not tested; a study of the renderers' error on the painter's own strokes would settle it. Until then the authors'
renderers stay the default, so that every result can be compared with the 2021 code.

**The Sinkhorn term under another loss.** The finding that the optimal-transport term does not help (Section 6.2) holds for the
pixel loss and photograph reconstruction. With a style loss, where strokes are not meant to land on their target, the term
might matter more; the style-transfer mode was a stretch goal and was not built.

**The renderers' training length.** The retrained renderers had a quarter of the reference's training; for the tape the full
schedule closed the gap and more (Section 4.3.6), and the same is plausible but untested for the other three brushes.

## 7.3 The application

The web application runs on the user's own computer and paints one picture at a time (Gradio's queue, Section 4.5.2): it is
not a service for many users. Making it reachable without a local server needs a hosted machine with a GPU, which costs money
or runs about twenty times slower on a free CPU (ROADMAP, optional plan). The interface depends on Gradio, whose behaviour
changes between versions; one such change left two result tabs blank until it was worked around (Section 4.5.2). The canvas is
512 pixels wide, so the paintings are meant for a screen, not for print.

## 7.4 Future work

1. Evaluate the full renderers and a finer grid, which may raise the useful number of strokes.
2. Explain, and then use, the better paintings of the retrained watercolour and marker-pen renderers.
3. A style-transfer mode on the stroke parameters (`losses/style_loss.py`), and vector export of the Bezier strokes
   (`export/svg_exporter.py`).
4. A larger aesthetic study with more images and raters, and a randomized order per rater.
5. A hosted version of the application with its own web page (ROADMAP, optional plan).
