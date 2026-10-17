# Neural painter

Turn a photo into a painting made of brush strokes: oil paint, watercolour, marker pen or coloured tape. A re-implementation and
modernization of *Stylized Neural Painting* (Zou et al., CVPR 2021) by Nguyen Le Hoang (VNUK, University of Danang), a course
project. The stroke parameters are optimized through a differentiable neural renderer, coarse to fine, and the result is drawn
with the real brush textures.

## Quick start

1. Python 3.11 and the project's virtual environment (`venv/`, see `CLAUDE.md` for the desktop set-up). On a new machine:

   ```
   python -m venv venv
   venv\Scripts\python.exe -m pip install -e .[app]        (Windows; on macOS: venv/bin/python -m pip install -e '.[app]')
   ```

   For an NVIDIA GPU install the CUDA build of PyTorch first (https://pytorch.org). The pretrained renderers of the 2021 code are
   expected in `stylized-neural-painting/checkpoints_G_*_light/` (see `CLAUDE.md`).

2. **The app:** double-click `Start_App.bat` (Windows) or `start_app.command` (macOS; once: `chmod +x start_app.command`), or run
   `python -m neural_painter.app.gradio_app`. The page opens in the browser at http://127.0.0.1:7860: choose a photo, a brush and
   press **Paint**. Paintings are saved under `output/`, with a time-lapse video (MP4 and GIF) of the painting being made.

3. **One command:** `python scripts/run_paint.py photo.jpg --brush watercolor --strokes 500` writes the painting, the photo it was
   scored against, the strokes and the settings to a new folder under `output/`. `--fast` paints in less than half the time;
   `--timelapse` adds the MP4 and the GIF. `python scripts/make_timelapse.py output/<folder>` makes them for a saved painting.

## Repository

| Path | What |
| :--- | :--- |
| `neural_painter/` | the package: brush models, renderers, painter, web app |
| `scripts/` | command-line tools and the experiment runners |
| `tests/` | `python -m pytest -q` |
| `experiments/` | one folder per experiment (tables committed, paintings not) |
| `report/` | the course report, chapter by chapter, and its figures |
| `ROADMAP.md`, `RESEARCH_PLAN.md` | status and plan |

Licence: CC BY-NC-SA 4.0 (`pyproject.toml`). The original 2021 code, used as the reference and for its pretrained renderers, is
released under CC0 1.0.
