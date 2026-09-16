# CLAUDE.md - working notes for AI assistants on this repository

Read this first. `AGENTS.md` is a pointer to this file.

## What this project is

A university course project (VNUK, University of Danang, student Nguyen Le Hoang,
academic year 2026-2027) that re-implements and modernizes **Stylized Neural
Painting** (Zou et al., CVPR 2021): turn a photo into a stroke-based painting by
optimizing stroke parameters through a *differentiable neural renderer*, with
four brush materials (oil, watercolour, marker pen, coloured tape), a progressive
coarse-to-fine grid, a Gradio web UI and MP4/GIF time-lapse export. The course
deliverables are working software, a 40 to 60 page report and a demo video; a
research paper draft is assembled from the same weekly outputs.

## The three documents that govern the work

| File | Role | Edit it when |
| :--- | :--- | :--- |
| `RESEARCH_PLAN.md` | The plan: 10-week schedule, research questions, metrics, experiment rules | Only when the plan itself changes (scope, decisions) |
| `PLAN.md` | Original engineering design (architecture, maths, package layout) | Rarely; superseded by RESEARCH_PLAN where they disagree |
| `ROADMAP.md` | Live status board with checkboxes per week | **Every time an item is finished, in the same commit** |

The proposal PDF `Do an co so Nguyen Le Hoang.pdf` (Vietnamese) is the course
contract; corrections to it are drafted in `report/proposal_corrections.md`.

## Repository layout

```
neural_painter/            the new package (PyTorch 2.x, cuda / mps / cpu)
  configs/*.yaml           one file per brush: parameter names, dims, defaults
  core/                    stroke_models, procedural_rasterizer, grid, image_io, device, morphology, differentiable_canvas
  models/ losses/ pipeline/ export/ app/   filled in Weeks 4 to 9 (see ROADMAP.md)
  assets/brushes/          the four oil-brush texture PNGs (copied from the original repo)
tests/                     pytest suite (run: venv/Scripts/python -m pytest)
scripts/                   runnable entry points and experiment runners
data/eval_set/             frozen evaluation images (Week 3); never change after freezing
experiments/               one folder or md file per experiment; results tables live here
  original_repo_modern_torch.patch   diff that makes the 2021 code run on torch>=2.6 and mps, plus --seed
report/                    everything that goes into the report / paper
  chapters/ paper/ notes/original_code/ figures/ bibliography.md related_work_outline.md
stylized-neural-painting/  the ORIGINAL 2021 code, a nested git checkout, git-ignored by this repo
venv/                      local virtual environment (git-ignored)
```

## Environment facts (desktop, Windows 11)

- Interpreter: `venv/Scripts/python.exe` (Python 3.11.9, torch 2.11.0+cu128). In
  Git Bash `python` resolves to an MSYS Python without torch; always call the venv
  interpreter explicitly.
- GPU: RTX 3070 8 GB (profile `cuda`); CPU: i5-13400F (profile `cpu`, force with
  `CUDA_VISIBLE_DEVICES=""`). The MacBook M4 is the `mps` profile.
- The original code runs from inside `stylized-neural-painting/` (relative paths
  to `./brushes` and `./checkpoints_G_*`). It is patched in place; if the checkout
  is ever reset, re-apply `patch -p1 < ../experiments/original_repo_modern_torch.patch`.
- No `pdftoppm`; read PDFs with `pypdf` (installed in the Windows Store Python, not the venv).
- The Bash tool chokes on very long multi-heredoc commands; write big files with the Write tool.

## Conventions that must not drift

- **Stroke parameters** are always in `[0, 1]`, layout defined once in the YAML
  configs and exposed through `neural_painter.core.stroke_models.BrushSpec`. Dims:
  oil 12 (5/6/1), tape 9 (5/3/1), watercolour 15 (8/6/1), marker 12 (8/3/1).
  Bezier control points are stored *relative to the chord*.
- **Images** are RGB `float32 (H, W, 3)` in `[0, 1]` as numpy, `(1, 3, H, W)` as
  tensors. The original code's BGR usage stops at the loader.
- **The procedural rasterizer is a pixel-exact port** of the original
  `renderer.py`; keep it that way (the pretrained renderers were trained on it).
  Any deliberate deviation must be behind a flag and noted in the report.
- **Devices are explicit** (`core.device.get_device`), never module-level globals.
- **Seeds**: every experiment fixes `seed`; numbers in tables come from seeded runs
  (mean and std over 3 seeds for headline numbers).
- **Experiment logging** (RESEARCH_PLAN 3.6): each run gets its own folder under
  `experiments/` with config, results CSV, final PNG, stroke `.npz`, device and
  commit hash. Never overwrite an experiment folder. Frames, MP4 and checkpoints
  are not committed.
- **Writing goes to `report/`** in English Markdown with LaTeX math, one file per
  chapter or note, so the report can be assembled by concatenation.

## How to work here

1. Open `ROADMAP.md`, take the next open item of the current week.
2. Build it, test it (`venv/Scripts/python -m pytest -q`), record any measurement
   under `experiments/`, and put any prose under `report/`.
3. Tick the item in `ROADMAP.md`, add a change-log line, run
   `venv/Scripts/python.exe scripts/roadmap_progress.py` to refresh the progress
   bars at the top of the roadmap, and commit with a message that names the week
   and the item.
4. If something in the plan turns out to be wrong, say so in `ROADMAP.md`
   under the week's "Notes and deviations" and only then change `RESEARCH_PLAN.md`.

## Useful commands

```
venv/Scripts/python.exe -m pytest -q                       # test suite
venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available())"
cd stylized-neural-painting && ../venv/Scripts/python.exe demo_prog.py --img_path ./test_images/apple.jpg \
  --canvas_color white --max_m_strokes 500 --max_divide 5 --renderer oilpaintbrush \
  --renderer_checkpoint_dir checkpoints_G_oilpaintbrush_light --net_G zou-fusion-net-light \
  --disable_preview --seed 0 --output_dir ./output_cuda_oilpaintbrush      # original baseline
```
