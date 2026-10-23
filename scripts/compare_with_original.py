"""Paint evaluation images with the original 2021 code and with this package, and score both the same way (Weeks 5 and 7).

Both run one algorithm of the original: ``--mode fixed`` is ``demo.py`` (a ``--grid`` x ``--grid`` grid), ``--mode progressive`` is
``demo_prog.py`` (grids ``1 x 1`` to ``--grid`` x ``--grid`` in turn, Week 7), with the pixel loss only, the pretrained light
renderer of ``--brush`` (oil by default; Week 9 added the other three), the brush's default canvas (white for oil and watercolour,
black for marker and tape) and ``--strokes`` strokes. The original is run as a subprocess from its own checkout (``demo.py`` has no
seed flag, so its repeats are unseeded and the ``seed`` column is only the repeat number; ``demo_prog.py`` got a ``--seed`` flag in
our patch and is seeded with the repeat number, which does not make its draws equal to ours); ours runs with the same repeat
numbers as seeds. The finished 512 px paintings are scored with the same ``metrics.image_metrics``. One row per run goes to
``results.csv`` in the layout of ``run_ablation.py`` (plus the wall time of the whole process), so
``summarize_ablation.py --baseline original`` gives the paired comparison of the two implementations.

A third config, ``ours_linear``, is our engine with the original's way of shrinking the target to the blocks (OpenCV's default
bilinear resize instead of area averaging), the original's starting stroke sizes (Week 6 enlarged the default) and, for the
progressive mode, the centred RMSprop of ``demo_prog.py``, so that it differs from the original in implementation only; ``ours`` is
this package's defaults. ``pipeline`` (Week 9) is what ``run_paint.py`` and the web page paint with, ``PIPELINE_SETTINGS`` (cosine
learning rate from 0.004); ``--configs`` picks which of the four run (default: the first three, as in Weeks 5 and 7).

The original writes one PNG per stroke and a video next to its result; they are deleted after the final image has been read.
The script is resumable with ``--dir`` and ``--max-minutes``, like ``run_ablation.py``; a config added later (``ours_linear`` was)
is simply the next missing runs of an existing folder.

Usage: venv/Scripts/python.exe scripts/compare_with_original.py --name oil_fixed_grid [--mode fixed|progressive] [--every 6] [--repeats 3]
       [--grid 5] [--strokes 500] [--device auto] [--max-minutes 8] [--dir experiments/<existing folder>]
       [--brush oil|watercolor|marker|tape] [--configs original pipeline]
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neural_painter.core.device import device_summary, disable_power_throttling, get_device, set_deterministic  # noqa: E402
from neural_painter.core.image_io import load_image, save_image  # noqa: E402
from neural_painter.core.stroke_models import canonical_brush_name, get_brush  # noqa: E402
from neural_painter.models.neural_renderer import load_original_checkpoint, original_checkpoint_path  # noqa: E402
from neural_painter.pipeline import metrics  # noqa: E402
from neural_painter.pipeline.painter_engine import PainterConfig, config_dict, save_strokes_npz  # noqa: E402
from neural_painter.pipeline.progressive_painter import PIPELINE_SETTINGS, paint_image  # noqa: E402
from neural_painter.pipeline.train_renderer import EXPERIMENT_ROOT, commit_info, lower_process_priority, new_experiment_dir  # noqa: E402
from neural_painter.core.grid import grid_levels, strokes_per_block  # noqa: E402
from run_ablation import FIELDS, check_resume, csv_fields, eval_images, read_done, start_ranges  # noqa: E402

ORIGINAL = ROOT / "stylized-neural-painting"
CONFIGS = ("original", "ours", "ours_linear", "pipeline")  # "ours_linear": our engine with the original's settings; "pipeline": PIPELINE_SETTINGS
ALL_FIELDS = FIELDS + ["wall_s"]
ORIGINAL_SETTINGS = {"resize": "linear", "init_size_range": (0.1, 0.25)}  # what "ours_linear" takes from the original
ORIGINAL_OPTIMIZER = {"fixed": {}, "progressive": {"centered": True}}  # demo_prog.py uses RMSprop(centered=True), demo.py the plain one


def original_device() -> str:
    """The device the original demo will pick: ``cuda:0`` when PyTorch sees a GPU (an empty ``CUDA_VISIBLE_DEVICES`` hides it)."""
    hidden = os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    return "cuda:0" if torch.cuda.is_available() and not hidden else "cpu"


def original_command(image_path: Path, run_dir: Path, grid: int, strokes: int, mode: str = "fixed", seed: int = 0,
                     brush: str = "oilpaintbrush", canvas: str = "white") -> list[str]:
    """``demo.py`` (fixed grid, unseeded) or ``demo_prog.py`` (progressive, ``--max_divide grid``, seeded) with the comparison's settings."""
    script, grid_flag = ("demo_prog.py", "--max_divide") if mode == "progressive" else ("demo.py", "--m_grid")
    argv = [
        script, "--img_path", image_path.resolve().as_posix(), "--canvas_color", canvas, "--max_m_strokes", str(strokes),
        grid_flag, str(grid), "--renderer", brush, "--renderer_checkpoint_dir", f"checkpoints_G_{brush}_light",  # relative to the checkout
        "--net_G", "zou-fusion-net-light",
        "--disable_preview", "--output_dir", run_dir.resolve().as_posix(), *(["--seed", str(seed)] if mode == "progressive" else []),
    ]
    # the demo runs unmodified, but opted out of Windows' power throttling like our side, so that the wall times compare
    start = (f"import sys, runpy; sys.path.insert(1, {str(ROOT)!r}); from neural_painter.core.device import disable_power_throttling; "
             f"disable_power_throttling(); sys.argv = {argv!r}; runpy.run_path({script!r}, run_name='__main__')")
    return [sys.executable, "-c", start, *argv]


def run_original(image_path: Path, run_dir: Path, grid: int, strokes: int, mode: str = "fixed", seed: int = 0,
                 brush: str = "oilpaintbrush", canvas: str = "white") -> tuple[np.ndarray, float]:
    """Paint ``image_path`` with the original demo; returns its final 512 px image and the wall time of the process."""
    run_dir.mkdir(parents=True, exist_ok=True)
    command = original_command(image_path, run_dir, grid, strokes, mode, seed, brush, canvas)
    started = time.perf_counter()
    done = subprocess.run(command, cwd=ORIGINAL, capture_output=True, text=True)
    wall = time.perf_counter() - started
    (run_dir / "log.txt").write_text(done.stdout + done.stderr, encoding="utf-8")
    if done.returncode:
        raise RuntimeError(f"the original demo failed on {image_path.name} (exit {done.returncode}); see {run_dir / 'log.txt'}")
    final = load_image(run_dir / f"{image_path.stem}_final.png")
    for leftover in [*run_dir.glob(f"{image_path.stem}_rendered_stroke_*.png"), *run_dir.glob("*.mp4")]:
        leftover.unlink()
    return final, wall


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="original_vs_ours", help="experiment folder suffix (a new folder <date>_check_<name> is made)")
    ap.add_argument("--dir", type=Path, default=None, help="resume this existing experiment folder instead of making a new one")
    ap.add_argument("--mode", default="fixed", choices=["fixed", "progressive"], help="demo.py (fixed grid) or demo_prog.py (progressive)")
    ap.add_argument("--images", default="all")
    ap.add_argument("--every", type=int, default=6, help="use every K-th image of the selection (default 6: 5 of the 30 images)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--strokes", type=int, default=500)
    ap.add_argument("--device", default="auto", help="device of our run; the original picks cuda when it is available")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="start no new run after this many minutes (0 = no limit)")
    ap.add_argument("--low-priority", action="store_true")
    ap.add_argument("--deterministic", action="store_true", help="fix cuDNN's algorithm choice for our runs (the original cannot be seeded)")
    ap.add_argument("--brush", default="oil", help="oil, watercolor, marker or tape (the authors' light renderer of each)")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS[:3]), choices=CONFIGS, help="which implementations to run")
    args = ap.parse_args(argv)
    disable_power_throttling()
    if args.low_priority:
        lower_process_priority()
    set_deterministic(args.deterministic)

    device = get_device(args.device)
    brush = canonical_brush_name(args.brush)
    spec = get_brush(brush)
    canvas = spec.default_canvas_color
    renderer = load_original_checkpoint(original_checkpoint_path(brush, True), brush, True)
    grids = grid_levels(args.mode, args.grid)
    per_block = strokes_per_block(args.strokes, grids)
    if per_block < 1:
        raise SystemExit(f"{args.strokes} strokes are too few for the grids {grids}")
    reference = {**ORIGINAL_SETTINGS, **ORIGINAL_OPTIMIZER[args.mode]}
    images = eval_images(args.images)[:: max(1, args.every)]

    if args.dir is not None:
        out = args.dir if args.dir.is_absolute() else ROOT / args.dir
        if not out.is_dir():
            raise SystemExit(f"{out} does not exist; leave out --dir to start a new experiment")
        recorded = (yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8")) or {}).get("brush") if (out / "config.yaml").is_file() else None
        if recorded not in (None, brush):
            raise SystemExit(f"{out.name} holds {recorded} paintings, not {brush}")
        check_resume(out, brush)
    else:
        out = new_experiment_dir(EXPERIMENT_ROOT, args.name, kind="check")
    results_path = out / "results.csv"
    if not (out / "config.yaml").exists():
        info = {"created": datetime.now().isoformat(timespec="seconds"), **commit_info(), **device_summary(device), "brush": brush, "canvas": canvas,
                "mode": args.mode, "grid": args.grid, "strokes_per_block": per_block, "total_strokes": per_block * sum(m * m for m in grids),
                "images": [p.stem for p in images], "repeats": args.repeats, "cudnn_deterministic": bool(args.deterministic),
                "original": ("demo_prog.py (seeded with the repeat number)" if args.mode == "progressive" else "demo.py (unseeded)")
                + f" of stylized-neural-painting, pixel loss, light {brush} renderer, {canvas} canvas", "configs": args.configs,
                "ours": config_dict(PainterConfig(strokes_per_block=per_block)), "brush_sampler": start_ranges(brush),
                "pipeline": config_dict(PainterConfig(strokes_per_block=per_block, **PIPELINE_SETTINGS)),
                "ours_linear": config_dict(PainterConfig(strokes_per_block=per_block, **reference))}
        (out / "config.yaml").write_text(yaml.safe_dump(info, sort_keys=False), encoding="utf-8")

    done, fields = read_done(results_path), csv_fields(results_path) if results_path.is_file() else ALL_FIELDS
    todo = [(p, c, r) for p in images for r in range(args.repeats) for c in args.configs if (p.stem, c, r) not in done]
    print(f"{out.name}: {len(done)} runs done, {len(todo)} to do ({len(images)} images x {len(args.configs)} implementations x {args.repeats} repeats)", flush=True)
    started, new_runs = time.perf_counter(), 0
    for path, config, repeat in todo:
        if args.max_minutes and (time.perf_counter() - started) / 60.0 > args.max_minutes:
            print(f"stopping: the {args.max_minutes:g} minute budget is used up; rerun with --dir {out} to continue", flush=True)
            break
        target = load_image(path)
        run_dir = out / "runs" / path.stem / f"{config}_s{repeat}"
        row = {"image": path.stem, "config": config, "seed": repeat, "grid": args.grid, "strokes": per_block * sum(m * m for m in grids),
               "device": str(device), "mode": args.mode}
        t0 = time.perf_counter()
        if config == "original":
            final, row["wall_s"] = run_original(path, run_dir, args.grid, args.strokes, args.mode, repeat, brush, canvas)
            row.update(renderer=f"original {original_command(path, run_dir, 1, 1, args.mode)[3]}", device=original_device())  # the demo picks its own device
        else:
            settings = {"ours_linear": reference, "pipeline": PIPELINE_SETTINGS}.get(config, {})
            cfg = PainterConfig(strokes_per_block=per_block, seed=repeat, **settings)
            paint = paint_image(target, renderer, cfg, mode=args.mode, max_divide=args.grid, strokes=args.strokes, canvas_color=canvas,
                                device=device)
            final = paint.image
            row.update(optimize_s=paint.optimize_seconds, render_s=paint.render_seconds, neural_psnr=float(paint.result.psnr[-1]), renderer="original-light",
                       beta_ot=cfg.beta_ot, ot_epsilon=cfg.ot_epsilon, ot_iters=cfg.ot_iters, wall_s=time.perf_counter() - t0)
            save_image(run_dir / "final.png", final)
            save_strokes_npz(run_dir / "strokes.npz", paint.strokes, spec, seed=repeat, canvas=canvas)
        m = metrics.image_metrics(final, target, None)
        row.update(psnr=m["psnr"], ssim=m["ssim"])
        new = not results_path.exists()
        with open(results_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if new:
                writer.writeheader()
            writer.writerow({k: ("" if row.get(k) is None else (f"{row[k]:.6g}" if isinstance(row[k], float) else row[k])) for k in fields})
        new_runs += 1
        print(f"{path.stem[:14]:14s} {config:9s} r{repeat} | psnr {m['psnr']:.2f} ssim {m['ssim']:.3f} | {row['wall_s']:.1f} s", flush=True)
    remaining = len(todo) - new_runs
    print(f"done: {new_runs} new runs, {remaining} left" + ("" if remaining else " (complete)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
