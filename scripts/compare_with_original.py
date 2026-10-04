"""Paint evaluation images with the original 2021 code and with this package, and score both the same way (Week 5).

Both run the fixed-grid algorithm of the original ``demo.py``: pixel loss only, the pretrained light oil renderer, a white
canvas, a ``--grid`` x ``--grid`` grid and ``--strokes`` strokes. The original is run unmodified as a subprocess from its own
checkout (``demo.py`` has no seed flag, so its repeats are unseeded and the ``seed`` column is only the repeat number); ours runs
with the same repeat numbers as seeds. The finished 512 px paintings are scored with the same ``metrics.image_metrics``. One row
per run goes to ``results.csv`` in the layout of ``run_ablation.py`` (plus the wall time of the whole process), so
``summarize_ablation.py --baseline original`` gives the paired comparison of the two implementations.

A third config, ``ours_linear``, is our engine with the original's way of shrinking the target to the blocks (OpenCV's default
bilinear resize instead of area averaging), to show how much of any difference comes from that one choice.

The original writes one PNG per stroke and a video next to its result; they are deleted after the final image has been read.
The script is resumable with ``--dir`` and ``--max-minutes``, like ``run_ablation.py``; a config added later (``ours_linear`` was)
is simply the next missing runs of an existing folder.

Usage: venv/Scripts/python.exe scripts/compare_with_original.py --name oil_fixed_grid [--every 6] [--repeats 3] [--grid 5] [--strokes 500]
       [--device auto] [--max-minutes 8] [--dir experiments/<existing folder>]
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

from neural_painter.core.device import device_summary, get_device, set_deterministic  # noqa: E402
from neural_painter.core.image_io import load_image, save_image  # noqa: E402
from neural_painter.core.stroke_models import canonical_brush_name, get_brush  # noqa: E402
from neural_painter.models.neural_renderer import load_original_checkpoint, original_checkpoint_path  # noqa: E402
from neural_painter.pipeline import metrics  # noqa: E402
from neural_painter.pipeline.painter_engine import PainterConfig, config_dict, paint_fixed_grid, save_strokes_npz  # noqa: E402
from neural_painter.pipeline.train_renderer import EXPERIMENT_ROOT, commit_info, lower_process_priority, new_experiment_dir  # noqa: E402
from run_ablation import FIELDS, eval_images, read_done  # noqa: E402

BRUSH = canonical_brush_name("oil")
ORIGINAL = ROOT / "stylized-neural-painting"
CHECKPOINT_DIR = "checkpoints_G_oilpaintbrush_light"  # relative to the original checkout, as its demo expects
CONFIGS = ("original", "ours", "ours_linear")  # "ours_linear": our engine with the original's bilinear shrinking of the target (resize="linear")
ALL_FIELDS = FIELDS + ["wall_s"]


def original_device() -> str:
    """The device the original demo will pick: ``cuda:0`` when PyTorch sees a GPU (an empty ``CUDA_VISIBLE_DEVICES`` hides it)."""
    hidden = os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    return "cuda:0" if torch.cuda.is_available() and not hidden else "cpu"


def run_original(image_path: Path, run_dir: Path, grid: int, strokes: int) -> tuple[np.ndarray, float]:
    """Paint ``image_path`` with the original ``demo.py``; returns its final 512 px image and the wall time of the process."""
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "demo.py", "--img_path", image_path.resolve().as_posix(), "--canvas_color", "white", "--max_m_strokes", str(strokes),
        "--m_grid", str(grid), "--renderer", "oilpaintbrush", "--renderer_checkpoint_dir", CHECKPOINT_DIR, "--net_G", "zou-fusion-net-light",
        "--disable_preview", "--output_dir", run_dir.resolve().as_posix(),
    ]
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
    ap.add_argument("--images", default="all")
    ap.add_argument("--every", type=int, default=6, help="use every K-th image of the selection (default 6: 5 of the 30 images)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--strokes", type=int, default=500)
    ap.add_argument("--device", default="auto", help="device of our run; the original picks cuda when it is available")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="start no new run after this many minutes (0 = no limit)")
    ap.add_argument("--low-priority", action="store_true")
    ap.add_argument("--deterministic", action="store_true", help="fix cuDNN's algorithm choice for our runs (the original cannot be seeded)")
    args = ap.parse_args(argv)
    if args.low_priority:
        lower_process_priority()
    set_deterministic(args.deterministic)

    device = get_device(args.device)
    renderer = load_original_checkpoint(original_checkpoint_path(BRUSH, True), BRUSH, True)
    per_block = args.strokes // args.grid**2
    if per_block < 1:
        raise SystemExit(f"{args.strokes} strokes are too few for a {args.grid}x{args.grid} grid")
    images = eval_images(args.images)[:: max(1, args.every)]

    if args.dir is not None:
        out = args.dir if args.dir.is_absolute() else ROOT / args.dir
        if not out.is_dir():
            raise SystemExit(f"{out} does not exist; leave out --dir to start a new experiment")
    else:
        out = new_experiment_dir(EXPERIMENT_ROOT, args.name, kind="check")
    results_path = out / "results.csv"
    if not (out / "config.yaml").exists():
        info = {"created": datetime.now().isoformat(timespec="seconds"), **commit_info(), **device_summary(device), "brush": BRUSH,
                "grid": args.grid, "strokes_per_block": per_block, "total_strokes": per_block * args.grid**2, "images": [p.stem for p in images],
                "repeats": args.repeats, "cudnn_deterministic": bool(args.deterministic), "original": "demo.py of stylized-neural-painting, run unmodified (unseeded), pixel loss, light oil renderer, white canvas",
                "ours": config_dict(PainterConfig(strokes_per_block=per_block))}
        (out / "config.yaml").write_text(yaml.safe_dump(info, sort_keys=False), encoding="utf-8")

    done = read_done(results_path)
    todo = [(p, c, r) for p in images for r in range(args.repeats) for c in CONFIGS if (p.stem, c, r) not in done]
    print(f"{out.name}: {len(done)} runs done, {len(todo)} to do ({len(images)} images x {len(CONFIGS)} implementations x {args.repeats} repeats)", flush=True)
    started, new_runs = time.perf_counter(), 0
    for path, config, repeat in todo:
        if args.max_minutes and (time.perf_counter() - started) / 60.0 > args.max_minutes:
            print(f"stopping: the {args.max_minutes:g} minute budget is used up; rerun with --dir {out} to continue", flush=True)
            break
        target = load_image(path)
        run_dir = out / "runs" / path.stem / f"{config}_s{repeat}"
        row = {"image": path.stem, "config": config, "seed": repeat, "grid": args.grid, "strokes": args.strokes, "device": str(device)}
        t0 = time.perf_counter()
        if config == "original":
            final, row["wall_s"] = run_original(path, run_dir, args.grid, args.strokes)
            row.update(renderer="original demo.py", device=original_device())  # the demo picks its own device
        else:
            cfg = PainterConfig(strokes_per_block=per_block, seed=repeat, resize="linear" if config == "ours_linear" else "area")
            paint = paint_fixed_grid(target, renderer, cfg, grid=args.grid, device=device)
            final = paint.image
            row.update(optimize_s=paint.result.seconds, render_s=paint.render_seconds, neural_psnr=float(paint.result.psnr[-1]), renderer="original-light",
                       beta_ot=cfg.beta_ot, ot_epsilon=cfg.ot_epsilon, ot_iters=cfg.ot_iters, wall_s=time.perf_counter() - t0)
            save_image(run_dir / "final.png", final)
            save_strokes_npz(run_dir / "strokes.npz", paint.strokes, get_brush(BRUSH), seed=repeat)
        m = metrics.image_metrics(final, target, None)
        row.update(psnr=m["psnr"], ssim=m["ssim"])
        new = not results_path.exists()
        with open(results_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=ALL_FIELDS)
            if new:
                writer.writeheader()
            writer.writerow({k: ("" if row.get(k) is None else (f"{row[k]:.6g}" if isinstance(row[k], float) else row[k])) for k in ALL_FIELDS})
        new_runs += 1
        print(f"{path.stem[:14]:14s} {config:9s} r{repeat} | psnr {m['psnr']:.2f} ssim {m['ssim']:.3f} | {row['wall_s']:.1f} s", flush=True)
    remaining = len(todo) - new_runs
    print(f"done: {new_runs} new runs, {remaining} left" + ("" if remaining else " (complete)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
