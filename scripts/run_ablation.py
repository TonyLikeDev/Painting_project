"""Paint the evaluation set under several settings and log one row per run (Week 5: Ablation A and the grid comparison).

Every run paints one frozen evaluation image with ``--grid`` x ``--grid`` blocks (``--grid 1`` is full image mode) and
``--strokes`` strokes in total, rasterizes the strokes at 512 px and scores the result against the image. One row per
run goes to ``results.csv`` in the experiment folder; the final image, the strokes and the per-step loss / PSNR history
of every run go to ``runs/<image>/<config>_s<seed>/`` (git-ignored: only the table is committed).

A *config* is ``NAME=BETA_OT[:EPSILON[:ITERS]][@GRID][,FIELD=VALUE...]``: ``pixel=0`` is the pixel loss alone (the original's
default), ``ot=0.1`` adds the Sinkhorn term with weight 0.1 (the original's ``--with_ot_loss``), ``ot1=1:0.01:5`` sets the
weight, the regularization and the iteration count, and ``full=0@1`` paints with a 1 x 1 grid instead of ``--grid`` (the stroke
budget ``--strokes`` is split over the blocks of that grid). Fields of ``PainterConfig`` can be set after commas, for example
``flat=0,init=uniform`` (stroke centres drawn uniformly instead of from the error map). Several configs on the same images and
seeds make a paired comparison.

The script is resumable: with ``--dir`` pointing at an existing experiment folder it skips the runs already in
``results.csv`` and appends the rest, so a long ablation can be split into chunks (``--max-minutes``). A new folder is
never allowed to overwrite an old one.

Usage: venv/Scripts/python.exe scripts/run_ablation.py --name ablation_a --configs pixel=0 ot=0.1 --seeds 0 1 2
       [--images all|apple,jay] [--every 3] [--grid 5] [--strokes 500] [--brush oil] [--renderer original-light|original-full|PATH]
       [--device auto] [--lpips] [--max-minutes 8] [--dir experiments/<existing folder>]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neural_painter.core.device import device_summary, get_device, set_deterministic  # noqa: E402
from neural_painter.core.image_io import load_image, save_image  # noqa: E402
from neural_painter.core.stroke_models import canonical_brush_name, get_brush  # noqa: E402
from neural_painter.models.neural_renderer import load_original_checkpoint, load_renderer, original_checkpoint_path  # noqa: E402
from neural_painter.pipeline import metrics  # noqa: E402
from neural_painter.pipeline.painter_engine import PainterConfig, config_dict, paint_fixed_grid, save_strokes_npz  # noqa: E402
from neural_painter.pipeline.train_renderer import (  # noqa: E402
    EXPERIMENT_ROOT,
    commit_info,
    lower_process_priority,
    new_experiment_dir,
)

EVAL_SET = ROOT / "data" / "eval_set"
THRESHOLDS = (22.0, 24.0)  # neural-canvas PSNR (dB) whose first step is recorded
FIELDS = [
    "image", "config", "seed", "grid", "strokes", "beta_ot", "ot_epsilon", "ot_iters", "optimize_s", "render_s", "psnr", "ssim", "lpips",
    "neural_psnr", "steps_to_22db", "steps_to_24db", "device", "renderer",
]


def _flag(value: str) -> bool:
    return value.lower() in ("1", "true", "yes")


OVERRIDES = {  # PainterConfig fields that a config can set by name, and how to read their value
    "init": str, "resize": str, "lr": float, "optimizer": str, "centered": _flag, "morphology": _flag, "ot_channel": str, "ot_lse": str,
}


def parse_config(spec: str) -> tuple[str, dict]:
    name, _, rest = spec.partition("=")
    head, *extras = rest.split(",")
    head, at, grid = head.partition("@")
    if not name or not head or (at and not grid):
        raise argparse.ArgumentTypeError(f"config must look like NAME=BETA_OT[:EPSILON[:ITERS]][@GRID][,FIELD=VALUE...], got {spec!r}")
    parts = head.split(":")
    out = {"beta_ot": float(parts[0])}
    if len(parts) > 1:
        out["ot_epsilon"] = float(parts[1])
    if len(parts) > 2:
        out["ot_iters"] = int(parts[2])
    if grid:
        out["grid"] = int(grid)
    for item in extras:
        key, eq, value = item.partition("=")
        if not eq or key not in OVERRIDES:
            raise argparse.ArgumentTypeError(f"cannot read {item!r} in {spec!r}; the fields a config can set are {sorted(OVERRIDES)}")
        out[key] = OVERRIDES[key](value)
    return name, out


def eval_images(selection: str) -> list[Path]:
    paths = sorted(EVAL_SET.glob("*.png"))
    if selection == "all":
        return paths
    wanted = [s for s in selection.split(",") if s]
    chosen = [p for p in paths if any(p.stem == w or p.stem.startswith(w) for w in wanted)]
    missing = [w for w in wanted if not any(p.stem == w or p.stem.startswith(w) for p in paths)]
    if missing:
        raise SystemExit(f"no evaluation image matches {missing}")
    return chosen


def load_renderer_for(spec: str, brush: str, device):
    if spec in ("original-light", "original-full"):
        light = spec.endswith("light")
        return load_original_checkpoint(original_checkpoint_path(brush, light), brush, light), spec
    return load_renderer(spec, brush), Path(spec).name


def read_done(path: Path) -> set[tuple[str, str, int]]:
    if not path.is_file():
        return set()
    with open(path, newline="", encoding="utf-8") as fh:
        return {(r["image"], r["config"], int(r["seed"])) for r in csv.DictReader(fh)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="ablation", help="experiment folder suffix (a new folder <date>_ablation_<name> is made)")
    ap.add_argument("--dir", type=Path, default=None, help="resume this existing experiment folder instead of making a new one")
    ap.add_argument("--configs", nargs="+", type=parse_config, default=[parse_config("pixel=0"), parse_config("ot=0.1")])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--images", default="all")
    ap.add_argument("--every", type=int, default=1, help="use every K-th image of the selection (an evenly spread, fixed subset for the sweeps)")
    ap.add_argument("--brush", default="oil")
    ap.add_argument("--renderer", default="original-light", help="original-light, original-full or a checkpoint written by train_renderer")
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--strokes", type=int, default=500, help="total strokes; each block gets strokes // grid^2")
    ap.add_argument("--iters", type=int, default=0, help="optimizer steps per new stroke (0 = the original's int(500 / strokes per block))")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--lpips", action="store_true", help="also compute LPIPS (needs the AlexNet weights to be cached already)")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="start no new run after this many minutes (0 = no limit)")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many new runs (0 = no limit; for smoke tests)")
    ap.add_argument("--low-priority", action="store_true", help="run at below-normal priority to keep the machine responsive")
    ap.add_argument("--deterministic", action="store_true", help="fix cuDNN's algorithm choice so that a seed reproduces its painting exactly on cuda")
    args = ap.parse_args(argv)
    if args.low_priority:
        lower_process_priority()
    set_deterministic(args.deterministic)

    brush = canonical_brush_name(args.brush)
    device = get_device(args.device)
    renderer, renderer_name = load_renderer_for(args.renderer, brush, device)
    plans: dict[str, tuple[dict, int, int]] = {}  # config name -> (painter settings, grid, strokes per block)
    for name, cfg in args.configs:
        settings = dict(cfg)
        grid = settings.pop("grid", args.grid)
        if args.strokes // grid**2 < 1:
            raise SystemExit(f"{args.strokes} strokes are too few for a {grid}x{grid} grid")
        plans[name] = (settings, grid, args.strokes // grid**2)
    per_block = args.strokes // (args.grid**2)
    images = eval_images(args.images)[:: max(1, args.every)]
    lpips_metric = metrics.LPIPSMetric(device) if args.lpips else None

    if args.dir is not None:
        out = args.dir if args.dir.is_absolute() else ROOT / args.dir
        if not out.is_dir():
            raise SystemExit(f"{out} does not exist; leave out --dir to start a new experiment")
    else:
        out = new_experiment_dir(EXPERIMENT_ROOT, args.name, kind="ablation")
    results_path = out / "results.csv"
    if not (out / "config.yaml").exists():
        info = {"created": datetime.now().isoformat(timespec="seconds"), **commit_info(), **device_summary(device), "brush": brush,
                "renderer": renderer_name, "grid": args.grid, "strokes_per_block": per_block, "total_strokes": per_block * args.grid**2,
                "images": [p.stem for p in images], "seeds": args.seeds, "lpips": bool(args.lpips), "iters_per_stroke": args.iters or "original",
                "cudnn_deterministic": bool(args.deterministic),
                "configs": {n: {**c, "grid": plans[n][1], "strokes_per_block": plans[n][2], "total_strokes": plans[n][2] * plans[n][1] ** 2}
                            for n, c in args.configs},
                "painter_defaults": config_dict(PainterConfig(strokes_per_block=per_block))}
        (out / "config.yaml").write_text(yaml.safe_dump(info, sort_keys=False), encoding="utf-8")

    done = read_done(results_path)
    todo = [(p, n, s) for p in images for n, _ in args.configs for s in args.seeds if (p.stem, n, s) not in done]
    print(f"{out.name}: {len(done)} runs done, {len(todo)} to do ({len(images)} images x {len(args.configs)} configs x {len(args.seeds)} seeds)", flush=True)
    started, new_runs = time.perf_counter(), 0
    for path, name, seed in todo:
        if args.max_minutes and (time.perf_counter() - started) / 60.0 > args.max_minutes:
            print(f"stopping: the {args.max_minutes:g} minute budget is used up; rerun with --dir {out} to continue", flush=True)
            break
        if args.limit and new_runs >= args.limit:
            break
        image = load_image(path)
        settings, grid, n_block = plans[name]
        cfg = PainterConfig(strokes_per_block=n_block, iters_per_stroke=args.iters, seed=seed, **settings)
        paint = paint_fixed_grid(image, renderer, cfg, grid=grid, device=device)
        m = metrics.image_metrics(paint.image, image, lpips_metric)
        hit = {t: paint.result.steps_to_psnr(t) for t in THRESHOLDS}
        row = {
            "image": path.stem, "config": name, "seed": seed, "grid": grid, "strokes": len(paint.strokes), "beta_ot": cfg.beta_ot,
            "ot_epsilon": cfg.ot_epsilon, "ot_iters": cfg.ot_iters, "optimize_s": paint.result.seconds, "render_s": paint.render_seconds,
            "psnr": m["psnr"], "ssim": m["ssim"], "lpips": m["lpips"], "neural_psnr": float(paint.result.psnr[-1]),
            "steps_to_22db": hit[22.0], "steps_to_24db": hit[24.0], "device": str(device), "renderer": renderer_name,
        }
        run_dir = out / "runs" / path.stem / f"{name}_s{seed}"
        save_image(run_dir / "final.png", paint.image)
        save_strokes_npz(run_dir / "strokes.npz", paint.strokes, get_brush(brush), seed=seed)
        np.savez(run_dir / "history.npz", loss=paint.result.loss, psnr=paint.result.psnr, anchor=paint.result.anchor)
        new = not results_path.exists()
        with open(results_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                writer.writeheader()
            writer.writerow({k: ("" if v is None else (f"{v:.6g}" if isinstance(v, float) else v)) for k, v in row.items()})
        new_runs += 1
        print(f"{path.stem[:14]:14s} {name:10s} s{seed} | psnr {m['psnr']:.2f} ssim {m['ssim']:.3f} | optimize {row['optimize_s']:.1f} s", flush=True)
    remaining = len(todo) - new_runs
    print(f"done: {new_runs} new runs, {remaining} left" + ("" if remaining else " (complete)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
