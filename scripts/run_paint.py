"""One command from photo to painting (Week 7).

    venv/Scripts/python.exe scripts/run_paint.py PHOTO [--brush oil] [--strokes 500] [--grid progressive] [--max-divide 5]
        [--canvas white] [--renderer original-light] [--device auto] [--seed 0] [--size 512] [--fit stretch] [--out FOLDER]

The photo is fitted to a ``--size`` pixel square (``--fit stretch`` is the original's behaviour: the painting is stretched back to
the photo's aspect ratio when it is saved; ``crop`` and ``pad`` keep the strokes undistorted and give a square painting), painted
with ``pipeline.progressive_painter.paint_image`` (by default coarse to fine, with the cosine learning-rate schedule at a peak of
0.004 that Week 7 adopted) while one line shows the level, strokes done, loss, PSNR of the neural canvas
and elapsed time, and rasterized at ``--size`` pixels. A new folder (default ``output/<date>_paint_<photo>_<brush>``, git-ignored)
receives ``final.png``, ``input.png`` (the fitted photo the painting is scored against), ``strokes.npz``, ``config.yaml`` (every
setting, the device and the commit) and ``results.csv`` (one row: PSNR, SSIM, times, strokes, steps, peak memory).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neural_painter.core.device import PeakMemory, device_summary, disable_power_throttling, get_device, set_deterministic  # noqa: E402
from neural_painter.core.grid import GRID_MODES  # noqa: E402
from neural_painter.core.image_io import ImageTooLargeError, UnsupportedImageError, preprocess, resize, save_image  # noqa: E402
from neural_painter.core.stroke_models import canonical_brush_name, get_brush  # noqa: E402
from neural_painter.pipeline import metrics  # noqa: E402
from neural_painter.pipeline.painter_engine import ALLOCATIONS, LR_SCHEDULES, PRECISIONS, PainterConfig, config_dict, save_strokes_npz  # noqa: E402
from neural_painter.pipeline.progressive_painter import Progress, paint_image  # noqa: E402
from neural_painter.pipeline.train_renderer import commit_info, new_experiment_dir  # noqa: E402
from run_ablation import load_renderer_for  # noqa: E402

OUTPUT_ROOT = ROOT / "output"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paint a photo with the neural painter (Week 7)")
    p.add_argument("image")
    p.add_argument("--brush", default="oil", help="oil | watercolor | marker | tape")
    p.add_argument("--strokes", type=int, default=500)
    p.add_argument("--grid", default="progressive", choices=GRID_MODES)
    p.add_argument("--max-divide", type=int, default=5, help="finest grid (progressive, fixed)")
    p.add_argument("--canvas", default=None, help="black | white | #rrggbb (default: the brush's)")
    p.add_argument("--renderer", default="original-light", help="original-light, original-full or a checkpoint written by train_renderer")
    # the pipeline's defaults, tuned in Week 7 (report Section 4.4.3); PainterConfig keeps the reference's (constant 0.002)
    p.add_argument("--lr", type=float, default=0.004, help="peak learning rate")
    p.add_argument("--lr-schedule", default="cosine", choices=LR_SCHEDULES)
    p.add_argument("--allocation", default="equal", choices=ALLOCATIONS, help="'error' gives busy blocks more strokes (slower)")
    p.add_argument("--precision", default="float32", choices=PRECISIONS, help="bfloat16: about 13 %% faster on cuda, slightly different paintings")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=int, default=512, help="side of the square the photo is fitted to and the painting is drawn at")
    p.add_argument("--fit", default="stretch", choices=["stretch", "crop", "pad"])
    p.add_argument("--deterministic", action="store_true", help="bit-exact repeats on cuda")
    p.add_argument("--out", type=Path, default=None, help="output folder (must not exist or be empty)")
    p.add_argument("--quiet", action="store_true")
    return p


def progress_printer(quiet: bool):
    """A callback that keeps one status line up to date in a terminal, or prints a line every 2 s into a log file."""
    tty, last = sys.stdout.isatty(), [-1e9]

    def show(p: Progress) -> None:
        if quiet:
            return
        line = (f"level {p.level}/{p.levels} ({p.m}x{p.m}) | strokes {p.strokes_done}/{p.strokes_total} | loss {p.loss:.4f} | "
                f"psnr {p.psnr:.2f} dB | {p.elapsed:.1f} s")
        if tty:
            print("\r" + line, end="", flush=True)
        elif p.elapsed - last[0] >= 2.0 or p.strokes_done == p.strokes_total:
            last[0] = p.elapsed
            print(line, flush=True)

    return show


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    disable_power_throttling()
    set_deterministic(args.deterministic)
    brush = canonical_brush_name(args.brush)
    spec = get_brush(brush)
    source = Path(args.image)
    if args.out is not None and args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty; choose a new folder")
    try:  # a photo that cannot be painted stops here, with a message and before any folder is made
        photo = preprocess(source, size=args.size, mode=args.fit)
    except (FileNotFoundError, UnsupportedImageError, ImageTooLargeError) as exc:
        raise SystemExit(f"cannot paint {source}: {exc}") from None
    if args.out is not None:
        out = args.out
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = new_experiment_dir(OUTPUT_ROOT, f"{source.stem}_{spec.name}", kind="paint")

    device = get_device(args.device)
    renderer, renderer_name = load_renderer_for(args.renderer, brush, device)
    cfg = PainterConfig(seed=args.seed, lr=args.lr, lr_schedule=args.lr_schedule, allocation=args.allocation, precision=args.precision)
    started = time.perf_counter()
    with PeakMemory(device) as memory:
        paint = paint_image(photo.image, renderer, cfg, mode=args.grid, max_divide=args.max_divide, strokes=args.strokes,
                            canvas_size=args.size, canvas_color=args.canvas, device=device, callback=progress_printer(args.quiet))
    wall = time.perf_counter() - started
    if not args.quiet and sys.stdout.isatty():
        print()
    m = metrics.image_metrics(paint.image, photo.image)
    width, height = photo.output_size(args.size, keep_aspect_ratio=args.fit == "stretch")
    save_image(out / "final.png", resize(paint.image, (width, height)) if (width, height) != (args.size, args.size) else paint.image)
    save_image(out / "input.png", photo.image)
    save_strokes_npz(out / "strokes.npz", paint.strokes, spec, seed=args.seed)
    info = {"created": datetime.now().isoformat(timespec="seconds"), **commit_info(), **device_summary(device), "image": str(source),
            "original_size": list(photo.original_size), "fit": args.fit, "size": args.size, "output_size": [width, height], "brush": brush,
            "renderer": renderer_name, "grid": args.grid, "max_divide": args.max_divide, "strokes_requested": args.strokes,
            "canvas": args.canvas or spec.default_canvas_color, "cudnn_deterministic": bool(args.deterministic), "painter": config_dict(cfg),
            "levels": [{"m": lv.m, "strokes": int(lv.counts.sum()), "steps": len(lv.result.loss)} for lv in paint.levels]}
    (out / "config.yaml").write_text(yaml.safe_dump(info, sort_keys=False), encoding="utf-8")
    row = {"image": source.stem, "brush": brush, "grid": args.grid, "strokes": len(paint.strokes), "steps": paint.steps, "psnr": m["psnr"],
           "ssim": m["ssim"], "optimize_s": paint.optimize_seconds, "render_s": paint.render_seconds, "wall_s": wall,
           "peak_memory_gb": memory.gb, "device": str(device), "seed": args.seed}
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        writer.writeheader()
        writer.writerow({k: f"{v:.6g}" if isinstance(v, float) else v for k, v in row.items()})
    print(f"{len(paint.strokes)} strokes, {paint.steps} steps on {device}: PSNR {m['psnr']:.2f} dB, SSIM {m['ssim']:.3f}, "
          f"{paint.optimize_seconds:.1f} s optimizing + {paint.render_seconds:.1f} s rendering\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
