"""Compare the gradients of the pixel loss and of the Sinkhorn cost with respect to the strokes (Week 5 diagnostic).

Ablation A asks whether adding the Sinkhorn term changes the result. This script asks why, by measuring how large its gradient
is next to the pixel loss's. It paints with the pixel loss alone (so the path is the baseline's) and, with the engine's probe hook,
takes two gradients at the first and at the last optimizer step of every new stroke: that of the pixel loss and that of the
*unweighted* Sinkhorn cost (the paper's epsilon, iterations and 24 px grid), both with respect to all stroke parameters. Their
L2 norms are compared three ways: over all strokes placed so far, over the newest stroke (the one that was just initialized from the
error map and is the one the transport cost is meant to guide), and over the newest stroke's position (its first two shape parameters).
A weight beta multiplies the Sinkhorn column, so the ratio at beta = 0.1 is a tenth of the printed one.

One row per probe goes to ``gradients.csv`` and a table to ``summary.md`` in a new experiment folder.

Usage: venv/Scripts/python.exe scripts/loss_gradients.py [--images all|apple,jay] [--every 10] [--seeds 0] [--grid 5] [--strokes 500]
       [--iters 0] [--epsilon 0.01] [--ot-iters 5] [--device auto] [--name gradient_ratio]
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neural_painter.core.device import device_summary, get_device, set_deterministic  # noqa: E402
from neural_painter.core.grid import img2patches  # noqa: E402
from neural_painter.core.image_io import load_image  # noqa: E402
from neural_painter.core.stroke_models import canonical_brush_name  # noqa: E402
from neural_painter.losses.pixel_loss import PixelLoss  # noqa: E402
from neural_painter.losses.sinkhorn import SinkhornLoss  # noqa: E402
from neural_painter.models.neural_renderer import load_original_checkpoint, original_checkpoint_path  # noqa: E402
from neural_painter.pipeline.painter_engine import PainterConfig, PainterEngine  # noqa: E402
from neural_painter.pipeline.train_renderer import EXPERIMENT_ROOT, commit_info, new_experiment_dir  # noqa: E402
from run_ablation import eval_images  # noqa: E402

FIELDS = ["image", "seed", "anchor", "phase", "l1_all", "ot_all", "l1_new", "ot_new", "l1_pos", "ot_pos"]
RATIOS = (("all strokes", "all"), ("newest stroke", "new"), ("newest stroke, position", "pos"))


def gradient_norms(out: torch.Tensor, targets: torch.Tensor, params: list[torch.Tensor], anchor: int, losses: dict) -> dict[str, float]:
    """L2 norms of the gradient of each loss in ``losses`` (name -> callable) with respect to the stroke tensors."""
    row = {}
    for name, loss_fn in losses.items():
        shape_g, color_g, alpha_g = torch.autograd.grad(loss_fn(out, targets), params, retain_graph=True)
        row[f"{name}_all"] = float(torch.sqrt(sum((g**2).sum() for g in (shape_g, color_g, alpha_g))))
        row[f"{name}_new"] = float(torch.sqrt(sum((g[:, anchor] ** 2).sum() for g in (shape_g, color_g, alpha_g))))
        row[f"{name}_pos"] = float(torch.linalg.norm(shape_g[:, anchor, :2]))
    return row


def summarize(rows: list[dict]) -> str:
    lines = ["# Gradient of the Sinkhorn cost relative to the pixel loss", "",
             f"{len(rows)} probes. Each cell is the median over probes of ||grad Sinkhorn|| / ||grad pixel|| (unweighted; multiply by the weight beta).", "",
             "| Moment | " + " | ".join(label for label, _ in RATIOS) + " |", "| :--- | " + " | ".join("---:" for _ in RATIOS) + " |"]
    for phase, label in (("first", "first step of a new stroke"), ("last", "last step of a stroke")):
        sel = [r for r in rows if r["phase"] == phase]
        cells = [f"{np.median([float(r['ot_' + k]) / max(float(r['l1_' + k]), 1e-30) for r in sel]):.3g}" if sel else "-" for _, k in RATIOS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="gradient_ratio")
    ap.add_argument("--images", default="all")
    ap.add_argument("--every", type=int, default=10, help="use every K-th image of the selection (default 10: 3 of the 30)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--strokes", type=int, default=500)
    ap.add_argument("--iters", type=int, default=0, help="optimizer steps per new stroke (0 = the original's int(500 / strokes per block))")
    ap.add_argument("--epsilon", type=float, default=0.01)
    ap.add_argument("--ot-iters", type=int, default=5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--deterministic", action="store_true", help="fix cuDNN's algorithm choice so that a seed reproduces its painting exactly on cuda")
    args = ap.parse_args(argv)
    set_deterministic(args.deterministic)

    brush = canonical_brush_name("oil")
    device = get_device(args.device)
    renderer = load_original_checkpoint(original_checkpoint_path(brush, True), brush, True)
    per_block = args.strokes // args.grid**2
    if per_block < 1:
        raise SystemExit(f"{args.strokes} strokes are too few for a {args.grid}x{args.grid} grid")
    images = eval_images(args.images)[:: max(1, args.every)]
    out = new_experiment_dir(EXPERIMENT_ROOT, args.name, kind="diagnostic")
    (out / "config.yaml").write_text(yaml.safe_dump({
        "created": datetime.now().isoformat(timespec="seconds"), **commit_info(), **device_summary(device), "brush": brush, "grid": args.grid,
        "strokes_per_block": per_block, "iters_per_stroke": args.iters or "original", "images": [p.stem for p in images], "seeds": args.seeds,
        "cudnn_deterministic": bool(args.deterministic),
        "sinkhorn": {"epsilon": args.epsilon, "iters": args.ot_iters, "size": 24, "channel": "random"}, "painting_loss": "pixel (L1) only",
    }, sort_keys=False), encoding="utf-8")

    rows: list[dict] = []
    for path in images:
        image = load_image(path)
        for seed in args.seeds:
            engine = PainterEngine(renderer, PainterConfig(strokes_per_block=per_block, iters_per_stroke=args.iters, seed=seed), device)
            targets = img2patches(image, args.grid, renderer.out_size, engine.device)
            steps = engine.config.resolved_iters()
            losses = {"l1": PixelLoss("l1"), "ot": SinkhornLoss(args.epsilon, args.ot_iters, 24, channel="random", seed=seed)}

            def probe(anchor, step, canvas, tgt, params, path=path, seed=seed, steps=steps):
                phase = step % steps
                if phase in (0, steps - 1):
                    rows.append({"image": path.stem, "seed": seed, "anchor": anchor, "phase": "first" if phase == 0 else "last",
                                 **gradient_norms(canvas, tgt, params, anchor, losses)})

            engine.paint(targets, canvas_color=engine.spec.default_canvas_color, probe=probe)
            print(f"{path.stem[:14]:14s} s{seed}: {len(rows)} probes so far", flush=True)

    with open(out / "gradients.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    text = summarize(rows)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
