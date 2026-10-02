"""Score neural renderers against the procedural rasterizer on one fixed held-out set (Week 4, RQ4).

Every renderer sees the same strokes: ``--n`` (default 1,000) uniform random parameter vectors per brush and size,
fixed by ``VALIDATION_SEED``. By default the eight pretrained 2021 renderers are scored; add ``--ours LABEL=PATH``
for a renderer written by ``pipeline/train_renderer.py`` (``best.pt``). Besides fidelity (PSNR from the MSE over the
whole set, mean SSIM) each row reports the cost of the painter's inner loop on the chosen device: milliseconds for a
forward and for a forward + backward pass over 64 strokes, and peak memory where the backend reports it.

Writes ``experiments/<date>_fidelity_<name>/`` with ``results.csv``, ``results.md`` and ``config.yaml``.

Usage: venv/Scripts/python.exe scripts/renderer_fidelity.py [--device auto] [--n 1000] [--brush all]
       [--ours oil_light=checkpoints/oilpaintbrush_light_100ep/best.pt] [--figure] [--no-original] [--name renderers]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neural_painter.core.device import device_summary, get_device  # noqa: E402
from neural_painter.core.image_io import save_image  # noqa: E402
from neural_painter.core.stroke_models import BRUSH_NAMES, canonical_brush_name  # noqa: E402
from neural_painter.models.neural_renderer import (  # noqa: E402
    benchmark_renderer,
    load_original_checkpoint,
    load_renderer,
    original_checkpoint_path,
)
from neural_painter.pipeline.train_renderer import EXPERIMENT_ROOT, commit_info, evaluate, make_validation_set, new_experiment_dir  # noqa: E402

FIELDS = [
    "renderer", "brush", "size", "params_m", "epochs", "reported_acc_db", "psnr_fg", "psnr_alpha", "psnr_mean",
    "ssim_fg", "ssim_alpha", "forward_ms", "forward_backward_ms", "peak_memory_gb",
]


def score(label: str, model, device, val_cache: dict, n: int) -> dict:
    key = (model.spec.name, model.out_size)
    if key not in val_cache:
        val_cache[key] = make_validation_set(model.spec.name, model.out_size, n)
    model.to(device)
    row = {
        "renderer": label,
        "brush": model.spec.name,
        "size": model.out_size,
        "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
        "epochs": model.source.get("epoch", -1) + (1 if model.source.get("format") == "original" else 0),  # original stores the last index
        "reported_acc_db": model.source.get("val_acc_db"),
        **evaluate(model, val_cache[key], device),
        **benchmark_renderer(model, device),
    }
    model.to("cpu")  # so the peak memory of the next row does not include the renderers scored before it
    return row


@torch.no_grad()
def comparison_grid(val: dict, entries: list, device, n: int = 8, tile: int = 128, margin: int = 190) -> np.ndarray:
    """Rows: the rasterized foreground and then each renderer's, then the same for the opacity map; columns: the first ``n`` strokes.

    ``entries`` is a list of ``(label, renderer)``. Tiles are shown at ``tile`` pixels, so 32 px outputs are enlarged
    (nearest neighbour) and the figure looks the same for both renderer sizes.
    """
    params = val["params"][:n].to(device)
    groups = [[("rasterizer", val["foreground"][:n])], [("rasterizer", val["alpha"][:n])]]
    for label, model in entries:
        fg, alpha = model.to(device).eval()(params)
        groups[0].append((label, fg.clamp(0, 1).cpu()))
        groups[1].append((label, alpha.clamp(0, 1).cpu()))
    rows = []
    for k, group in enumerate(groups):
        for label, images in group:
            strip = np.concatenate(list(images.permute(0, 2, 3, 1).numpy()), axis=1)
            strip = cv2.resize(strip, None, fx=tile / images.shape[-1], fy=tile / images.shape[-1], interpolation=cv2.INTER_NEAREST)
            side = np.zeros((strip.shape[0], margin, 3), np.uint8)  # putText needs an 8-bit image
            cv2.putText(side, f"{'fg' if k == 0 else 'alpha'}: {label}", (6, strip.shape[0] // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            rows.append(np.concatenate([side.astype(np.float32) / 255.0, strip], axis=1))
    return np.clip(np.concatenate(rows, axis=0), 0.0, 1.0).astype(np.float32)


def fmt(field: str, value) -> str:
    if value is None or value == "":
        return "-"
    if field in ("psnr_fg", "psnr_alpha", "psnr_mean", "reported_acc_db"):
        return f"{value:.2f}"
    if field in ("ssim_fg", "ssim_alpha"):
        return f"{value:.3f}"
    if field in ("forward_ms", "forward_backward_ms"):
        return f"{value:.1f}"
    if field in ("params_m", "peak_memory_gb"):
        return f"{value:.2f}"
    return str(value)


def markdown_table(rows: list[dict]) -> str:
    head = ["Renderer", "Brush", "Size", "Params (M)", "Epochs", "Reported (dB)", "PSNR fg", "PSNR alpha", "PSNR mean",
            "SSIM fg", "SSIM alpha", "Fwd (ms)", "Fwd+bwd (ms)", "Peak mem (GB)"]
    lines = ["| " + " | ".join(head) + " |", "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(f, r.get(f)) for f in FIELDS) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="auto", help="auto, cuda, mps or cpu (the timing columns are for this device)")
    ap.add_argument("--n", type=int, default=1000, help="held-out strokes per brush and size")
    ap.add_argument("--brush", default="all", help="'all' or a comma-separated list (oil, watercolor, marker, tape)")
    ap.add_argument("--ours", action="append", default=[], metavar="LABEL=PATH", help="a checkpoint written by train_renderer (repeatable)")
    ap.add_argument("--no-original", action="store_true", help="skip the pretrained 2021 renderers")
    ap.add_argument("--figure", action="store_true", help="for each --ours renderer, save a side-by-side with the rasterizer and the original")
    ap.add_argument("--name", default="renderers", help="experiment folder suffix")
    args = ap.parse_args(argv)

    device = get_device(args.device)
    brushes = list(BRUSH_NAMES) if args.brush == "all" else [canonical_brush_name(b) for b in args.brush.split(",")]
    val_cache: dict = {}
    rows: list[dict] = []
    originals: dict = {}  # (brush, light) -> (label, renderer), kept for the comparison figures
    ours: list = []  # (label, renderer)

    if not args.no_original:
        for brush in brushes:
            for light in (False, True):
                path = original_checkpoint_path(brush, light)
                if not path.is_file():
                    print(f"skipping {path.parent.name}: checkpoint not found", file=sys.stderr)
                    continue
                label = f"original {'light' if light else 'full'}"
                model = load_original_checkpoint(path, brush, light)
                originals[(brush, light)] = (label, model)
                rows.append(score(label, model, device, val_cache, args.n))
                print(f"{label:15s} {brush:14s} psnr {rows[-1]['psnr_mean']:.2f} dB", flush=True)
    for spec in args.ours:
        label, _, path = spec.partition("=")
        if not path:
            ap.error(f"--ours expects LABEL=PATH, got {spec!r}")
        model = load_renderer(path)
        ours.append((label, model))
        rows.append(score(label, model, device, val_cache, args.n))
        print(f"{label:15s} {rows[-1]['brush']:14s} psnr {rows[-1]['psnr_mean']:.2f} dB", flush=True)
    if not rows:
        print("nothing to score", file=sys.stderr)
        return 1

    out = new_experiment_dir(EXPERIMENT_ROOT, args.name, kind="fidelity")
    if args.figure:
        for label, model in ours:
            entries = [e for e in [originals.get((model.spec.name, model.light))] if e is not None] + [(label, model)]
            grid = comparison_grid(val_cache[(model.spec.name, model.out_size)], entries, device)
            save_image(out / f"comparison_{re.sub(r'[^A-Za-z0-9_.-]+', '_', label)}.png", grid)  # no spaces in file names
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows([{f: ("" if r.get(f) is None else r.get(f)) for f in FIELDS} for r in rows])
    table = markdown_table(rows)
    (out / "results.md").write_text(
        f"# Renderer fidelity ({args.n} held-out strokes per brush and size)\n\n"
        f"Device for the timing columns: `{device}`. PSNR is computed from the MSE over the whole set; the SSIM is the mean per stroke; "
        "\"Reported\" is the validation accuracy the authors stored in their checkpoint (mean of foreground and alpha PSNR on their own set).\n\n"
        + table + "\n",
        encoding="utf-8",
    )
    info = {"created": datetime.now().isoformat(timespec="seconds"), **commit_info(), **device_summary(device), "n": args.n, "brushes": brushes,
            "ours": args.ours, "original": not args.no_original}
    (out / "config.yaml").write_text(yaml.safe_dump(info, sort_keys=False), encoding="utf-8")
    print("\n" + table + f"\n\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
