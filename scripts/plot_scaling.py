"""Quality and time against the number of strokes, one curve per brush (Week 7, research question RQ3).

Each brush is a ``run_ablation.py`` folder whose configs paint the same images and seeds with different stroke budgets
(``s100=0,...,strokes=100`` and so on). For every config the x value is the number of strokes actually painted (the equal split
of the progressive mode paints 55, 275, 495 and 990 of 100, 300, 500 and 1000) and the y values are the mean over seeds of the
per-seed image average of PSNR, LPIPS and the optimization time, with the standard deviation over seeds. Prints a Markdown table
and writes ``<tag>.png`` (three panels) to ``--out-dir``.

Usage: venv/Scripts/python.exe scripts/plot_scaling.py "Oil=experiments/<folder>" "Tape=experiments/<folder>" ...
       [--time-from "Oil=experiments/<clean timing folder>" ...] [--tag scaling_strokes] [--out-dir report/figures]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from summarize_ablation import column, per_seed_means, read_rows  # noqa: E402

COLORS = ["#1f4e79", "#b23a48", "#2a9d8f", "#e9a23b", "#6d597a"]
PANELS = [("psnr", "PSNR (dB)"), ("lpips", "LPIPS (lower is better)"), ("optimize_s", "optimization time (s)")]


def labelled(spec: str) -> tuple[str, Path]:
    label, eq, folder = spec.partition("=")
    if not eq or not label or not folder:
        raise argparse.ArgumentTypeError(f"expected LABEL=FOLDER, got {spec!r}")
    path = Path(folder)
    return label, path if path.is_absolute() else ROOT / path


def curve(rows: list[dict], key: str) -> list[tuple[float, float, float]]:
    """``(strokes painted, mean, std over seeds)`` per config, sorted by strokes."""
    out = []
    for config in dict.fromkeys(r["config"] for r in rows):
        mine = [r for r in rows if r["config"] == config]
        v = per_seed_means(rows, config, key)
        if v.size:
            out.append((float(np.median(column(mine, "strokes"))), float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0))
    return sorted(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("brushes", nargs="+", type=labelled, metavar="LABEL=FOLDER")
    ap.add_argument("--time-from", nargs="*", type=labelled, default=[], metavar="LABEL=FOLDER",
                    help="read the time panel from these folders instead (clean timing runs, one per brush)")
    ap.add_argument("--tag", default="scaling_strokes")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "report" / "figures")
    args = ap.parse_args(argv)
    timing = dict(args.time_from)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    lines = ["| Brush | Strokes | PSNR (dB) | SSIM | LPIPS | Optimize (s) |", "| :--- | ---: | ---: | ---: | ---: | ---: |"]
    for i, (label, folder) in enumerate(args.brushes):
        rows = read_rows(folder)
        curves = {key: curve(read_rows(timing[label]) if key == "optimize_s" and label in timing else rows, key) for key, _ in PANELS}
        ssim = {x: m for x, m, _ in curve(rows, "ssim")}
        for ax, (key, title) in zip(axes, PANELS):
            pts = curves[key]
            if pts:
                x, m, s = (np.array(v) for v in zip(*pts))
                ax.errorbar(x, m, yerr=s, marker="o", ms=4, capsize=2, color=COLORS[i % len(COLORS)], label=label)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("strokes painted")
            ax.grid(alpha=0.3)
        times = {x: m for x, m, _ in curves["optimize_s"]}
        lpips = {x: m for x, m, _ in curves["lpips"]}
        for x, m, s in curves["psnr"]:
            lp = f"{lpips[x]:.3f}" if x in lpips else "-"
            t = f"{times[x]:.1f}" if x in times else "-"
            lines.append(f"| {label} | {x:.0f} | {m:.2f} +- {s:.2f} | {ssim.get(x, float('nan')):.3f} | {lp} | {t} |")
    axes[0].legend(fontsize=8)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("\n".join(lines))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
