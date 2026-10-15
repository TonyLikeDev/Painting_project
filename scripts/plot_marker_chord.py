"""Figure for the marker-pen start (Week 6): which renderers draw a short stroke, and what the start length does to the painting.

Left: the share of straight marker strokes of a given chord length that each renderer draws (largest opacity above 0.5), from
``scripts/marker_chord.py``. Right: PSNR (and LPIPS, right axis) of the finished paintings against the length a new stroke
starts with, from one or more ``run_ablation.py`` folders whose configs set ``init_chord``; each point is the mean over the
seeds of the per-seed image average, with the standard deviation over seeds. A config that appears in several folders is
taken from the first.

Usage: venv/Scripts/python.exe scripts/plot_marker_chord.py experiments/<diagnostic folder> experiments/<sweep> [experiments/<sweep> ...]
       [--out report/figures/marker_chord.png]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from summarize_ablation import per_seed_means, read_rows  # noqa: E402

STYLE = {("rasterizer", 32): ("#1f4e79", "-", "rasterizer, 32 px"), ("rasterizer", 128): ("#1f4e79", ":", "rasterizer, 128 px"),
         ("neural light", 32): ("#b23a48", "-", "neural, light (32 px)"), ("neural full", 128): ("#b23a48", ":", "neural, full (128 px)")}


def chord_points(folders: list[Path]) -> list[tuple[float, str, list[dict]]]:
    """``(chord, config, rows)`` per config, in chord order; a config already seen in an earlier folder is skipped."""
    seen, points = set(), []
    for folder in folders:
        configs = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))["configs"]
        rows = read_rows(folder)
        for name, cfg in configs.items():
            if name in seen or "init_chord" not in cfg:
                continue
            seen.add(name)
            points.append((float(cfg["init_chord"]), name, [r for r in rows if r["config"] == name]))
    runs = {frozenset((r["image"], int(r["seed"])) for r in rows) for _, _, rows in points}
    if len(runs) > 1:  # one curve must compare the start lengths on the same images and seeds
        raise SystemExit("the configs to plot were not run on the same images and seeds; pass sweeps of one image set")
    return sorted(points)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("diagnostic", type=Path)
    ap.add_argument("sweeps", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "report" / "figures" / "marker_chord.png")
    args = ap.parse_args(argv)

    with open(args.diagnostic / "results.csv", newline="", encoding="utf-8") as fh:
        diag = list(csv.DictReader(fh))
    fig, (ax_vis, ax_psnr) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for (renderer, size), (color, ls, label) in STYLE.items():
        pts = sorted((float(r["chord"]), float(r["visible"])) for r in diag if r["renderer"] == renderer and int(r["size"]) == size)
        if pts:
            ax_vis.plot(*zip(*pts), color=color, ls=ls, marker="o", ms=3, lw=1.5, label=label)
    ax_vis.set_xlabel("chord length of the stroke (block units)")
    ax_vis.set_ylabel("share of strokes drawn")
    ax_vis.set_title("Which renderers draw a short marker stroke")
    ax_vis.legend(frameon=False, fontsize=8, loc="lower right")

    points = chord_points(args.sweeps)
    x = np.array([c for c, _, _ in points])
    stats = {k: np.array([[v.mean(), v.std(ddof=1)] for v in (per_seed_means(rows, name, k) for _, name, rows in points)])
             for k in ("psnr", "lpips")}
    ax_psnr.errorbar(x, stats["psnr"][:, 0], yerr=stats["psnr"][:, 1], color="#1f4e79", marker="o", ms=4, lw=1.5, capsize=3, label="PSNR")
    ax_psnr.set_xlabel("length a new stroke starts with (block units)")
    ax_psnr.set_ylabel("PSNR of the painting (dB)", color="#1f4e79")
    ax_l = ax_psnr.twinx()
    ax_l.errorbar(x, stats["lpips"][:, 0], yerr=stats["lpips"][:, 1], color="#b23a48", marker="s", ms=4, lw=1.2, ls="--", capsize=3, label="LPIPS")
    ax_l.set_ylabel("LPIPS (lower is better)", color="#b23a48")
    n_images = len({r["image"] for r in points[0][2]})
    n_seeds = len({r["seed"] for r in points[0][2]})
    ax_psnr.set_title(f"Marker paintings, {n_images} images x {n_seeds} seeds")
    for ax in (ax_vis, ax_psnr):
        ax.grid(alpha=0.25, lw=0.5)
    ax_vis.spines[["top", "right"]].set_visible(False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    for (c, name, _), (p, ps), (l, ls) in zip(points, stats["psnr"], stats["lpips"]):
        print(f"chord {c:4.2f} ({name}): PSNR {p:.2f} +- {ps:.2f}, LPIPS {l:.3f} +- {ls:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
