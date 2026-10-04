"""Dose-response figure for a sweep written by ``run_ablation.py`` (Week 5).

For every config except the baseline it draws the paired difference to the baseline (one difference per image, the seeds of an image
averaged first) against the value that was swept, on a logarithmic axis: the mean with its 95 % confidence interval, and one small dot
per image so that the spread is visible. Two panels, PSNR and SSIM; the baseline is the zero line. The swept value is read from the
``beta_ot`` or ``ot_epsilon`` column of ``results.csv``, which every row carries.

Usage: venv/Scripts/python.exe scripts/plot_sweep.py experiments/<folder> --baseline pixel --x beta_ot|ot_epsilon
       [--configs a b c] [--only-config-prefix ot] [--out PATH] [--xlabel TEXT]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from summarize_ablation import paired_by_unit, read_rows  # noqa: E402

PANELS = (("psnr", "PSNR difference (dB)"), ("ssim", "SSIM difference"))
XLABELS = {"beta_ot": "Sinkhorn weight $\\beta_{OT}$", "ot_epsilon": "Sinkhorn regularization $\\varepsilon$"}


def sweep_points(rows: list[dict], baseline: str, x_key: str, configs: list[str] | None = None) -> list[dict]:
    """One record per config: ``x`` (the swept value), and for each metric the per-image differences, their mean and the CI half width."""
    names: list[str] = []
    for r in rows:
        if r["config"] != baseline and r["config"] not in names and (configs is None or r["config"] in configs):
            names.append(r["config"])
    points = []
    for name in names:
        x = float(next(r[x_key] for r in rows if r["config"] == name))
        record = {"config": name, "x": x}
        for key, _ in PANELS:
            d = np.array(list(paired_by_unit(rows, baseline, name, key).values()))
            half = stats.t.ppf(0.975, d.size - 1) * d.std(ddof=1) / np.sqrt(d.size) if d.size > 1 else 0.0
            record[key] = {"diffs": d, "mean": float(d.mean()) if d.size else float("nan"), "half": float(half)}
        points.append(record)
    return sorted(points, key=lambda p: p["x"])


def plot(points: list[dict], title: str, x_key: str, xlabel: str | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, len(PANELS), figsize=(10, 3.8), constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, (key, label) in zip(axes, PANELS):
        xs = np.array([p["x"] for p in points])
        for p in points:
            d = p[key]["diffs"]
            ax.scatter(p["x"] * np.exp(rng.normal(0, 0.04, d.size)), d, s=9, color="#1f4e79", alpha=0.28, linewidths=0)
        ax.errorbar(xs, [p[key]["mean"] for p in points], yerr=[p[key]["half"] for p in points], color="#b23a48", marker="o", ms=5, lw=1.6,
                    capsize=3, label="mean and 95 % CI")
        ax.axhline(0.0, color="#444444", lw=0.9)
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.minorticks_off()
        ax.set_xlabel(xlabel or XLABELS.get(x_key, x_key))
        ax.set_ylabel(label)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="best", frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=10)
    return fig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--baseline", default="pixel")
    ap.add_argument("--x", default="beta_ot", choices=["beta_ot", "ot_epsilon"], help="the swept value")
    ap.add_argument("--configs", nargs="+", default=None, help="configs to draw (default: every one but the baseline)")
    ap.add_argument("--xlabel", default=None)
    ap.add_argument("--out", type=Path, default=None, help="PNG to write (default: report/figures/<folder name without date>.png)")
    args = ap.parse_args(argv)

    rows = read_rows(args.folder)
    points = sweep_points(rows, args.baseline, args.x, args.configs)
    if not points:
        raise SystemExit(f"{args.folder} has no config besides {args.baseline!r}")
    name = args.folder.name.split("_", 1)[1] if args.folder.name[:4].isdigit() else args.folder.name
    out = args.out or ROOT / "report" / "figures" / (f"{name}.png" if name.endswith("sweep") else f"{name}_sweep.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plot(points, f"{args.folder.name}: paired difference to {args.baseline}, per image", args.x, args.xlabel)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}: " + ", ".join(f"{p['config']} {p['psnr']['mean']:+.3f} dB" for p in points))
    return 0


if __name__ == "__main__":
    sys.exit(main())
