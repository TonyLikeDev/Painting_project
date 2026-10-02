"""Plot the learning curve of a renderer training run (Week 4 figure).

Reads ``curve.csv`` from an experiment folder written by ``pipeline/train_renderer.py`` and draws two panels: the
training loss (log scale) and the validation PSNR on the held-out strokes (foreground, opacity and their mean). Dotted
vertical lines mark the learning-rate drops; an optional dashed horizontal line marks a reference, for example the
pretrained renderer's score from ``scripts/renderer_fidelity.py``.

Usage: venv/Scripts/python.exe scripts/plot_training_curve.py experiments/<run> [--reference 24.38 --reference-label "original light"] [--out PATH]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402


def read_curve(folder: Path) -> dict[str, list[float]]:
    with open(folder / "curve.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{folder / 'curve.csv'} has no rows")
    return {key: [float(r[key]) for r in rows] for key in rows[0]}


def plot(curve: dict[str, list[float]], title: str, reference: float | None = None, reference_label: str = "reference") -> plt.Figure:
    epochs = curve["epoch"]
    drops = [e for e, prev, lr in zip(epochs[1:], curve["lr"][:-1], curve["lr"][1:]) if lr < prev]  # first epoch trained at the lower rate
    fig, (ax_loss, ax_psnr) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)

    ax_loss.plot(epochs, curve["train_loss"], color="#1f4e79", lw=1.6)
    ax_loss.set_yscale("log")
    plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")  # 0.5, 1, 2 rather than 5 x 10^-1
    ax_loss.yaxis.set_major_formatter(plain)
    ax_loss.yaxis.set_minor_formatter(plain)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("training loss (log scale)")
    ax_loss.set_title("Training loss")

    ax_psnr.plot(epochs, curve["psnr_fg"], color="#2a9d8f", lw=1.2, label="foreground")
    ax_psnr.plot(epochs, curve["psnr_alpha"], color="#e9a23b", lw=1.2, label="opacity")
    ax_psnr.plot(epochs, curve["psnr_mean"], color="#1f4e79", lw=1.8, label="mean")
    if reference is not None:
        ax_psnr.axhline(reference, color="#b23a48", ls="--", lw=1.2, label=f"{reference_label} ({reference:.2f} dB)")
    ax_psnr.set_xlabel("epoch")
    ax_psnr.set_ylabel("PSNR on held-out strokes (dB)")
    ax_psnr.set_title("Validation fidelity")
    ax_psnr.legend(loc="lower right", frameon=False, fontsize=8)

    for ax in (ax_loss, ax_psnr):
        for e in drops:
            ax.axvline(e, color="#888888", ls=":", lw=0.9)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=10)
    return fig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", type=Path, help="experiment folder containing curve.csv")
    ap.add_argument("--reference", type=float, default=None, help="draw a dashed line at this PSNR (dB)")
    ap.add_argument("--reference-label", default="original renderer")
    ap.add_argument("--out", type=Path, default=None, help="PNG to write (default: <folder>/training_curve.png)")
    args = ap.parse_args(argv)

    curve = read_curve(args.folder)
    out = args.out or args.folder / "training_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plot(curve, args.folder.name, args.reference, args.reference_label)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
