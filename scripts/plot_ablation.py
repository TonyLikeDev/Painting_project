"""Figures for an ablation written by ``run_ablation.py`` (Week 5).

* ``<tag>_qualitative.png``: the target and the final painting of every config for three images that a rule, not the eye,
  picks: the image where ``--other`` gains most over ``--baseline`` (PSNR with the seeds averaged), the median image and the
  image where it loses most. The paintings shown are those of ``--seed``.
* ``<tag>_convergence.png``: PSNR of the neural canvas against the optimizer step. Left: the mean over images (seeds averaged
  first) of every config, with a 95 % confidence band; right: the paired difference ``other - baseline``. The neural canvas has
  the size of the grid times the renderer's block, so curves of different grids are not comparable with each other.

Both need the per-run folders ``runs/<image>/<config>_s<seed>/`` that ``run_ablation.py`` writes (they are git-ignored).

Usage: venv/Scripts/python.exe scripts/plot_ablation.py experiments/<folder> --baseline pixel --other ot
       [--seed 0] [--configs pixel ot] [--out-dir report/figures] [--tag NAME] [--eval-dir data/eval_set]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neural_painter.core.image_io import load_image  # noqa: E402
from summarize_ablation import paired_by_unit, read_rows  # noqa: E402

COLORS = ["#1f4e79", "#b23a48", "#2a9d8f", "#e9a23b", "#6d597a", "#888888"]


def config_order(rows: list[dict]) -> list[str]:
    seen: list[str] = []
    for r in rows:
        if r["config"] not in seen:
            seen.append(r["config"])
    return seen


def pick_images(rows: list[dict], baseline: str, other: str, key: str = "psnr") -> list[tuple[str, str, float]]:
    """``(kind, image, difference)`` for the images where ``other`` gains most over ``baseline``, the median one and the worst.

    Images are ranked by the seed-averaged difference ``other - baseline``; for an even number of images the median is the
    upper of the two middle ones. Fewer than three images give fewer pictures.
    """
    diffs = sorted(paired_by_unit(rows, baseline, other, key).items(), key=lambda kv: (-kv[1], kv[0]))
    if not diffs:
        return []
    chosen = [("best", *diffs[0]), ("median", *diffs[len(diffs) // 2]), ("worst", *diffs[-1])]
    out, used = [], set()
    for kind, image, d in chosen:
        if image not in used:
            used.add(image)
            out.append((kind, image, d))
    return out


def qualitative(folder: Path, rows: list[dict], configs: list[str], picks: list[tuple[str, str, float]], seed: int, eval_dir: Path) -> plt.Figure:
    fig, axes = plt.subplots(len(picks), 1 + len(configs), figsize=(2.6 * (1 + len(configs)), 2.75 * len(picks)), constrained_layout=True, squeeze=False)
    scores = {(r["image"], r["config"], int(r["seed"])): float(r["psnr"]) for r in rows}
    for i, (kind, image, diff) in enumerate(picks):
        cells = [("target", load_image(eval_dir / f"{image}.png"))]
        for c in configs:
            run = folder / "runs" / image / f"{c}_s{seed}" / "final.png"
            psnr = scores.get((image, c, seed))
            cells.append((c if psnr is None else f"{c}  {psnr:.2f} dB", load_image(run)))
        for j, (label, img) in enumerate(cells):
            ax = axes[i, j]
            ax.imshow(img, interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(label, fontsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[i, 0].set_ylabel(f"{kind}: {image[:8]}\n({diff:+.2f} dB)", fontsize=8)
    return fig


def history_matrix(folder: Path, config: str, images: list[str], seeds: list[int]) -> np.ndarray:
    """``(images, steps)``: neural-canvas PSNR per optimizer step, the seeds of each image averaged."""
    rows = []
    for image in images:
        runs = [np.load(folder / "runs" / image / f"{config}_s{s}" / "history.npz")["psnr"] for s in seeds]
        steps = min(len(r) for r in runs)
        rows.append(np.mean([r[:steps] for r in runs], axis=0))
    steps = min(len(r) for r in rows)
    return np.stack([r[:steps] for r in rows])


def band(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean over images and the half width of the 95 % confidence interval (t distribution), per step."""
    n = matrix.shape[0]
    half = stats.t.ppf(0.975, n - 1) * matrix.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(matrix.shape[1])
    return matrix.mean(axis=0), half


def convergence(folder: Path, rows: list[dict], configs: list[str], baseline: str, other: str) -> plt.Figure:
    images = sorted({r["image"] for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    curves = {c: history_matrix(folder, c, images, seeds) for c in configs}
    steps = min(m.shape[1] for m in curves.values())
    x = np.arange(1, steps + 1)
    fig, (ax_psnr, ax_diff) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for k, c in enumerate(configs):
        mean, half = band(curves[c][:, :steps])
        ax_psnr.plot(x, mean, color=COLORS[k % len(COLORS)], lw=1.5, label=c)
        ax_psnr.fill_between(x, mean - half, mean + half, color=COLORS[k % len(COLORS)], alpha=0.18, lw=0)
    ax_psnr.set_xlabel("optimizer step")
    ax_psnr.set_ylabel("PSNR of the neural canvas (dB)")
    ax_psnr.set_title(f"Mean over {len(images)} images, {len(seeds)} seeds")
    ax_psnr.legend(loc="lower right", frameon=False, fontsize=8)
    for c in configs:
        if c == baseline:
            continue
        mean, half = band(curves[c][:, :steps] - curves[baseline][:, :steps])
        k = configs.index(c)
        ax_diff.plot(x, mean, color=COLORS[k % len(COLORS)], lw=1.5, label=f"{c} - {baseline}")
        ax_diff.fill_between(x, mean - half, mean + half, color=COLORS[k % len(COLORS)], alpha=0.18, lw=0)
    ax_diff.axhline(0.0, color="#444444", lw=0.8)
    ax_diff.set_xlabel("optimizer step")
    ax_diff.set_ylabel("paired difference (dB)")
    ax_diff.set_title(f"Against {baseline}, paired by image (95 % CI)")
    ax_diff.legend(loc="best", frameon=False, fontsize=8)
    for ax in (ax_psnr, ax_diff):
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(folder.name, fontsize=10)
    return fig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", type=Path, help="experiment folder written by run_ablation.py")
    ap.add_argument("--baseline", default=None, help="config the others are compared with (default: the first one)")
    ap.add_argument("--other", default=None, help="config whose gain ranks the images in the qualitative figure (default: the last one)")
    ap.add_argument("--configs", nargs="+", default=None, help="configs to show (default: all)")
    ap.add_argument("--seed", type=int, default=None, help="seed of the paintings in the qualitative figure (default: the smallest)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "report" / "figures")
    ap.add_argument("--tag", default=None, help="file name prefix (default: the folder name without its date)")
    ap.add_argument("--eval-dir", type=Path, default=ROOT / "data" / "eval_set")
    ap.add_argument("--skip", nargs="*", choices=["qualitative", "convergence"], default=[], help="figures not to draw")
    args = ap.parse_args(argv)

    rows = read_rows(args.folder)
    configs = args.configs or config_order(rows)
    baseline = args.baseline or configs[0]
    other = args.other or configs[-1]
    seed = args.seed if args.seed is not None else min(int(r["seed"]) for r in rows)
    tag = args.tag or re.sub(r"^\d{4}-\d{2}-\d{2}_", "", args.folder.name)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if "qualitative" not in args.skip:
        picks = pick_images(rows, baseline, other)
        fig = qualitative(args.folder, rows, configs, picks, seed, args.eval_dir)
        path = args.out_dir / f"{tag}_qualitative.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"wrote {path}: " + ", ".join(f"{k} {i[:8]} ({d:+.2f} dB)" for k, i, d in picks))
    if "convergence" not in args.skip:
        fig = convergence(args.folder, rows, configs, baseline, other)
        path = args.out_dir / f"{tag}_convergence.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
