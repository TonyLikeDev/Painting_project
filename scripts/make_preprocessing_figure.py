"""Figure for the dataset / preprocessing subsection (Week 3).

Four panels: (a) the input photo with the centre-crop box, (b) the 512 x 512 crop,
(c) the normalized float tensor shown per channel with its value range, (d) the
4 x 4 grid split with block indices in the row-major order used by ``core.grid``.

Usage: ``venv/Scripts/python.exe scripts/make_preprocessing_figure.py [image] [--grid 4]``
Writes ``report/figures/preprocessing_pipeline.png`` and ``.svg``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neural_painter.core import grid, image_io  # noqa: E402

DEFAULT_IMAGE = ROOT / "stylized-neural-painting" / "test_images" / "iceland.jpg"
OUT = ROOT / "report" / "figures" / "preprocessing_pipeline"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=str(DEFAULT_IMAGE))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--grid", type=int, default=4)
    args = ap.parse_args(argv)

    img = image_io.load_image(args.image)
    pre = image_io.preprocess(img, args.size, "crop")
    t = pre.tensor[0].numpy()  # (3, S, S)
    m = args.grid

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), constrained_layout=True)

    ax = axes[0]
    ax.imshow(img)
    left, top, right, bottom = pre.crop_box
    ax.add_patch(patches.Rectangle((left, top), right - left, bottom - top, fill=False, ec="#e53935", lw=2))
    ax.set_title(f"(a) input {pre.original_size[0]} x {pre.original_size[1]}, EXIF-corrected\ncentre-crop box (red)", fontsize=10)

    ax = axes[1]
    ax.imshow(pre.image)
    ax.set_title(f"(b) centre crop, INTER_AREA resize to {args.size} x {args.size}", fontsize=10)

    ax = axes[2]
    # 2 x 2 montage (R, G / B, recomposed) so the panel stays square like the others
    s = args.size
    montage = np.zeros((2 * s, 2 * s, 3), np.float32)
    montage[:s, :s] = t[0][..., None]
    montage[:s, s:] = t[1][..., None]
    montage[s:, :s] = t[2][..., None]
    montage[s:, s:] = pre.image
    ax.imshow(montage)
    ax.axhline(s, color="white", lw=1.5)
    ax.axvline(s, color="white", lw=1.5)
    for (px, py), name in zip([(0, 0), (s, 0), (0, s), (s, s)], ["R", "G", "B", "RGB"]):
        ax.text(px + 14, py + 14, name, color="white", ha="left", va="top", fontsize=11, weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="#c62828", ec="none"))
    ax.set_title(f"(c) float32 tensor (1, 3, {s}, {s}) in [0, 1]\nmin {t.min():.3f}  max {t.max():.3f}  mean {t.mean():.3f}", fontsize=10)

    ax = axes[3]
    ax.imshow(grid.draw_grid_lines(pre.image, m, color=(0.9, 0.2, 0.2), thickness=3))
    for k, (l, tt, r, b) in enumerate(grid.grid_boxes(args.size, args.size, m)):
        ax.text(l + 8, tt + 8, str(k), color="white", ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.15", fc="#c62828", ec="none"))
    ax.set_title(f"(d) {m} x {m} grid split, {m * m} blocks of {args.size // m} px\nrow-major block index", fontsize=10)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=150)
    fig.savefig(OUT.with_suffix(".svg"))
    print(f"wrote {OUT.with_suffix('.png')} and .svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
