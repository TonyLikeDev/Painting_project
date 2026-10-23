"""Compare the four brush materials on the same images (Week 6): one table and one image grid from several ``run_ablation.py`` folders.

Each brush is painted by its own ``run_ablation.py`` call (a folder per brush, same images, seeds and stroke budget). This script
reads them side by side:

* ``summary.md`` / ``summary.csv`` in a new folder ``experiments/<date>_brushes_<name>/``: per brush the mean over seeds of the
  per-seed image average of PSNR, SSIM, LPIPS, optimization time and final-rendering time (plus or minus the standard deviation over
  seeds), and a per-image PSNR table (seeds averaged);
* ``<tag>.png`` in ``--out-dir``: rows are the images, columns the target and the painting of every brush (the seed ``--seed``),
  each labelled with its PSNR.

A brush is given as ``LABEL=FOLDER[#CONFIG]``; ``CONFIG`` picks one config of that folder (default: its only or first config).

Usage: venv/Scripts/python.exe scripts/compare_brushes.py "Oil=experiments/<oil folder>" "Marker=experiments/<folder>#c15" ...
       [--seed 0] [--images apple,jay] [--name brush_comparison] [--tag brush_comparison] [--out-dir report/figures]
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neural_painter.core.image_io import load_image  # noqa: E402
from neural_painter.pipeline.train_renderer import EXPERIMENT_ROOT, commit_info, new_experiment_dir  # noqa: E402
from summarize_ablation import column, per_seed_means, read_rows  # noqa: E402

METRICS = [("psnr", "PSNR (dB)", 2), ("ssim", "SSIM", 3), ("lpips", "LPIPS", 3), ("optimize_s", "optimize (s)", 1), ("render_s", "render (s)", 2)]


def parse_brush(spec: str) -> tuple[str, Path, str | None]:
    label, eq, rest = spec.partition("=")
    folder, _, config = rest.partition("#")
    if not eq or not label or not folder:
        raise argparse.ArgumentTypeError(f"a brush is LABEL=FOLDER[#CONFIG], got {spec!r}")
    path = Path(folder)
    return label, (path if path.is_absolute() else ROOT / path), config or None


def load(folder: Path, config: str | None) -> tuple[str, list[dict]]:
    rows = read_rows(folder)
    configs = list(dict.fromkeys(r["config"] for r in rows))
    config = config or configs[0]
    if config not in configs:
        raise SystemExit(f"{folder.name} has no config {config!r} (it has {configs})")
    return config, [r for r in rows if r["config"] == config]


def painting(run: Path, image: str) -> Path:
    """``final.png`` (this package) or ``<image>_final.png`` (the original 2021 demo, in ``compare_with_original.py``'s folders)."""
    return run / "final.png" if (run / "final.png").is_file() else run / f"{image}_final.png"


def mean_std(rows: list[dict], config: str, key: str) -> tuple[float, float] | None:
    v = per_seed_means(rows, config, key)
    if v.size == 0:
        return None
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("brushes", nargs="+", type=parse_brush, metavar="LABEL=FOLDER[#CONFIG]")
    ap.add_argument("--seed", type=int, default=0, help="seed of the paintings shown in the figure")
    ap.add_argument("--name", default="brush_comparison")
    ap.add_argument("--tag", default=None, help="figure file name (default: --name)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "report" / "figures")
    ap.add_argument("--eval-dir", type=Path, default=ROOT / "data" / "eval_set")
    ap.add_argument("--images", default="", help="comma-separated image names (or prefixes) to keep; default: all in common")
    args = ap.parse_args(argv)

    data = []  # (label, folder, config, rows)
    for label, folder, config in args.brushes:
        config, rows = load(folder, config)
        data.append((label, folder, config, rows))
    common = set.intersection(*({(r["image"], int(r["seed"])) for r in rows} for *_, rows in data))
    if args.images:
        wanted = [w for w in args.images.split(",") if w]
        common = {(i, s) for i, s in common if any(i == w or i.startswith(w) for w in wanted)}
    images, seeds = sorted({i for i, _ in common}), sorted({s for _, s in common})
    if not common:
        raise SystemExit("the folders have no (image, seed) run in common")
    missing = sorted(set((i, s) for i in images for s in seeds) - common)
    if missing:  # every image must have every seed, or the per-seed means would average different image sets
        raise SystemExit(f"incomplete comparison, missing (image, seed) runs in some folder: {missing}")
    data = [(label, folder, config, [r for r in rows if (r["image"], int(r["seed"])) in common]) for label, folder, config, rows in data]
    if args.seed not in seeds:
        raise SystemExit(f"--seed {args.seed} is not among the common seeds {seeds}")
    paintings = {(label, image): painting(folder / "runs" / image / f"{config}_s{args.seed}", image) for label, folder, config, _ in data for image in images}
    absent = [str(p) for p in paintings.values() if not p.is_file()]
    if absent:  # checked before the experiment folder is made, so a failed figure leaves nothing that looks finished
        raise SystemExit(f"paintings not found (the runs/ folders are not committed): {absent[:3]}")

    out = new_experiment_dir(EXPERIMENT_ROOT, args.name, kind="brushes")
    lines = ["# Brush materials on the same images", "",
             f"{len(images)} images ({', '.join(images)}), seeds {seeds}. Each cell is the mean over seeds of the per-seed image average, "
             "plus or minus the standard deviation over seeds. Times are seconds per painting.", "",
             "| Brush | Folder (config) | Strokes | " + " | ".join(m[1] for m in METRICS) + " |",
             "| :--- | :--- | ---: | " + " | ".join("---:" for _ in METRICS) + " |"]
    records = []
    for label, folder, config, rows in data:
        cells, record = [], {"brush": label, "folder": folder.name, "config": config}
        for key, _, digits in METRICS:
            ms = mean_std(rows, config, key)
            cells.append("-" if ms is None else f"{ms[0]:.{digits}f} +- {ms[1]:.{digits}f}")
            record[key], record[f"{key}_std"] = (None, None) if ms is None else ms
        strokes = int(np.nanmedian(column(rows, "strokes")))
        lines.append(f"| {label} | {folder.name} ({config}) | {strokes} | " + " | ".join(cells) + " |")
        records.append({**record, "strokes": strokes})
    lines += ["", "## PSNR (dB) per image, seeds averaged", "", "| Image | " + " | ".join(d[0] for d in data) + " |",
              "| :--- | " + " | ".join("---:" for _ in data) + " |"]
    for image in images:
        vals = [np.mean(column([r for r in rows if r["image"] == image], "psnr")) for *_, rows in data]
        lines.append(f"| {image} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (out / "config.yaml").write_text(yaml.safe_dump({"created": datetime.now().isoformat(timespec="seconds"), **commit_info(),
                                                     "brushes": [{"label": l, "folder": f.name, "config": c} for l, f, c, _ in data],
                                                     "images": images, "seeds": seeds, "figure_seed": args.seed}, sort_keys=False), encoding="utf-8")

    scores = {(d[0], r["image"], int(r["seed"])): float(r["psnr"]) for d in data for r in d[3]}
    fig, axes = plt.subplots(len(images), 1 + len(data), figsize=(2.4 * (1 + len(data)), 2.5 * len(images)), constrained_layout=True, squeeze=False)
    for i, image in enumerate(images):
        cells = [("target", load_image(args.eval_dir / f"{image}.png"))]
        for label, folder, config, _ in data:
            cells.append((f"{label}  {scores[(label, image, args.seed)]:.2f} dB", load_image(paintings[(label, image)])))
        for j, (title, img) in enumerate(cells):
            ax = axes[i, j]
            ax.imshow(img, interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[i, 0].set_ylabel(image[:14], fontsize=8)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.tag or args.name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("\n".join(lines))
    print(f"wrote {out} and {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
