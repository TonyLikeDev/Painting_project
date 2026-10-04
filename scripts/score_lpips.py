"""Add LPIPS to the results of an ablation after the fact (Week 5).

``run_ablation.py`` leaves the ``lpips`` column empty unless it was started with ``--lpips``, which needs the AlexNet weights
(about 230 MB, cached by torchvision on the first download). The paintings themselves are kept
(``runs/<image>/<config>_s<seed>/final.png``), so the metric can be computed later from them and the frozen evaluation image
and written into ``results.csv``. Rows that already have a value are left alone, so the script is safe to repeat. It refuses to
download the weights unless ``--allow-download`` is given.

Usage: venv/Scripts/python.exe scripts/score_lpips.py experiments/<folder> [--device auto] [--eval-dir data/eval_set] [--allow-download]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neural_painter.core.device import get_device  # noqa: E402
from neural_painter.core.image_io import load_image  # noqa: E402
from neural_painter.pipeline.metrics import LPIPSMetric  # noqa: E402


def painting_path(folder: Path, row: dict) -> Path:
    """The saved painting of a run: ``final.png`` (this package), or ``<image>_final.png`` (the original 2021 demo)."""
    run = folder / "runs" / row["image"] / f"{row['config']}_s{row['seed']}"
    ours = run / "final.png"
    return ours if ours.is_file() else run / f"{row['image']}_final.png"


def score_folder(folder: Path, metric, eval_dir: Path) -> tuple[int, int]:
    """Fill the empty ``lpips`` cells of ``folder/results.csv``; returns (scored, already had a value)."""
    path = folder / "results.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields, rows = list(reader.fieldnames or []), list(reader)
    if "lpips" not in fields:
        raise SystemExit(f"{path} has no lpips column")
    scored = kept = 0
    for row in rows:
        if row["lpips"] != "":
            kept += 1
            continue
        row["lpips"] = f"{metric(load_image(painting_path(folder, row)), load_image(eval_dir / (row['image'] + '.png'))):.6g}"
        scored += 1
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)  # one rename: an interrupted run never leaves half a table
    return scored, kept


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-dir", type=Path, default=ROOT / "data" / "eval_set")
    ap.add_argument("--allow-download", action="store_true", help="let LPIPS download the AlexNet weights if they are not cached")
    args = ap.parse_args(argv)
    metric = LPIPSMetric(get_device(args.device), allow_download=args.allow_download)
    scored, kept = score_folder(args.folder, metric, args.eval_dir)
    print(f"{args.folder.name}: {scored} runs scored, {kept} already had LPIPS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
