"""Paired comparison of one config in one ``run_ablation.py`` folder against one config in another (Week 6).

``summarize_ablation.py`` pairs configs inside one folder. Some comparisons cannot be run as two configs of one call, for
example the same settings painted with two different renderers (a renderer is chosen per call). This script pairs the runs
of ``--base FOLDER#CONFIG`` and ``--other FOLDER#CONFIG`` by (image, seed), averages the seeds of each image, and reports the
same statistics as ``summarize_ablation.py``: mean difference, 95 % confidence interval over the images, share of images
won, Wilcoxon p. Only (image, seed) runs present in both are used, and the script says how many.

Usage: venv/Scripts/python.exe scripts/paired_folders.py --base experiments/<a>#low --other experiments/<b>#paint [--label NAME]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from summarize_ablation import METRICS, read_rows  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def side(spec: str) -> dict[tuple[str, int], dict]:
    folder, _, config = spec.partition("#")
    path = Path(folder) if Path(folder).is_absolute() else ROOT / folder
    rows = [r for r in read_rows(path) if not config or r["config"] == config]
    if not rows:
        raise SystemExit(f"no runs for {spec!r}")
    return {(r["image"], int(r["seed"])): r for r in rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="FOLDER#CONFIG the other is compared with")
    ap.add_argument("--other", required=True, help="FOLDER#CONFIG")
    ap.add_argument("--label", default="other - base")
    args = ap.parse_args(argv)
    base, other = side(args.base), side(args.other)
    keys = sorted(set(base) & set(other))
    images = sorted({i for i, _ in keys})
    print(f"{args.label}: {len(keys)} paired runs on {len(images)} images ({len(base)} and {len(other)} runs given)")
    print("| Metric | Base | Other | Mean difference | 95 % CI | Wins | Wilcoxon p |")
    print("| :--- | ---: | ---: | ---: | :--- | ---: | ---: |")
    for key in ("psnr", "ssim", "lpips", "neural_psnr"):
        if any(base[k][key] == "" or other[k][key] == "" for k in keys):
            continue
        d = np.array([np.mean([float(other[k][key]) - float(base[k][key]) for k in keys if k[0] == i]) for i in images])
        b = np.mean([float(base[k][key]) for k in keys])
        o = np.mean([float(other[k][key]) for k in keys])
        half = stats.t.ppf(0.975, d.size - 1) * d.std(ddof=1) / np.sqrt(d.size) if d.size > 1 else float("nan")
        wins = float(np.mean(d > 0 if METRICS[key][1] else d < 0))
        p = float(stats.wilcoxon(d).pvalue) if d.size > 1 and np.any(d != 0) else float("nan")
        print(f"| {METRICS[key][0]} | {b:.3f} | {o:.3f} | {d.mean():+.3f} | [{d.mean() - half:+.3f}, {d.mean() + half:+.3f}] | "
              f"{wins:.0%} of {d.size} | {p:.3g} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
