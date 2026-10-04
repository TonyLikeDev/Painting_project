"""Summarize an ablation written by ``run_ablation.py``: means over seeds, paired comparisons, a Markdown table.

For each config and metric the headline number is the mean over seeds of the per-seed average over images, with the
standard deviation over seeds (RESEARCH_PLAN section 3.6). Against a baseline config the script also reports the paired
difference per image: the seeds of one image are averaged first (they are repeated measurements of the same photograph, not
independent photographs), so the sample size is the number of images. The table gives the mean difference, a 95 %
confidence interval (t distribution over the images), the share of images in which the config wins and a Wilcoxon
signed-rank p value.

Usage: venv/Scripts/python.exe scripts/summarize_ablation.py experiments/<folder> [--baseline pixel] [--labels pixel="Pixel loss" ot="+ Sinkhorn"]
Writes ``summary.md`` and ``summary.csv`` next to ``results.csv``.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# metric -> (label, higher is better)
METRICS = {"psnr": ("PSNR (dB)", True), "ssim": ("SSIM", True), "lpips": ("LPIPS", False), "neural_psnr": ("neural-canvas PSNR (dB)", True),
           "optimize_s": ("optimize (s)", False), "steps_to_22db": ("steps to 22 dB", False), "steps_to_24db": ("steps to 24 dB", False)}


def read_rows(folder: Path) -> list[dict]:
    with open(folder / "results.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def column(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) if r[key] != "" else np.nan for r in rows])


def per_seed_means(rows: list[dict], config: str, key: str) -> np.ndarray:
    """One number per seed: the mean of ``key`` over the images painted with ``config`` and that seed."""
    out = []
    for seed in sorted({int(r["seed"]) for r in rows}):
        v = column([r for r in rows if r["config"] == config and int(r["seed"]) == seed], key)
        if v.size and not np.isnan(v).all():
            out.append(np.nanmean(v))
    return np.array(out)


STEP_KEYS = ("steps_to_22db", "steps_to_24db")


def reached(rows: list[dict], config: str, key: str) -> tuple[list[float], int]:
    """The values of a steps-to-threshold column for the runs of ``config`` that reached it, and the number of runs."""
    runs = [r for r in rows if r["config"] == config]
    return [float(r[key]) for r in runs if r[key] != ""], len(runs)


def paired_by_unit(rows: list[dict], baseline: str, other: str, key: str, per: str = "image") -> dict[str, float]:
    """``other - baseline`` on the matched runs of the two configs, keyed by image (or by ``image/seed`` for ``per="run"``).

    ``per="image"`` averages the seeds of each image first, so there is one difference per image.
    """
    index = {(r["image"], int(r["seed"])): r for r in rows if r["config"] == baseline}
    pairs: dict[str, list[float]] = {}
    for r in rows:
        b = index.get((r["image"], int(r["seed"])))
        if r["config"] == other and b is not None and r[key] != "" and b[key] != "":
            pairs.setdefault(r["image"] if per == "image" else f"{r['image']}/{r['seed']}", []).append(float(r[key]) - float(b[key]))
    return {unit: float(np.mean(v)) for unit, v in pairs.items()}


def paired(rows: list[dict], baseline: str, other: str, key: str, per: str = "image") -> np.ndarray:
    """The values of :func:`paired_by_unit` as an array: one paired difference per image (default) or per (image, seed) run."""
    return np.array(list(paired_by_unit(rows, baseline, other, key, per).values()))


def summarize(rows: list[dict], baseline: str | None = None, labels: dict[str, str] | None = None) -> tuple[str, list[dict]]:
    """Markdown summary and one record per config. ``labels`` renames configs in the Markdown only (the records keep their names)."""
    configs = sorted({r["config"] for r in rows}, key=lambda c: [r["config"] for r in rows].index(c))
    baseline = baseline or configs[0]

    def show(config: str) -> str:
        return (labels or {}).get(config, config)

    n_images = len({r["image"] for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    lines = [f"# Ablation summary", "", f"{len(rows)} runs: {n_images} images x {len(configs)} configs x {len(seeds)} seeds {seeds}. "
             "Each cell is the mean over seeds of the per-seed image average, plus or minus the standard deviation over seeds.", ""]
    shown = [k for k in METRICS if any(r[k] != "" for r in rows)]
    lines += ["| Config | " + " | ".join(METRICS[k][0] for k in shown) + " |", "| :--- | " + " | ".join("---:" for _ in shown) + " |"]
    csv_rows = []
    for c in configs:
        cells, record = [], {"config": c}
        for k in shown:
            if k in STEP_KEYS:  # a threshold that is never reached has no step: report the median of those that do, and how many
                hit, n_runs = reached(rows, c, k)
                cells.append(f"{np.median(hit):.0f} ({len(hit)}/{n_runs})" if hit else f"- (0/{n_runs})")
                record[k] = float(np.median(hit)) if hit else None
                continue
            v = per_seed_means(rows, c, k)
            cells.append("-" if v.size == 0 else (f"{v.mean():.2f} +- {v.std(ddof=1) if v.size > 1 else 0.0:.2f}" if k != "ssim" and k != "lpips"
                                                   else f"{v.mean():.3f} +- {v.std(ddof=1) if v.size > 1 else 0.0:.3f}"))
            record[k] = float(v.mean()) if v.size else None
        lines.append(f"| {show(c)} | " + " | ".join(cells) + " |")
        csv_rows.append(record)
    if any(k in STEP_KEYS for k in shown):
        lines += ["", "Steps to a PSNR threshold: median over the runs whose neural canvas reaches it, with (runs that do / runs)."]
    lines += ["", f"## Paired against `{show(baseline)}` (one difference per image, seeds averaged)", "",
              "| Config | Metric | Mean difference | 95 % CI | Wins | Wilcoxon p |", "| :--- | :--- | ---: | :--- | ---: | ---: |"]
    for c in configs:
        if c == baseline:
            continue
        for k in [m for m in ("psnr", "ssim", "lpips") if m in shown]:
            d = paired(rows, baseline, c, k)
            if d.size < 2:
                continue
            better = METRICS[k][1]
            wins = float(np.mean(d > 0 if better else d < 0))
            half = stats.t.ppf(0.975, d.size - 1) * d.std(ddof=1) / np.sqrt(d.size)
            p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            lines.append(f"| {show(c)} | {METRICS[k][0]} | {d.mean():+.3f} | [{d.mean() - half:+.3f}, {d.mean() + half:+.3f}] | {wins:.0%} of {d.size} | {p:.3g} |")
    return "\n".join(lines) + "\n", csv_rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--baseline", default=None, help="config the others are compared with (default: the first one)")
    ap.add_argument("--labels", nargs="*", default=[], metavar="CONFIG=TEXT", help='names to show in summary.md, e.g. pixel="Pixel loss"')
    args = ap.parse_args(argv)
    labels = dict(item.split("=", 1) for item in args.labels if "=" in item)
    rows = read_rows(args.folder)
    text, csv_rows = summarize(rows, args.baseline, labels)
    (args.folder / "summary.md").write_text(text, encoding="utf-8")
    with open(args.folder / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
