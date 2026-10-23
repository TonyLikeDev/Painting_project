"""Score the aesthetic survey of Week 9 (``scripts/make_survey.py``): which brush do people like, and does PSNR predict it?

``ratings.csv`` has one row per rater: ``rater``, ``art`` (``co`` / ``khong``, or yes / no: has the rater studied art or worked in
it), and one column per painting whose header holds its survey number (``1`` to ``20``, as in
``experiments/aesthetic_survey_template/ratings.csv``; a Google Forms export with questions titled ``Tranh 1`` ... works as is).
Raters who left a painting out are dropped and counted.

Writes ``summary.md`` next to the ratings: per brush the mean of the raters' brush means with its standard deviation and 95 %
confidence interval and the share of 4 and 5 answers; the Friedman test over the four brushes (raters as blocks) and pairwise
Wilcoxon signed-rank tests, Holm-corrected; the mean per image and brush; the Spearman correlation of the 20 paintings' mean
scores with their PSNR, SSIM and LPIPS (from ``key.csv``); and the brush means of raters with and without art experience.

Usage: venv/Scripts/python.exe scripts/survey_summary.py experiments/<date>_aesthetic_survey/ratings.csv [--key report/notes/aesthetic_survey/key.csv]
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
from itertools import combinations
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
NAMES = {"oilpaintbrush": "Oil paint", "watercolor": "Watercolour", "markerpen": "Marker pen", "rectangle": "Coloured tape"}
YES, NO = {"co", "có", "yes", "y", "1", "true"}, {"khong", "không", "no", "n", "0", "false"}


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:  # utf-8-sig: Excel's and Google's CSV exports may start with a BOM
        return [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in csv.DictReader(fh)]


def item_columns(header: list[str], n: int) -> dict[int, str]:
    """Survey number -> column: the first whole number in the header, for the columns other than rater / art."""
    found = {}
    for column in header:
        if column.lower() in ("rater", "art"):
            continue
        m = re.search(r"\d+", column)
        if m and 1 <= int(m.group()) <= n and int(m.group()) not in found:
            found[int(m.group())] = column
    if len(found) != n:
        raise SystemExit(f"the ratings have columns for {sorted(found)}, the key has {n} paintings")
    return found


def ci95(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    half = stats.t.ppf(0.975, len(values) - 1) * statistics.stdev(values) / len(values) ** 0.5
    return statistics.mean(values) - half, statistics.mean(values) + half


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def summarize(ratings_path: Path, key_path: Path) -> str:
    key = {int(r["item"]): r for r in read_csv(key_path)}
    rows = read_csv(ratings_path)
    if not rows:
        raise SystemExit(f"{ratings_path} has no ratings")
    columns = item_columns(list(rows[0]), len(key))
    raters, dropped = [], 0
    for i, row in enumerate(rows, 1):
        try:
            scores = {n: int(row[c]) for n, c in columns.items()}
        except ValueError:
            dropped += 1
            continue
        if not all(1 <= s <= 5 for s in scores.values()):
            raise SystemExit(f"rater {row.get('rater') or i}: scores must be 1 to 5")
        art = row.get("art", "").lower()
        raters.append({"rater": row.get("rater") or f"row {i}", "art": "yes" if art in YES else "no" if art in NO else "", "scores": scores})
    if len(raters) < 2:
        raise SystemExit("fewer than two complete raters")
    brushes = [b for b in NAMES if any(k["brush"] == b for k in key.values())]
    images = list(dict.fromkeys(k["image"] for k in key.values()))

    def brush_mean(rater: dict, brush: str) -> float:
        return statistics.mean(s for n, s in rater["scores"].items() if key[n]["brush"] == brush)

    means = {b: [brush_mean(r, b) for r in raters] for b in brushes}
    out = [f"# Aesthetic survey: {len(raters)} raters", "",
           f"Ratings: `{ratings_path.as_posix()}`; key: `{key_path.as_posix()}`. {len(raters)} complete raters"
           + (f", {dropped} dropped for missing answers" if dropped else "") + ". Scores 1 (not beautiful at all) to 5 (very beautiful).", "",
           "## Per brush", "", "Mean over raters of each rater's mean over the brush's five paintings.", "",
           "| Brush | Mean | SD | 95 % CI | Share of 4 and 5 |", "| :--- | ---: | ---: | :--- | ---: |"]
    for b in sorted(brushes, key=lambda b: -statistics.mean(means[b])):
        lo, hi = ci95(means[b])
        answers = [s for r in raters for n, s in r["scores"].items() if key[n]["brush"] == b]
        out.append(f"| {NAMES[b]} | {statistics.mean(means[b]):.2f} | {statistics.stdev(means[b]):.2f} | [{lo:.2f}, {hi:.2f}] | "
                   f"{sum(s >= 4 for s in answers) / len(answers):.0%} |")
    chi2, p = stats.friedmanchisquare(*(means[b] for b in brushes))
    out += ["", f"Friedman test over the {len(brushes)} brushes (raters as blocks): chi2 = {chi2:.2f}, p = {p:.3g}.", "",
            "| Pair | Mean difference | Wilcoxon p | Holm p |", "| :--- | ---: | ---: | ---: |"]
    pairs = list(combinations(brushes, 2))
    raw = []
    for a, b in pairs:
        diff = [x - y for x, y in zip(means[a], means[b])]
        raw.append(1.0 if not any(diff) else float(stats.wilcoxon(means[a], means[b]).pvalue))
    for (a, b), p_raw, p_holm in zip(pairs, raw, holm(raw)):
        out.append(f"| {NAMES[a]} - {NAMES[b]} | {statistics.mean(means[a]) - statistics.mean(means[b]):+.2f} | {p_raw:.3g} | {p_holm:.3g} |")

    item_mean = {n: statistics.mean(r["scores"][n] for r in raters) for n in key}
    out += ["", "## Per image and brush", "", "Mean score of each painting (survey number in brackets).", "",
            "| Image | " + " | ".join(NAMES[b] for b in brushes) + " |", "| :--- |" + " ---: |" * len(brushes)]
    for image in images:
        cells = [next(f"{item_mean[n]:.2f} ({n})" for n, k in key.items() if k["image"] == image and k["brush"] == b) for b in brushes]
        out.append(f"| {image[:24]} | " + " | ".join(cells) + " |")

    out += ["", "## Do the metrics predict the scores?", "", "Spearman correlation over the 20 paintings of the mean score with each metric "
            "(PSNR and SSIM: higher is closer to the photo; LPIPS: lower is closer).", "", "| Metric | rho | p |", "| :--- | ---: | ---: |"]
    for metric in ("psnr", "ssim", "lpips"):
        pairs_m = [(item_mean[n], float(k[metric])) for n, k in key.items() if k.get(metric)]
        if len(pairs_m) >= 3:
            rho, p_rho = stats.spearmanr(*zip(*pairs_m))
            out.append(f"| {metric.upper()} | {rho:+.2f} | {p_rho:.3g} |")

    groups = {g: [r for r in raters if r["art"] == g] for g in ("yes", "no")}
    if all(groups.values()):
        out += ["", "## Raters with and without art experience", "", "| Brush | Art (n = %d) | No art (n = %d) |" % (len(groups["yes"]), len(groups["no"])),
                "| :--- | ---: | ---: |"]
        for b in brushes:
            out.append(f"| {NAMES[b]} | " + " | ".join(f"{statistics.mean(brush_mean(r, b) for r in groups[g]):.2f}" for g in ("yes", "no")) + " |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ratings", type=Path)
    ap.add_argument("--key", type=Path, default=ROOT / "report" / "notes" / "aesthetic_survey" / "key.csv")
    args = ap.parse_args(argv)
    text = summarize(args.ratings, args.key)
    (args.ratings.parent / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
