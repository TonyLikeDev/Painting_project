"""The aesthetic survey of Week 9: the order of the paintings and the scoring of the filled sheets."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_order_mixes_images_and_brushes_and_can_be_made_again():
    survey = load_script("make_survey")
    order = survey.survey_order(9)
    assert len(order) == len(set(order)) == 20 and order == survey.survey_order(9)
    assert all(a[0] != b[0] and a[1] != b[1] for a, b in zip(order, order[1:]))  # no neighbours with the same image or brush


def test_the_summary_ranks_the_brushes_and_drops_incomplete_raters(tmp_path):
    survey, summary = load_script("make_survey"), load_script("survey_summary")
    order = survey.survey_order(9)
    liking = {"rectangle": 4.6, "oilpaintbrush": 3.8, "watercolor": 3.0, "markerpen": 2.0}
    with open(tmp_path / "key.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item", "image", "brush", "psnr", "ssim", "lpips"])
        for n, (image, brush) in enumerate(order, 1):
            w.writerow([n, image, brush, 30 - 2 * liking[brush] + n / 100, 0.9 - liking[brush] / 10, 0.2 + liking[brush] / 10])  # PSNR falls as the liking rises
    rng = np.random.default_rng(0)
    with open(tmp_path / "ratings.csv", "w", newline="", encoding="utf-8-sig") as fh:  # a Google Forms export: BOM, "Tranh n" titles
        w = csv.writer(fh)
        w.writerow(["Timestamp", "rater", "art"] + [f"Tranh {n}" for n in range(1, 21)])
        for r in range(12):
            scores = [int(np.clip(round(liking[b] + rng.normal(0, 0.6)), 1, 5)) for _, b in order]
            if r == 11:
                scores[4] = ""  # one painting left out: this rater is dropped
            w.writerow(["2026-10-21", f"R{r + 1}", "co" if r < 4 else "khong"] + scores)
    assert summary.main([str(tmp_path / "ratings.csv"), "--key", str(tmp_path / "key.csv")]) == 0
    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "11 complete raters, 1 dropped" in text
    table = [line.split("|")[1].strip() for line in text.split("## Per brush")[1].splitlines() if line.startswith("| ") and "Mean" not in line and ":---" not in line]
    assert table[:4] == ["Coloured tape", "Oil paint", "Watercolour", "Marker pen"]
    friedman_p = float(text.split("Friedman test")[1].split("p = ")[1].split(".\n")[0])
    assert friedman_p < 0.001
    assert "| PSNR | -" in text and "Art (n = 4) | No art (n = 7)" in text
    assert summary.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
