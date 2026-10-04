"""The comparison with the original 2021 code: a tiny end-to-end run of both implementations on the CPU."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

from neural_painter.models.neural_renderer import original_checkpoint_path

ROOT = Path(__file__).resolve().parent.parent
ORIGINAL = ROOT / "stylized-neural-painting"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not (ORIGINAL / "demo.py").is_file(), reason="the original 2021 checkout is not present")
def test_both_implementations_paint_the_same_image_and_are_scored_alike(tmp_path, monkeypatch, capsys):
    if not original_checkpoint_path("oilpaintbrush", light=True).is_file():
        pytest.skip("the pretrained light oil renderer has not been downloaded")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # the original picks cuda when it can: keep this smoke test on the CPU
    compare = load_script("compare_with_original")
    monkeypatch.setattr(compare, "EXPERIMENT_ROOT", tmp_path)
    args = ["--images", "apple", "--every", "1", "--repeats", "1", "--grid", "2", "--strokes", "8", "--device", "cpu"]

    assert compare.main(args + ["--name", "smoke"]) == 0
    folder = next(tmp_path.glob("*_check_smoke"))
    rows = {r["config"]: r for r in csv.DictReader(open(folder / "results.csv", encoding="utf-8"))}
    assert set(rows) == {"original", "ours", "ours_linear"}
    for r in rows.values():
        assert 0 < float(r["psnr"]) < 100 and 0 < float(r["ssim"]) <= 1 and float(r["wall_s"]) > 0 and r["strokes"] == "8"
    assert rows["original"]["device"] == "cpu" and rows["original"]["optimize_s"] == "" and float(rows["ours"]["optimize_s"]) > 0
    assert rows["ours"]["psnr"] != rows["ours_linear"]["psnr"]  # the two shrinking filters give different targets, hence different paintings
    run = folder / "runs" / "apple" / "original_s0"
    assert (run / "apple_final.png").is_file() and (run / "apple_strokes.npz").is_file() and (run / "log.txt").is_file()
    assert not list(run.glob("*_rendered_stroke_*.png")) and not list(run.glob("*.mp4"))  # frames and video are deleted after use
    assert (folder / "runs" / "apple" / "ours_s0" / "final.png").is_file()
    assert "(complete)" in capsys.readouterr().out

    assert compare.main(args + ["--dir", str(folder)]) == 0  # resuming a finished comparison repeats nothing
    assert len(list(csv.DictReader(open(folder / "results.csv", encoding="utf-8")))) == 3
    with pytest.raises(SystemExit, match="too few"):
        compare.main(["--grid", "5", "--strokes", "10", "--name", "x", "--device", "cpu"])
    with pytest.raises(SystemExit, match="does not exist"):
        compare.main(args + ["--dir", str(tmp_path / "missing")])
