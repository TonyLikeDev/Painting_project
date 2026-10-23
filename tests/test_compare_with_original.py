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
    assert compare.ORIGINAL_SETTINGS == {"resize": "linear", "init_size_range": (0.1, 0.25)}  # what ours_linear takes from the original
    assert rows["ours"]["psnr"] != rows["ours_linear"]["psnr"]  # different shrinking filter and start sizes, hence different paintings
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


@pytest.mark.skipif(not (ORIGINAL / "demo_prog.py").is_file(), reason="the original 2021 checkout is not present")
def test_progressive_mode_runs_demo_prog_seeded_and_ours_with_its_optimizer(tmp_path, monkeypatch):
    if not original_checkpoint_path("oilpaintbrush", light=True).is_file():
        pytest.skip("the pretrained light oil renderer has not been downloaded")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    compare = load_script("compare_with_original")
    monkeypatch.setattr(compare, "EXPERIMENT_ROOT", tmp_path)
    command = compare.original_command(ROOT / "x.png", tmp_path, 5, 500, "progressive", 2)
    assert command[3] == "demo_prog.py" and "--max_divide" in command and command[-2:] == ["--seed", "2"]
    assert "disable_power_throttling()" in command[2] and "runpy.run_path('demo_prog.py'" in command[2]
    assert "--seed" not in compare.original_command(ROOT / "x.png", tmp_path, 5, 500, "fixed", 2)  # demo.py cannot be seeded
    assert compare.ORIGINAL_OPTIMIZER["progressive"] == {"centered": True}
    assert compare.main(["--images", "apple", "--every", "1", "--repeats", "1", "--grid", "2", "--strokes", "10", "--device", "cpu",
                         "--mode", "progressive", "--name", "prog"]) == 0
    folder = next(tmp_path.glob("*_check_prog"))
    rows = {r["config"]: r for r in csv.DictReader(open(folder / "results.csv", encoding="utf-8"))}
    assert set(rows) == {"original", "ours", "ours_linear"} and all(r["strokes"] == "10" and r["mode"] == "progressive" for r in rows.values())
    assert rows["original"]["renderer"] == "original demo_prog.py"
    assert "centered: true" in (folder / "config.yaml").read_text(encoding="utf-8")


@pytest.mark.skipif(not (ORIGINAL / "demo_prog.py").is_file(), reason="the original 2021 checkout is not present")
def test_other_brushes_run_on_their_canvas_with_the_pipeline_settings(tmp_path, monkeypatch):
    """Week 9: ``--brush`` (the authors' renderer and the brush's canvas on both sides) and ``--configs`` with ``pipeline``."""
    if not original_checkpoint_path("rectangle", light=True).is_file():
        pytest.skip("the pretrained light tape renderer has not been downloaded")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    compare = load_script("compare_with_original")
    monkeypatch.setattr(compare, "EXPERIMENT_ROOT", tmp_path)
    command = compare.original_command(ROOT / "x.png", tmp_path, 5, 500, "progressive", 0, "rectangle", "black")
    assert command[command.index("--renderer") + 1] == "rectangle" and command[command.index("--canvas_color") + 1] == "black"
    assert command[command.index("--renderer_checkpoint_dir") + 1] == "checkpoints_G_rectangle_light"
    args = ["--images", "apple", "--every", "1", "--repeats", "1", "--grid", "2", "--strokes", "10", "--device", "cpu", "--mode", "progressive"]
    assert compare.main(args + ["--brush", "tape", "--configs", "original", "pipeline", "--name", "tape"]) == 0
    folder = next(tmp_path.glob("*_check_tape"))
    rows = {r["config"]: r for r in csv.DictReader(open(folder / "results.csv", encoding="utf-8"))}
    assert set(rows) == {"original", "pipeline"} and "light rectangle renderer, black canvas" in (folder / "config.yaml").read_text(encoding="utf-8")
    assert "lr_schedule: cosine" in (folder / "config.yaml").read_text(encoding="utf-8")
    from neural_painter.core.image_io import load_image

    assert load_image(folder / "runs" / "apple" / "pipeline_s0" / "final.png").min() == 0.0  # black canvas left between the strokes
    with pytest.raises(SystemExit, match="rectangle paintings"):
        compare.main(args + ["--brush", "oil", "--dir", str(folder)])
