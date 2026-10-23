"""``scripts/run_paint.py``: one command from photo to painting. Random renderer weights: this checks the plumbing, not the look."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from neural_painter.core.image_io import save_image
from neural_painter.models.neural_renderer import NeuralRenderer, save_checkpoint
from neural_painter.pipeline.painter_engine import load_strokes_npz

ROOT = Path(__file__).resolve().parent.parent


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup(tmp_path):
    ckpt = save_checkpoint(tmp_path / "renderer.pt", NeuralRenderer("oil", light=True))
    image = tmp_path / "photo.png"
    save_image(image, np.random.default_rng(0).random((48, 64, 3), dtype=np.float32))  # not square: exercises the centre crop
    out = tmp_path / "run"
    argv = [str(image), "--brush", "oil", "--strokes", "8", "--grid", "fixed", "--max-divide", "2", "--size", "64",
            "--renderer", str(ckpt), "--device", "cpu", "--seed", "0", "--out", str(out)]
    return argv, out


def test_run_paint_writes_a_complete_experiment_folder(setup):
    argv, out = setup
    assert load_script("run_paint").main(argv) == 0
    for name in ("final.png", "input.png", "strokes.npz", "config.yaml", "results.csv"):
        assert (out / name).is_file(), name
    (row,) = csv.DictReader(open(out / "results.csv", encoding="utf-8"))
    assert row["strokes"] == "8" and row["grid"] == "fixed" and float(row["psnr"]) > 0 and -1 <= float(row["ssim"]) <= 1
    strokes, spec, _ = load_strokes_npz(out / "strokes.npz")
    assert strokes.shape == (8, 12) and strokes.min() >= 0 and strokes.max() <= 1 and spec.name == "oilpaintbrush"
    config = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
    assert config["strokes_requested"] == 8 and config["max_divide"] == 2 and config["painter"]["seed"] == 0 and "commit" in config


def test_run_paint_never_reuses_an_experiment_folder(setup):
    argv, _ = setup
    run_paint = load_script("run_paint")
    assert run_paint.main(argv) == 0
    with pytest.raises(SystemExit, match="not empty"):
        run_paint.main(argv)


def test_run_paint_refuses_a_missing_photo_before_making_a_folder(setup):
    argv, out = setup
    with pytest.raises(SystemExit, match="cannot paint"):
        load_script("run_paint").main(["no_such_photo.png", *argv[1:]])
    assert not out.exists()
