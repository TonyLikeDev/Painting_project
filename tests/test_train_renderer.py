"""Renderer training: the synthetic data stream, losses, metrics and a tiny end-to-end run."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from neural_painter.core.procedural_rasterizer import ProceduralRasterizer
from neural_painter.models.neural_renderer import NeuralRenderer, load_renderer
from neural_painter.pipeline.train_renderer import (
    SyntheticStrokes,
    TrainConfig,
    evaluate,
    init_like_original,
    make_validation_set,
    new_experiment_dir,
    reconstruction_loss,
    train,
)


# ----- data stream -----------------------------------------------------------------------
def test_stream_shapes_dtypes_and_ranges():
    batches = list(SyntheticStrokes("rectangle", 32, batch_size=8, batches_per_epoch=3, seed=1))
    assert len(batches) == 3
    for params, fg, alpha in batches:
        assert params.shape == (8, 9) and fg.shape == alpha.shape == (8, 3, 32, 32)
        assert params.dtype == fg.dtype == alpha.dtype == torch.float32
        for t in (params, fg, alpha):
            assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0
        assert float(alpha.max()) > 0.0


def test_stream_is_the_rasterizers_output():
    params, fg, alpha = next(iter(SyntheticStrokes("watercolor", 32, 4, 1, seed=2)))
    rasterizer = ProceduralRasterizer("watercolor", 32, "black", train=True)  # train mode: no dilate / erode, like the original data
    for i in range(4):
        f, a = rasterizer.render_stroke(params[i].numpy())
        np.testing.assert_array_equal(fg[i].numpy(), f.transpose(2, 0, 1))
        np.testing.assert_array_equal(alpha[i].numpy(), a.transpose(2, 0, 1))


def test_stream_is_deterministic_independent_of_threads_and_fresh_each_pass():
    one = list(SyntheticStrokes("markerpen", 32, 4, 3, seed=3, threads=1))
    three = list(SyntheticStrokes("markerpen", 32, 4, 3, seed=3, threads=3))
    for x, y in zip(one, three):
        for a, b in zip(x, y):
            assert torch.equal(a, b)
    ds = SyntheticStrokes("markerpen", 32, 4, 1, seed=3)
    first, second = next(iter(ds)), next(iter(ds))
    assert not torch.equal(first[0], second[0]), "a new pass must bring new strokes"
    other_seed = next(iter(SyntheticStrokes("markerpen", 32, 4, 1, seed=4)))
    assert not torch.equal(one[0][0], other_seed[0])


def test_resumed_stream_continues_where_the_uninterrupted_one_would_be():
    uninterrupted = SyntheticStrokes("rectangle", 32, 4, 1, seed=7)
    passes = [next(iter(uninterrupted)) for _ in range(4)]
    resumed = next(iter(SyntheticStrokes("rectangle", 32, 4, 1, seed=7, start_pass=3)))  # resuming at epoch 4
    for a, b in zip(passes[3], resumed):
        assert torch.equal(a, b)
    assert not torch.equal(passes[0][0], resumed[0])  # and it is not a replay of the first epoch


def test_validation_set_is_fixed_and_disjoint_from_training():
    a = make_validation_set("rectangle", 32, n=16)
    b = make_validation_set("rectangle", 32, n=16)
    assert all(torch.equal(a[k], b[k]) for k in a) and a["params"].shape == (16, 9) and a["foreground"].shape == (16, 3, 32, 32)
    train_params = next(iter(SyntheticStrokes("rectangle", 32, 16, 1, seed=0)))[0]
    assert not torch.equal(a["params"], train_params)
    assert make_validation_set("rectangle", 32, n=16)["params"].shape == a["params"].shape


# ----- loss and metrics ------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["l2", "l1_ssim"])
def test_loss_is_zero_for_a_perfect_prediction_and_positive_otherwise(kind):
    x = torch.rand(2, 3, 32, 32)
    assert abs(float(reconstruction_loss(kind, x, x, x, x))) < 1e-5
    assert float(reconstruction_loss(kind, x * 0, x * 0, x, x)) > 0.1


def test_unknown_loss_is_rejected():
    x = torch.rand(1, 3, 32, 32)
    with pytest.raises(ValueError, match="loss"):
        reconstruction_loss("l3", x, x, x, x)


class _Replay(torch.nn.Module):
    """Stands in for a renderer: returns prepared outputs batch by batch."""

    def __init__(self, fg: torch.Tensor, alpha: torch.Tensor) -> None:
        super().__init__()
        self.fg, self.alpha, self.i = fg, alpha, 0

    def forward(self, params):
        sl = slice(self.i, self.i + params.shape[0])
        self.i += params.shape[0]
        return self.fg[sl], self.alpha[sl]


def test_evaluate_scores_a_perfect_model_and_a_blank_one():
    val = make_validation_set("rectangle", 32, n=20)
    perfect = evaluate(_Replay(val["foreground"], val["alpha"]), val, torch.device("cpu"))
    assert perfect["psnr_fg"] > 100 and perfect["psnr_alpha"] > 100 and perfect["psnr_mean"] > 100
    assert perfect["ssim_fg"] > 0.999 and perfect["ssim_alpha"] > 0.999
    blank = evaluate(_Replay(torch.zeros_like(val["foreground"]), torch.zeros_like(val["alpha"])), val, torch.device("cpu"))
    assert blank["psnr_alpha"] < 20 and blank["ssim_alpha"] < perfect["ssim_alpha"]
    # PSNR must follow the MSE over the whole set: 10 log10(1 / mse)
    mse = float((val["alpha"] ** 2).mean())
    assert blank["psnr_alpha"] == pytest.approx(10 * np.log10(1 / mse), rel=1e-4)


# ----- training --------------------------------------------------------------------------
def test_initialisation_follows_the_original():
    model = NeuralRenderer("oilpaintbrush", light=True)
    init_like_original(model)
    w = model.shape_decoder.fc1.weight
    assert abs(float(w.std()) - 0.02) < 0.002 and float(model.shape_decoder.fc1.bias.abs().max()) == 0.0
    bn = model.color_decoder.main[1]
    assert abs(float(bn.weight.mean()) - 1.0) < 0.01 and float(bn.bias.abs().max()) == 0.0


def test_training_steps_reduce_the_loss_on_a_fixed_batch():
    torch.manual_seed(0)
    model = NeuralRenderer("rectangle", light=True)
    init_like_original(model)
    params, fg, alpha = next(iter(SyntheticStrokes("rectangle", 32, 32, 1, seed=0)))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(25):
        optimizer.zero_grad()
        p_fg, p_alpha = model(params)
        loss = reconstruction_loss("l2", p_fg, p_alpha, fg, alpha)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    assert losses[-1] < 0.5 * losses[0]


def test_experiment_folders_are_never_reused(tmp_path):
    first = new_experiment_dir(tmp_path, "run")
    second = new_experiment_dir(tmp_path, "run")
    third = new_experiment_dir(tmp_path, "run")
    assert len({first, second, third}) == 3 and second.name.endswith("-2") and third.name.endswith("-3")


def test_tiny_run_writes_everything_and_resumes(tmp_path):
    cfg = TrainConfig(
        brush="rectangle", light=True, epochs=2, samples_per_epoch=64, batch_size=16, threads=2, device="cpu", val_size=32,
        vis_every=1, checkpoint_dir=str(tmp_path / "ckpt"), experiments_dir=str(tmp_path / "exp"),
    )
    result = train(cfg, log=lambda message: None)
    exp = Path(result["experiment_dir"])
    for name in ("config.yaml", "curve.csv", "log.txt", "final_metrics.json", "val_epoch_0001.png", "val_epoch_0002.png"):
        assert (exp / name).is_file(), name
    info = yaml.safe_load((exp / "config.yaml").read_text(encoding="utf-8"))  # plain YAML, with the provenance a log needs
    assert {"commit", "torch", "device", "python", "config", "model"} <= set(info) and info["config"]["brush"] == "rectangle"
    final = json.loads((exp / "final_metrics.json").read_text())
    assert final["epochs_done"] == 2 and final["train_seconds_total"] > 0
    rows = list(csv.DictReader(open(exp / "curve.csv", encoding="utf-8")))
    assert [r["epoch"] for r in rows] == ["1", "2"] and float(rows[0]["train_loss"]) > 0
    for name in ("best.pt", "last.pt"):
        assert (tmp_path / "ckpt" / name).is_file()
    best = load_renderer(tmp_path / "ckpt" / "best.pt")  # the file painting will use
    assert best.spec.name == "rectangle" and best.light and best.source["epoch"] in (1, 2)

    resumed = train(TrainConfig(**{**asdict(cfg), "epochs": 3, "resume": True}), log=lambda message: None)
    assert resumed["experiment_dir"] == result["experiment_dir"]  # continues the same folder instead of starting a new one
    rows = list(csv.DictReader(open(exp / "curve.csv", encoding="utf-8")))
    assert [r["epoch"] for r in rows] == ["1", "2", "3"]

    # resuming a run that is already complete reports it and leaves its results alone
    final_before = (exp / "final_metrics.json").read_text(encoding="utf-8")
    again = train(TrainConfig(**{**asdict(cfg), "epochs": 3, "resume": True}), log=lambda message: None)
    assert again["epochs_done"] == 3 and (exp / "final_metrics.json").read_text(encoding="utf-8") == final_before
    assert len(list(csv.DictReader(open(exp / "curve.csv", encoding="utf-8")))) == 3


def test_invalid_settings_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="loss"):
        train(TrainConfig(brush="rectangle", loss="l3", checkpoint_dir=str(tmp_path), experiments_dir=str(tmp_path)))
    with pytest.raises(ValueError, match="activation"):
        train(TrainConfig(brush="rectangle", activation="relu", checkpoint_dir=str(tmp_path), experiments_dir=str(tmp_path)))
