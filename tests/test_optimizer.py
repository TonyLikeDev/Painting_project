"""Painter engine: the optimization loop on a toy renderer (always), and end to end with the pretrained oil renderer (when present)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from neural_painter.core.image_io import load_image
from neural_painter.core.stroke_models import get_brush
from neural_painter.models.neural_renderer import load_original_checkpoint, original_checkpoint_path
from neural_painter.pipeline import metrics
from neural_painter.pipeline.painter_engine import (
    PainterConfig,
    PainterEngine,
    paint_fixed_grid,
    save_strokes_npz,
)

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


class ToyRenderer(nn.Module):
    """A tiny differentiable stand-in for the neural renderer: a soft-edged axis-aligned rectangle (xc, yc, w, h) in colour (r, g, b)."""

    def __init__(self, size: int = 16) -> None:
        super().__init__()
        self.spec = get_brush("rectangle")
        self.out_size = size
        ys, xs = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij")
        self.register_buffer("xs", xs)
        self.register_buffer("ys", ys)

    def forward(self, p: torch.Tensor):
        xc, yc, w, h = (p[:, i, None, None] for i in range(4))
        mask = torch.sigmoid((0.25 * w - (self.xs - xc).abs()) * 40) * torch.sigmoid((0.25 * h - (self.ys - yc).abs()) * 40)
        fg = p[:, 5:8, None, None] * mask[:, None]
        alpha = p[:, 8, None, None, None] * mask[:, None].expand(-1, 3, -1, -1)
        return fg, alpha


def two_colour_blocks(blocks: int = 2, size: int = 16) -> torch.Tensor:
    img = torch.zeros(blocks, 3, size, size)
    img[:, 0, :, : size // 2] = 0.9  # left half red
    img[:, 2, :, size // 2 :] = 0.9  # right half blue
    return img


def engine(seed: int = 0, **overrides) -> PainterEngine:
    cfg = PainterConfig(strokes_per_block=5, iters_per_stroke=10, seed=seed, **overrides)
    return PainterEngine(ToyRenderer(), cfg, device="cpu")


# ----- the loop on a toy renderer ----------------------------------------------------------
def test_loss_decreases_over_50_steps_on_a_toy_target():
    """At the original's learning rate (0.002) a parameter moves at most about 0.1 in 50 steps, so the drop is small
    (6 to 8 % over four seeds); at 0.02 it is 43 to 50 % and +3.3 to +4 dB."""
    slow = engine().paint(two_colour_blocks(), canvas_color="black")
    assert slow.loss.shape == (50,) and slow.psnr.shape == (50,)
    assert slow.loss[-1] < 0.97 * slow.loss[0] and slow.psnr[-1] > slow.psnr[0] + 0.2
    fast = engine(lr=0.02).paint(two_colour_blocks(), canvas_color="black")
    assert fast.loss[-1] < 0.7 * fast.loss[0] and fast.psnr[-1] > fast.psnr[0] + 2.0


def test_parameters_stay_inside_their_ranges_and_have_the_right_shape():
    result = engine().paint(two_colour_blocks(3), canvas_color="black")
    spec = get_brush("rectangle")
    assert result.params.shape == (3, 5, spec.dim) and result.canvas.shape == (3, 3, 16, 16)
    shape, color, alpha = spec.split(result.params)
    assert shape.min() >= 0.1 - 1e-6 and shape.max() <= 0.9 + 1e-6  # kept off the block border
    assert color.min() >= 0.0 and color.max() <= 1.0 and alpha.min() >= 0.0 and alpha.max() <= 1.0


def test_same_seed_same_strokes_and_a_different_seed_differs():
    a = engine(1).paint(two_colour_blocks(), canvas_color="black")
    b = engine(1).paint(two_colour_blocks(), canvas_color="black")
    c = engine(2).paint(two_colour_blocks(), canvas_color="black")
    np.testing.assert_array_equal(a.params, b.params)
    np.testing.assert_array_equal(a.loss, b.loss)
    assert not np.array_equal(a.params, c.params)


def test_callback_history_and_steps_to_psnr():
    seen = []
    result = engine().paint(two_colour_blocks(), canvas_color="black", callback=lambda a, s, loss, p: seen.append((a, s, loss, p)))
    assert [t[0] for t in seen] == [0, 1, 2, 3, 4] and [t[1] for t in seen] == [10, 20, 30, 40, 50]
    assert all(np.isfinite(t[2]) and np.isfinite(t[3]) for t in seen)
    np.testing.assert_array_equal(result.anchor, np.repeat(np.arange(5), 10))
    reached = result.steps_to_psnr(float(result.psnr.max()))
    assert reached is not None and 1 <= reached <= 50 and result.steps_to_psnr(1e9) is None
    assert result.seconds > 0.0


def test_set_deterministic_toggles_the_cudnn_flags(monkeypatch):
    from neural_painter.core.device import set_deterministic

    monkeypatch.setattr(torch.backends.cudnn, "deterministic", torch.backends.cudnn.deterministic)  # restored after the test
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", torch.backends.cudnn.benchmark)
    set_deterministic(True)
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    set_deterministic(False)
    assert not torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark


def test_a_cuda_painting_repeats_exactly_when_cudnn_is_deterministic(monkeypatch, pretrained):
    """Without the flag two runs of one seed on cuda end in different strokes (cuDNN picks algorithms whose summation order varies)."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from neural_painter.core.device import set_deterministic

    monkeypatch.setattr(torch.backends.cudnn, "deterministic", torch.backends.cudnn.deterministic)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", torch.backends.cudnn.benchmark)
    set_deterministic(True)
    image = load_image(ROOT / "data" / "eval_set" / "apple.png")
    cfg = PainterConfig(strokes_per_block=3, seed=1)
    a, b = (paint_fixed_grid(image, pretrained, cfg, grid=2, device="cuda") for _ in range(2))
    np.testing.assert_array_equal(a.strokes, b.strokes)
    np.testing.assert_array_equal(a.result.loss, b.result.loss)


def test_the_resize_option_changes_the_target_the_strokes_are_fitted_to():
    image = np.random.default_rng(0).random((64, 64, 3), dtype=np.float32)  # white noise: averaging and bilinear sampling disagree strongly
    cfg = dict(strokes_per_block=2, iters_per_stroke=3, seed=0)
    area, linear = (paint_fixed_grid(image, ToyRenderer(), PainterConfig(resize=r, **cfg), grid=2, canvas_size=64, device="cpu") for r in ("area", "linear"))
    assert area.strokes.shape == linear.strokes.shape and not np.array_equal(area.strokes, linear.strokes)
    assert PainterConfig().resize == "area"  # the default is unchanged
    with pytest.raises(ValueError, match="resize"):
        PainterEngine(ToyRenderer(), PainterConfig(resize="nearest"), device="cpu")


def test_a_probe_sees_every_step_and_changes_nothing():
    seen = []

    def probe(anchor, step, canvas, targets, params):
        (g,) = torch.autograd.grad((canvas - targets).abs().mean(), params[0], retain_graph=True)  # a gradient taken in the probe
        seen.append((anchor, step, canvas.requires_grad, float(g.abs().sum())))

    plain = engine(4).paint(two_colour_blocks(), canvas_color="black")
    probed = engine(4).paint(two_colour_blocks(), canvas_color="black", probe=probe)
    assert [s[:2] for s in seen] == [(i // 10, i) for i in range(50)] and all(s[2] for s in seen)
    assert any(s[3] > 0 for s in seen)  # the gradient reached the shape parameters
    np.testing.assert_array_equal(plain.params, probed.params)  # probing did not disturb the optimization
    np.testing.assert_array_equal(plain.loss, probed.loss)


def test_the_sinkhorn_term_is_used_only_when_asked_for():
    plain = engine().paint(two_colour_blocks(), canvas_color="black")
    with_ot = engine(beta_ot=0.1, ot_size=8).paint(two_colour_blocks(), canvas_color="black")
    assert engine().sinkhorn is None and engine(beta_ot=0.1).sinkhorn is not None
    assert np.isfinite(with_ot.loss).all() and not np.allclose(plain.loss, with_ot.loss)  # the OT term adds to the objective
    assert (with_ot.loss > 0).all()


def test_only_the_strokes_placed_so_far_are_drawn():
    e = engine()
    base = torch.zeros(1, 3, 16, 16)
    x_shape, x_color, x_alpha = (torch.rand(1, 5, d) for d in (5, 3, 1))
    first = e.render(base, x_shape, x_color, x_alpha, anchor=0)
    x_shape2 = x_shape.clone()
    x_shape2[:, 1:] = torch.rand(1, 4, 5)  # change strokes that are not placed yet
    torch.testing.assert_close(e.render(base, x_shape2, x_color, x_alpha, anchor=0), first)
    assert not torch.allclose(e.render(base, x_shape, x_color, x_alpha, anchor=4), first)


def test_a_given_starting_canvas_is_the_base_of_the_composite():
    start = torch.full((2, 3, 16, 16), 0.5)
    result = engine().paint(two_colour_blocks(), canvas=start)
    assert result.canvas.shape == start.shape and float(result.canvas.mean()) != 0.5  # strokes were drawn onto it


def test_validation_and_defaults():
    with pytest.raises(ValueError, match="optimizer"):
        PainterEngine(ToyRenderer(), PainterConfig(optimizer="sgd"), device="cpu")
    with pytest.raises(ValueError, match="renderer draws"):
        engine().paint(torch.zeros(1, 3, 32, 32))
    assert PainterConfig(strokes_per_block=20).resolved_iters() == 25  # int(500 / 20), the original
    assert PainterConfig(strokes_per_block=9).resolved_iters() == 55 and PainterConfig(strokes_per_block=500).resolved_iters() == 1
    assert PainterConfig(strokes_per_block=900).resolved_iters() == 1 and PainterConfig(iters_per_stroke=7).resolved_iters() == 7
    adam = engine(optimizer="adam").paint(two_colour_blocks(), canvas_color="black")
    assert np.isfinite(adam.loss).all()
    with pytest.raises(ValueError, match="init"):
        PainterEngine(ToyRenderer(), PainterConfig(init="random"), device="cpu")


def test_uniform_initialization_is_a_different_but_valid_start():
    guided = engine(3).paint(two_colour_blocks(), canvas_color="black")
    flat = engine(3, init="uniform").paint(two_colour_blocks(), canvas_color="black")
    assert np.isfinite(flat.loss).all() and flat.params.shape == guided.params.shape
    assert flat.params.min() >= 0.0 and flat.params.max() <= 1.0
    assert not np.array_equal(flat.params, guided.params)  # the strokes started somewhere else


def test_strokes_are_saved_in_the_layout_of_the_original_npz(tmp_path):
    spec = get_brush("rectangle")
    strokes = np.random.default_rng(0).random((7, spec.dim), dtype=np.float32)
    path = save_strokes_npz(tmp_path / "sub" / "strokes.npz", strokes, spec, seed=3)
    data = np.load(path)
    assert data["x_ctt"].shape == (1, 7, 5) and data["x_color"].shape == (1, 7, 3) and data["x_alpha"].shape == (1, 7, 1)
    np.testing.assert_array_equal(np.concatenate([data["x_ctt"], data["x_color"], data["x_alpha"]], -1)[0], strokes)
    assert str(data["brush"]) == "rectangle" and int(data["seed"]) == 3


# ----- metrics -----------------------------------------------------------------------------
def test_image_metrics():
    a = np.random.default_rng(0).random((32, 32, 3), dtype=np.float32)
    assert metrics.psnr(a, a) == float("inf") and metrics.ssim(a, a) > 0.999
    b = np.clip(a + 0.1, 0, 1)
    assert metrics.psnr(a, b) == pytest.approx(20.0, abs=0.5) and metrics.ssim(a, b) < metrics.ssim(a, a)
    with pytest.raises(ValueError, match="same size"):
        metrics.psnr(a, a[:16])
    assert metrics.image_metrics(a, b)["lpips"] is None
    if not metrics.LPIPSMetric.available():
        with pytest.raises(RuntimeError, match="download"):  # never downloads on its own
            metrics.LPIPSMetric()


# ----- end to end with the pretrained renderer ---------------------------------------------
@pytest.fixture(scope="module")
def pretrained():
    path = original_checkpoint_path("oilpaintbrush", light=True)
    if not path.is_file():
        pytest.skip(f"{path} has not been downloaded")
    return load_original_checkpoint(path, "oilpaintbrush", light=True)


def test_a_small_painting_with_the_pretrained_oil_renderer(pretrained):
    image = load_image(ROOT / "data" / "eval_set" / "apple.png")
    cfg = PainterConfig(strokes_per_block=10, seed=0)  # 2 x 2 grid: 40 strokes
    paint = paint_fixed_grid(image, pretrained, cfg, grid=2, device="cpu")
    assert paint.image.shape == (512, 512, 3) and 0.0 <= paint.image.min() and paint.image.max() <= 1.0
    assert paint.strokes.shape == (40, 12) and paint.strokes.min() >= 0.0 and paint.strokes.max() <= 1.0
    assert paint.result.psnr[-1] > paint.result.psnr[0] + 3.0  # the optimization is working on a real image
    blank = np.ones_like(image)  # an empty white canvas is the baseline the strokes must beat
    assert metrics.psnr(paint.image, image) > metrics.psnr(blank, image)


def test_full_image_mode_is_a_one_block_grid(pretrained):
    image = load_image(ROOT / "data" / "eval_set" / "iceland.png")
    paint = paint_fixed_grid(image, pretrained, PainterConfig(strokes_per_block=8, seed=1), grid=1, device="cpu")
    assert paint.strokes.shape == (8, 12) and paint.result.params.shape == (1, 8, 12)
    np.testing.assert_allclose(np.sort(paint.strokes, axis=0), np.sort(paint.result.params[0], axis=0), atol=1e-6)  # block == global coordinates
