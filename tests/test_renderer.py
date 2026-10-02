"""Neural renderer: shapes, alpha handling, gradients, checkpoints and exact equivalence with the original 2021 network."""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest
import torch
import torch.nn as nn

from neural_painter.core.procedural_rasterizer import ProceduralRasterizer
from neural_painter.core.stroke_models import get_brush
from neural_painter.models.neural_renderer import (
    CHECKPOINT_FORMAT,
    NeuralRenderer,
    convert_original_state_dict,
    load_original_checkpoint,
    load_renderer,
    original_checkpoint_path,
    save_checkpoint,
)

BRUSHES = ["oilpaintbrush", "watercolor", "markerpen", "rectangle"]


def randomize(model: nn.Module, seed: int = 0) -> None:
    """Give a freshly initialised network non-trivial weights, biases and BatchNorm statistics (any dtype)."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.1, 0.1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight, 0.5, 1.5)
                nn.init.uniform_(m.bias, -0.2, 0.2)
                m.running_mean.normal_(0.0, 0.1)
                m.running_var.uniform_(0.5, 1.5)


def to_original_names(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename our keys back to the original ``huangnet.*`` / ``dcgan.*`` ones."""
    return {k.replace("shape_decoder.", "huangnet.").replace("color_decoder.", "dcgan."): v for k, v in state_dict.items()}


@pytest.fixture
def original(original_repo, monkeypatch):
    """Builder for the original fusion nets, with the original ``networks`` module forced onto the CPU."""
    import networks

    monkeypatch.setattr(networks, "device", torch.device("cpu"))  # the oil branch creates a tensor on this device

    def build(brush: str, light: bool):
        rdrr = original_repo.Renderer(renderer=brush)
        with contextlib.redirect_stdout(io.StringIO()):  # define_G prints a line
            net = networks.define_G(rdrr=rdrr, netG="zou-fusion-net-light" if light else "zou-fusion-net")
        return rdrr, net

    return build


# ----- behaviour -------------------------------------------------------------------------
@pytest.mark.parametrize("light", [True, False])
@pytest.mark.parametrize("brush", BRUSHES)
def test_output_shapes_range_and_input_layouts(brush, light):
    model = NeuralRenderer(brush, light=light).eval()
    dim = get_brush(brush).dim
    size = 32 if light else 128
    params = torch.rand(3, dim)
    with torch.no_grad():
        fg, alpha = model(params)
        fg_4d, alpha_4d = model(params.reshape(3, dim, 1, 1))  # the layout the original code uses
        fg_one, _ = model(params[:1])  # a batch of one must not be squeezed into a vector
    assert model.out_size == size
    assert fg.shape == alpha.shape == (3, 3, size, size) and fg_one.shape == (1, 3, size, size)
    for t in (fg, alpha):
        assert torch.isfinite(t).all() and 0.0 <= float(t.min()) and float(t.max()) <= 1.0  # sigmoid heads
    torch.testing.assert_close(fg_4d, fg)
    torch.testing.assert_close(alpha_4d, alpha)
    torch.testing.assert_close(fg_one, fg[:1])


def test_parameter_counts_are_pinned():
    counts = {light: sum(p.numel() for p in NeuralRenderer("oilpaintbrush", light=light).parameters()) for light in (True, False)}
    # the original has 5,373,004 / 18,093,044: the same nets minus the discarded 3-channel head (6,144 / 3,072 weights)
    assert counts == {True: 5_366_860, False: 18_089_972}


def test_oil_alpha_map_ignores_the_alpha_parameter():
    """Oil strokes are opaque: the alpha map is the mask alone. (The colour decoder still receives the alpha
    parameter as one of its inputs, exactly like the original network, so the foreground can depend on it weakly.)"""
    model = NeuralRenderer("oilpaintbrush", light=True).eval()
    randomize(model)
    p = torch.rand(2, 12)
    flipped = p.clone()
    flipped[:, -1] = 1.0 - p[:, -1]
    assert model.alpha_forced_to_one
    assert torch.equal(model(p)[1], model(flipped)[1])
    q = p.clone().requires_grad_(True)
    _, alpha = model(q)
    alpha.sum().backward()
    shape_dim = get_brush("oilpaintbrush").shape_dim
    assert q.grad[:, -1].abs().max() == 0.0 and q.grad[:, :shape_dim].abs().max() > 0.0  # only the shape parameters move the mask


@pytest.mark.parametrize("brush", ["watercolor", "markerpen", "rectangle"])
def test_alpha_map_scales_with_the_alpha_parameter(brush):
    model = NeuralRenderer(brush, light=True).eval()
    randomize(model)
    dim = get_brush(brush).dim
    full = torch.rand(2, dim)
    full[:, -1] = 1.0
    half = full.clone()
    half[:, -1] = 0.5
    assert not model.alpha_forced_to_one
    # the mask comes from the shape parameters only, so the alpha map is exactly alpha_parameter * mask
    torch.testing.assert_close(model(half)[1], 0.5 * model(full)[1])


def test_bad_arguments_are_rejected():
    with pytest.raises(ValueError, match="expected 12"):
        NeuralRenderer("oilpaintbrush", light=True)(torch.rand(2, 9))
    with pytest.raises(ValueError, match="output_activation"):
        NeuralRenderer("oilpaintbrush", output_activation="relu")


# ----- gradients -------------------------------------------------------------------------
@pytest.mark.parametrize("brush,light", [(b, True) for b in BRUSHES] + [("oilpaintbrush", False)])
def test_autograd_matches_finite_differences(brush, light):
    """The gradient the painter relies on: autograd against central differences, in float64, for every parameter."""
    model = NeuralRenderer(brush, light=light).double().eval()
    randomize(model, seed=5)
    spec = get_brush(brush)
    size = model.out_size
    g = torch.Generator().manual_seed(11)
    p = torch.rand(1, spec.dim, dtype=torch.float64, generator=g)
    w_fg = torch.randn(1, 3, size, size, dtype=torch.float64, generator=g)
    w_alpha = torch.randn(1, 3, size, size, dtype=torch.float64, generator=g)

    def objective(q: torch.Tensor) -> torch.Tensor:
        fg, alpha = model(q)
        return (fg * w_fg).sum() + (alpha * w_alpha).sum()

    q = p.clone().requires_grad_(True)
    objective(q).backward()
    grad = q.grad[0]
    assert float(grad.abs().max()) > 1e-6, "vacuous test: no gradient reaches the stroke parameters"
    eps = 1e-6
    for i in range(spec.dim):
        hi, lo = p.clone(), p.clone()
        hi[0, i] += eps
        lo[0, i] -= eps
        numeric = float(objective(hi) - objective(lo)) / (2 * eps)
        assert abs(numeric - float(grad[i])) <= 1e-5 * (1.0 + abs(numeric)), (
            f"{spec.param_names[i]}: autograd {float(grad[i])!r} vs finite difference {numeric!r}"
        )


# ----- equivalence with the original network ---------------------------------------------
@pytest.mark.parametrize("light", [True, False])
@pytest.mark.parametrize("brush", BRUSHES)
def test_matches_original_network_with_random_weights(original, brush, light):
    rdrr, net = original(brush, light)
    randomize(net, seed=1)
    net.eval()
    ours = NeuralRenderer(brush, light=light, output_activation="none")
    ours.load_state_dict(convert_original_state_dict(net.state_dict()))
    ours.eval()
    x = torch.rand(4, rdrr.d, 1, 1)
    with torch.no_grad():
        fg0, alpha0 = net(x)
        fg1, alpha1 = ours(x)
    assert float(fg0.abs().max()) > 1e-3 and float(alpha0.abs().max()) > 1e-3, "vacuous comparison"
    torch.testing.assert_close(fg1, fg0, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(alpha1, alpha0, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("light", [True, False])
@pytest.mark.parametrize("brush", BRUSHES)
def test_pretrained_checkpoint_matches_original_network(original, brush, light):
    path = original_checkpoint_path(brush, light)
    if not path.is_file():
        pytest.skip(f"{path} has not been downloaded")
    rdrr, net = original(brush, light)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_G_state_dict"])  # the original's own loader
    net.eval()
    ours = load_original_checkpoint(path, brush, light)
    x = torch.rand(4, rdrr.d, 1, 1, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        fg0, alpha0 = net(x)
        fg1, alpha1 = ours(x)
    torch.testing.assert_close(fg1, fg0, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(alpha1, alpha0, rtol=1e-5, atol=1e-6)
    assert ours.source["format"] == "original" and ours.source["val_acc_db"] > 20.0
    assert not ours.training and not any(p.requires_grad for p in ours.parameters())


@pytest.mark.parametrize("light", [True, False])
def test_pretrained_oil_foreground_depends_only_weakly_on_the_alpha_parameter(light):
    """The colour decoder is fed the alpha parameter too, so an oil stroke's foreground is not strictly independent of it."""
    path = original_checkpoint_path("oilpaintbrush", light)
    if not path.is_file():
        pytest.skip(f"{path} has not been downloaded")
    model = load_original_checkpoint(path, "oilpaintbrush", light)
    p = torch.rand(256, 12, generator=torch.Generator().manual_seed(0))
    lo, hi = p.clone(), p.clone()
    lo[:, -1], hi[:, -1] = 0.0, 1.0
    with torch.no_grad():
        fg_lo, alpha_lo = model(lo)
        fg_hi, alpha_hi = model(hi)
    assert torch.equal(alpha_lo, alpha_hi)  # the opacity map never depends on it
    relative = float((fg_lo - fg_hi).abs().mean() / fg_lo.abs().mean())
    assert 0.0 < relative < 0.05  # about 1 % to 1.5 % measured: weak, but not zero


def test_pretrained_oil_renderer_approximates_the_rasterizer():
    """End to end: stroke parameters -> original weights through our adapter -> close to what OpenCV draws."""
    path = original_checkpoint_path("oilpaintbrush", light=True)
    if not path.is_file():
        pytest.skip(f"{path} has not been downloaded")
    model = load_original_checkpoint(path, "oilpaintbrush", light=True)
    rasterizer = ProceduralRasterizer("oilpaintbrush", 32, "black", train=True, rng=np.random.default_rng(0))
    params = rasterizer.sample_uniform(128)
    truth = [rasterizer.render_stroke(p) for p in params]
    gt_fg = np.stack([t[0] for t in truth])
    gt_alpha = np.stack([t[1] for t in truth])
    with torch.no_grad():
        fg, alpha = model(torch.from_numpy(params))

    def psnr(pred: torch.Tensor, gt: np.ndarray) -> float:
        mse = float(np.mean((pred.permute(0, 2, 3, 1).numpy() - gt) ** 2))
        return 10.0 * np.log10(1.0 / mse)

    # the authors report 24.3 dB (mean of the two) on their own validation set
    assert (psnr(fg, gt_fg) + psnr(alpha, gt_alpha)) / 2 > 22.0


# ----- loading and saving ----------------------------------------------------------------
class _NotATensor:
    """Module-level so that it can be pickled."""


def test_loader_refuses_pickled_objects_unless_trusted(tmp_path):
    path = tmp_path / "suspicious.pt"
    torch.save({"model_G_state_dict": {}, "payload": _NotATensor()}, path)
    with pytest.raises(RuntimeError, match="trusted=True"):
        load_original_checkpoint(path, "oilpaintbrush", light=True)
    with pytest.raises(KeyError, match="not a fusion-net checkpoint"):  # unpickled, then rejected for its (empty) content
        load_original_checkpoint(path, "oilpaintbrush", light=True, trusted=True)
    with pytest.raises(FileNotFoundError):
        load_original_checkpoint(tmp_path / "missing.pt", "oilpaintbrush", light=True)


def test_convert_strips_dataparallel_prefix_and_rejects_unknown_keys():
    model = NeuralRenderer("oilpaintbrush", light=True)
    original_style = {"module." + k: v for k, v in to_original_names(model.state_dict()).items()}
    assert set(convert_original_state_dict(original_style)) == set(model.state_dict())
    with pytest.raises(KeyError, match="unexpected key"):
        convert_original_state_dict({"fc.weight": torch.zeros(1)})


def test_load_renderer_opens_original_format_files(tmp_path):
    """Rebuild what the original code saved (old key names, a 6-channel last layer, a NumPy scalar) and load it."""
    model = NeuralRenderer("rectangle", light=True, output_activation="none")
    randomize(model, seed=4)
    state = to_original_names(model.state_dict())
    last = "dcgan.main.9.weight"
    state[last] = torch.cat([state[last], torch.randn_like(state[last])], dim=1)  # the discarded head
    path = tmp_path / "last_ckpt.pt"
    torch.save({"epoch_id": 399, "best_val_acc": np.float64(24.3), "model_G_state_dict": state}, path)

    with pytest.raises(ValueError, match="pass brush= and light="):
        load_renderer(path)
    loaded = load_renderer(path, "tape", True)
    assert loaded.source["format"] == "original" and loaded.source["epoch"] == 399
    assert loaded.source["val_acc_db"] == pytest.approx(24.3)
    x = torch.rand(2, 9)
    model.eval()
    torch.testing.assert_close(loaded(x)[0], model(x)[0])
    torch.testing.assert_close(loaded(x)[1], model(x)[1])


def test_save_and_load_round_trip(tmp_path):
    model = NeuralRenderer("watercolor", light=True)
    randomize(model, seed=3)
    model.eval()
    path = save_checkpoint(tmp_path / "sub" / "renderer.pt", model, epoch=7, metrics={"psnr_mean": 24.5})
    assert path.is_file() and not path.with_name(path.name + ".tmp").exists()  # written atomically
    loaded = load_renderer(path)
    assert loaded.spec.name == "watercolor" and loaded.light and loaded.output_activation == "sigmoid"
    assert loaded.source["format"] == CHECKPOINT_FORMAT and loaded.source["epoch"] == 7 and loaded.source["psnr_mean"] == 24.5
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
    x = torch.rand(3, 15)
    for a, b in zip(loaded(x), model(x)):
        torch.testing.assert_close(a, b)
    with pytest.raises(ValueError, match="requested"):
        load_renderer(path, brush="oil")
    with pytest.raises(ValueError, match="requested"):
        load_renderer(path, light=False)
    assert loaded.describe()["parameters"] == sum(p.numel() for p in model.parameters())


def test_original_checkpoint_paths():
    light = original_checkpoint_path("oil", light=True)
    full = original_checkpoint_path("tape", light=False)
    assert light.parent.name == "checkpoints_G_oilpaintbrush_light" and light.name == "last_ckpt.pt"
    assert full.parent.name == "checkpoints_G_rectangle"
