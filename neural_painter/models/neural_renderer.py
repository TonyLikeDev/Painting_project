"""Neural renderer G_phi (Week 4).

A differentiable surrogate of the procedural rasterizer: ``forward(params)`` maps
``(N, d)`` stroke parameters in ``[0, 1]`` to a foreground and an alpha map of
shape ``(N, 3, S, S)``, so that the gradient of an image loss reaches the stroke
parameters. ``S`` is 128 (full renderer) or 32 (light renderer).

Architecture (the "fusion" net of Zou et al., CVPR 2021)::

    shape parameters --> ShapeDecoder (MLP + PixelShuffle) --> mask    (N, 3, S, S)
    all parameters   --> ColorDecoder (transposed convs)   --> colour  (N, 3, S, S)
    foreground = colour * mask            alpha = alpha_parameter * mask

Two kinds of weights share this class:

* the 2021 pretrained weights, opened with :func:`load_original_checkpoint`
  (``output_activation="none"``: the original has no output non-linearity);
* renderers trained by ``pipeline/train_renderer.py`` (default
  ``output_activation="sigmoid"``, which keeps both heads inside ``[0, 1]``).

:func:`load_renderer` opens either kind of file, so the painter never needs to
know where the weights came from.

Differences from the original ``networks.py`` (outputs are identical, see
``tests/test_renderer.py``): no module-level device global; a batch of one is not
squeezed into a vector; the opaque alpha of the oil brush comes from the brush
config instead of a string test; and the colour decoder has 3 output channels
instead of 6, because the original computed a second 3-channel head and threw it
away.

Layer names (``fc1``, ``conv1``, ``main.0`` ...) are kept from the original so that
converting its state dict is a prefix rename (:func:`convert_original_state_dict`).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.device import peak_memory_gb, reset_peak_memory, synchronize
from ..core.stroke_models import BrushSpec, get_brush

OUTPUT_ACTIVATIONS = ("none", "sigmoid")
CHECKPOINT_FORMAT = "neural_painter.renderer.v1"
ORIGINAL_REPO_DIR = Path(__file__).resolve().parents[2] / "stylized-neural-painting"

_ORIGINAL_PREFIXES = {"huangnet.": "shape_decoder.", "dcgan.": "color_decoder."}


class ShapeDecoder(nn.Module):
    """Mask decoder: the shape parameters become a ``(N, 3, S, S)`` mask (original ``PixelShuffleNet``)."""

    def __init__(self, in_dim: int, light: bool = False) -> None:
        super().__init__()
        self.light = light
        self.out_size = 32 if light else 128
        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        if light:
            self.conv1 = nn.Conv2d(8, 64, 3, 1, 1)
            self.conv2 = nn.Conv2d(64, 4 * 3, 3, 1, 1)
        else:
            self.fc4 = nn.Linear(2048, 4096)
            self.conv1 = nn.Conv2d(16, 32, 3, 1, 1)
            self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
            self.conv3 = nn.Conv2d(8, 16, 3, 1, 1)
            self.conv4 = nn.Conv2d(16, 16, 3, 1, 1)
            self.conv5 = nn.Conv2d(4, 8, 3, 1, 1)
            self.conv6 = nn.Conv2d(8, 4 * 3, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        if self.light:
            x = x.view(-1, 8, 16, 16)
            x = F.relu(self.conv1(x))
            x = self.pixel_shuffle(self.conv2(x))  # 16 -> 32 px
        else:
            x = F.relu(self.fc4(x))
            x = x.view(-1, 16, 16, 16)
            x = F.relu(self.conv1(x))
            x = self.pixel_shuffle(self.conv2(x))  # 16 -> 32 px
            x = F.relu(self.conv3(x))
            x = self.pixel_shuffle(self.conv4(x))  # 32 -> 64 px
            x = F.relu(self.conv5(x))
            x = self.pixel_shuffle(self.conv6(x))  # 64 -> 128 px
        return x


class ColorDecoder(nn.Module):
    """Colour decoder: all parameters become a ``(N, 3, S, S)`` colour image (original ``DCGAN`` / ``DCGAN_32``)."""

    def __init__(self, in_dim: int, light: bool = False, ngf: int = 64, out_channels: int = 3) -> None:
        super().__init__()
        self.in_dim = in_dim
        widths = [ngf * 8, ngf * 4, ngf * 2] if light else [ngf * 8, ngf * 8, ngf * 4, ngf * 2, ngf]
        # 1x1 input -> 4x4, then each stride-2 stage doubles the size (4 -> 8 -> ... -> 32 or 128)
        layers: list[nn.Module] = [nn.ConvTranspose2d(in_dim, widths[0], 4, 1, 0, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU(True)]
        for c_in, c_out in zip(widths[:-1], widths[1:]):
            layers += [nn.ConvTranspose2d(c_in, c_out, 4, 2, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True)]
        layers.append(nn.ConvTranspose2d(widths[-1], out_channels, 4, 2, 1, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x.reshape(x.shape[0], self.in_dim, 1, 1))


class NeuralRenderer(nn.Module):
    """Differentiable stroke renderer for one brush.

    Parameters
    ----------
    brush:             brush name / alias / :class:`BrushSpec` (layout and ``renderer`` section come from the YAML).
    light:             ``True`` for the 32 px renderer (5.4 M parameters), ``False`` for the 128 px one (18.1 M).
    output_activation: ``"sigmoid"`` bounds colour and mask to ``[0, 1]`` (use for new training);
                       ``"none"`` reproduces the 2021 networks, which are unbounded.
    """

    def __init__(self, brush: str | BrushSpec, light: bool = False, output_activation: str = "sigmoid") -> None:
        super().__init__()
        if output_activation not in OUTPUT_ACTIVATIONS:
            raise ValueError(f"output_activation must be one of {OUTPUT_ACTIVATIONS}, got {output_activation!r}")
        self.spec = get_brush(brush)
        self.light = bool(light)
        self.out_size = 32 if self.light else 128
        configured = int(self.spec.renderer["out_size_light" if self.light else "out_size_full"])
        if configured != self.out_size:
            raise ValueError(f"{self.spec.name}: config says out_size {configured}, the network produces {self.out_size}")
        self.output_activation = output_activation
        self.alpha_forced_to_one = bool(self.spec.renderer.get("alpha_forced_to_one", False))
        self.shape_decoder = ShapeDecoder(self.spec.shape_dim, self.light)
        self.color_decoder = ColorDecoder(self.spec.dim, self.light)
        self.source: dict[str, Any] = {"format": "untrained"}

    def forward(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``params`` is ``(N, d)`` (or ``(N, d, 1, 1)``); returns ``(foreground, alpha)``, each ``(N, 3, S, S)``."""
        n = params.shape[0]
        p = params.reshape(n, -1)
        if p.shape[1] != self.spec.dim:
            raise ValueError(f"{self.spec.name}: expected {self.spec.dim} parameters per stroke, got {p.shape[1]}")
        mask = self.shape_decoder(p[:, : self.spec.shape_dim])
        color = self.color_decoder(p)
        if self.output_activation == "sigmoid":
            mask, color = torch.sigmoid(mask), torch.sigmoid(color)
        foreground = color * mask
        alpha = mask if self.alpha_forced_to_one else p[:, -1].reshape(n, 1, 1, 1) * mask
        return foreground, alpha

    def describe(self) -> dict[str, Any]:
        """Facts an experiment log should record about this renderer."""
        return {
            "brush": self.spec.name,
            "light": self.light,
            "out_size": self.out_size,
            "output_activation": self.output_activation,
            "parameters": sum(p.numel() for p in self.parameters()),
            "source": dict(self.source),
        }


def benchmark_renderer(
    model: NeuralRenderer, device: torch.device, batch_size: int = 64, iters: int = 20, warmup: int = 3
) -> dict[str, float | None]:
    """Time the painter's inner loop on ``device``: a forward pass, and a forward + backward pass to the stroke parameters.

    The weights are frozen during the measurement (the painter optimizes the parameters, not the network), the clock
    is read after the device has finished, and the peak memory is whatever the backend can report (``None`` on CPU).
    """
    flags = [p.requires_grad for p in model.parameters()]
    was_training = model.training
    model.to(device).eval().requires_grad_(False)
    params = torch.rand(batch_size, model.spec.dim, device=device)

    def step(backward: bool) -> None:
        if backward:
            q = params.clone().requires_grad_(True)
            fg, alpha = model(q)
            (fg.sum() + alpha.sum()).backward()
        else:
            with torch.no_grad():
                model(params)

    timings: dict[str, float | None] = {}
    try:
        reset_peak_memory(device)
        for name, backward in (("forward_ms", False), ("forward_backward_ms", True)):
            for _ in range(warmup):
                step(backward)
            synchronize(device)
            t0 = time.perf_counter()
            for _ in range(iters):
                step(backward)
            synchronize(device)
            timings[name] = (time.perf_counter() - t0) / iters * 1e3
        timings["peak_memory_gb"] = peak_memory_gb(device)
    finally:
        for p, flag in zip(model.parameters(), flags):
            p.requires_grad_(flag)
        model.train(was_training)
    return timings


# ----- checkpoints -------------------------------------------------------------------------
def original_checkpoint_path(brush: str | BrushSpec, light: bool = False, root: str | Path | None = None) -> Path:
    """Where the authors' ``last_ckpt.pt`` for ``brush`` lives inside the nested original checkout."""
    folder = get_brush(brush).renderer["checkpoint_light" if light else "checkpoint_full"]
    return Path(root if root is not None else ORIGINAL_REPO_DIR) / folder / "last_ckpt.pt"


def _allowed_numpy_globals() -> list[Any]:
    """NumPy types inside the 2021 checkpoints (``best_val_acc`` was saved as a NumPy scalar under numpy 1.x)."""
    core = getattr(np, "_core", None) or np.core  # numpy 2 renamed numpy.core to numpy._core
    scalar = core.multiarray.scalar
    allowed: list[Any] = [(scalar, "numpy.core.multiarray.scalar"), (scalar, "numpy._core.multiarray.scalar"), np.dtype]
    float64 = getattr(getattr(np, "dtypes", None), "Float64DType", None)
    if float64 is not None:
        allowed.append(float64)
    return allowed


def _torch_load(path: str | Path, *, trusted: bool = False, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """``torch.load`` that refuses arbitrary pickled code unless ``trusted=True``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if trusted:
        return torch.load(path, map_location=map_location, weights_only=False)
    try:
        with torch.serialization.safe_globals(_allowed_numpy_globals()):
            return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:  # unpickling refused, or a torch too old for safe_globals
        raise RuntimeError(
            f"{path.name} could not be loaded with weights_only=True ({type(exc).__name__}). If the file comes from a "
            "source you trust, such as the authors' release of the 2021 weights, pass trusted=True to allow full unpickling."
        ) from exc


def convert_original_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename the keys of an original ``ZouFCNFusion`` / ``ZouFCNFusionLight`` state dict for :class:`NeuralRenderer`.

    ``huangnet.*`` becomes ``shape_decoder.*`` and ``dcgan.*`` becomes ``color_decoder.*``. The original colour
    decoder ends in 6 channels of which only the first three are ever used, so the last weight is cut to 3.
    """
    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        name = key[len("module.") :] if key.startswith("module.") else key  # DataParallel prefix
        for old, new in _ORIGINAL_PREFIXES.items():
            if name.startswith(old):
                converted[new + name[len(old) :]] = value
                break
        else:
            raise KeyError(f"unexpected key {key!r} in an original fusion-net state dict")
    conv_weights = [k for k, v in converted.items() if k.startswith("color_decoder.main.") and k.endswith(".weight") and v.dim() == 4]
    if not conv_weights:
        raise KeyError("the state dict has no colour-decoder convolution weights, so it is not a fusion-net checkpoint")
    last = max(conv_weights, key=lambda k: int(k.split(".")[2]))
    if converted[last].shape[1] == 6:
        converted[last] = converted[last][:, :3].contiguous()
    return converted


def _finish(model: NeuralRenderer, freeze: bool) -> NeuralRenderer:
    """Inference state: eval mode (BatchNorm uses its running statistics) and, by default, frozen weights."""
    model.eval()
    model.requires_grad_(not freeze)
    return model


def _from_original(ckpt: Mapping[str, Any], brush: str | BrushSpec, light: bool, path: Any, freeze: bool) -> NeuralRenderer:
    if "model_G_state_dict" not in ckpt:
        raise KeyError(f"{path}: not an original checkpoint (no 'model_G_state_dict'; found {sorted(ckpt)})")
    model = NeuralRenderer(brush, light=light, output_activation="none")
    model.load_state_dict(convert_original_state_dict(ckpt["model_G_state_dict"]), strict=True)
    model.source = {
        "format": "original",
        "path": str(path),
        "epoch": int(ckpt.get("epoch_id", -1)),
        "val_acc_db": float(ckpt.get("best_val_acc", float("nan"))),
    }
    return _finish(model, freeze)


def load_original_checkpoint(
    path: str | Path,
    brush: str | BrushSpec,
    light: bool = False,
    *,
    freeze: bool = True,
    trusted: bool = False,
) -> NeuralRenderer:
    """Build a :class:`NeuralRenderer` from an original ``last_ckpt.pt`` (the authors' 2021 weights).

    The result is in eval mode with frozen weights (the painter optimizes the stroke parameters, not the network);
    pass ``freeze=False`` to fine-tune it. ``trusted=True`` allows full unpickling, see :func:`_torch_load`.
    """
    return _from_original(_torch_load(path, trusted=trusted), brush, light, path, freeze)


def save_checkpoint(path: str | Path, model: NeuralRenderer, **extra: Any) -> Path:
    """Write ``model`` (plus ``extra`` metadata or training state) atomically, so an interrupted run cannot corrupt it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "brush": model.spec.name,
        "light": model.light,
        "output_activation": model.output_activation,
        "model": model.state_dict(),
        **extra,
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_renderer(
    path: str | Path,
    brush: str | BrushSpec | None = None,
    light: bool | None = None,
    *,
    freeze: bool = True,
    trusted: bool = False,
) -> NeuralRenderer:
    """Open a renderer checkpoint of either kind: the authors' ``last_ckpt.pt`` or one written by ``save_checkpoint``.

    ``brush`` and ``light`` are required for an original file (it does not say what it contains) and optional,
    but checked when given, for our own format.
    """
    ckpt = _torch_load(path, trusted=trusted)
    if "model_G_state_dict" in ckpt:
        if brush is None or light is None:
            raise ValueError("an original checkpoint does not record its brush or size: pass brush= and light=")
        return _from_original(ckpt, brush, light, path, freeze)
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: unknown checkpoint format {ckpt.get('format')!r}")
    if brush is not None and get_brush(brush).name != ckpt["brush"]:
        raise ValueError(f"{path}: holds a {ckpt['brush']!r} renderer, but brush={brush!r} was requested")
    if light is not None and bool(light) != ckpt["light"]:
        raise ValueError(f"{path}: holds a {'light' if ckpt['light'] else 'full'} renderer, but light={light} was requested")
    model = NeuralRenderer(ckpt["brush"], light=ckpt["light"], output_activation=ckpt["output_activation"])
    model.load_state_dict(ckpt["model"], strict=True)
    model.source = {"format": CHECKPOINT_FORMAT, "path": str(path), "epoch": int(ckpt.get("epoch", -1)), **dict(ckpt.get("metrics", {}))}
    return _finish(model, freeze)
