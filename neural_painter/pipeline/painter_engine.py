"""Stroke optimization against a target image (Week 5).

:class:`PainterEngine` is the loop of the original ``Painter`` (``demo.py``) / ``ProgressivePainter`` level
(``demo_prog.py``) written as a library: for a batch of ``B`` target blocks it adds the strokes of every block one at a time,
and after each new stroke runs ``iters_per_stroke`` optimizer steps on all the strokes placed so far.

One step, in the original's order::

    zero_grad -> clamp parameters -> render strokes 0..k with the neural renderer -> dilate foreground / erode
    alpha (3x3) -> composite in order onto the starting canvas -> loss = beta_l1 * L1 + beta_ot * Sinkhorn
    -> backward -> clamp parameters -> optimizer step

Shape parameters are clamped to ``[0.1, 0.9]`` so a stroke never collapses onto the block border, colour and
alpha to ``[0, 1]``. A new stroke is drawn from the error map of the canvas left by the previous anchor
(:class:`neural_painter.pipeline.stroke_sampler.StrokeSampler`).

:func:`paint_fixed_grid` runs the engine on an ``m x m`` grid of blocks (``m = 1`` is "full image mode"), maps the strokes
back to whole-image coordinates and renders them with the procedural rasterizer at the output size, which is what
the final image of the original is.

Differences from the original, none of which changes what is optimized: strokes, tensors and the optimizer are
owned by one object instead of a class with module globals; the three random generators (parameter initialization, stroke
sampler, block shuffle) come from one seed; metrics are read back once per anchor, not printed every step; the progress callback
replaces ``print``.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from ..core.device import get_device, synchronize
from ..core.differentiable_canvas import blank_canvas, composite_sequence
from ..core.grid import block_to_global, flatten_blocks, img2patches
from ..core.morphology import dilation2d, erosion2d
from ..core.procedural_rasterizer import ProceduralRasterizer
from ..core.stroke_models import BrushSpec
from ..losses.pixel_loss import PixelLoss, psnr
from ..losses.sinkhorn import SinkhornLoss
from ..models.neural_renderer import NeuralRenderer
from .stroke_sampler import StrokeSampler

OPTIMIZERS = ("rmsprop", "adam")
INITS = ("error_map", "uniform")
RESIZES = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR}  # how the image is shrunk to the blocks' resolution


@dataclass
class PainterConfig:
    """Settings of one painting run. Defaults are the original's (``demo.py`` with the pixel loss only)."""

    strokes_per_block: int = 20  # 500 strokes on a 5x5 grid
    iters_per_stroke: int = 0  # 0 means int(500 / strokes_per_block), the original
    lr: float = 0.002
    optimizer: str = "rmsprop"  # "rmsprop" (the original) or "adam"
    centered: bool = False  # RMSprop(centered=True) is what the progressive demo uses
    init: str = "error_map"  # where new strokes start: the error map (the original) or uniformly over the block (ablation only)
    resize: str = "area"  # how the image is shrunk to the blocks: "area" averages, "linear" is OpenCV's default as in the original
    beta_l1: float = 1.0
    beta_ot: float = 0.0  # 0 = pixel loss only (the original's default); the demos use 0.1 with --with_ot_loss
    ot_epsilon: float = 0.01
    ot_iters: int = 5
    ot_size: int = 24
    ot_channel: str = "random"
    ot_lse: str = "reference"
    shape_range: tuple[float, float] = (0.1, 0.9)
    morphology: bool = True  # 3x3 dilate foreground / erode alpha, mimicking the rasterizer's inference step
    seed: int = 0

    def resolved_iters(self) -> int:
        return self.iters_per_stroke or max(1, int(500 / self.strokes_per_block))


@dataclass
class PaintResult:
    """Output of :meth:`PainterEngine.paint` for ``B`` blocks of ``N`` strokes."""

    params: np.ndarray  # (B, N, d), block coordinates, all in [0, 1]
    canvas: torch.Tensor  # (B, 3, S, S) final neural canvas, detached
    loss: np.ndarray  # one value per optimizer step
    psnr: np.ndarray  # PSNR (dB) of the neural canvas against the targets, per step
    anchor: np.ndarray  # index of the newest stroke at each step
    seconds: float
    config: PainterConfig = field(repr=False)

    def steps_to_psnr(self, threshold: float) -> int | None:
        """First optimizer step at which the PSNR reaches ``threshold`` dB (``None`` if it never does)."""
        hit = np.nonzero(self.psnr >= threshold)[0]
        return int(hit[0]) + 1 if hit.size else None


class PainterEngine:
    """Optimizes stroke parameters for a batch of target blocks through a frozen neural renderer."""

    def __init__(self, renderer: NeuralRenderer, config: PainterConfig | None = None, device: str | torch.device | None = None) -> None:
        self.config = config or PainterConfig()
        if self.config.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {self.config.optimizer!r}")
        if self.config.init not in INITS:
            raise ValueError(f"init must be one of {INITS}, got {self.config.init!r}")
        if self.config.resize not in RESIZES:
            raise ValueError(f"resize must be one of {tuple(RESIZES)}, got {self.config.resize!r}")
        self.device = get_device(device)
        self.renderer = renderer.to(self.device).eval().requires_grad_(False)  # the strokes are optimized, not the network
        self.spec: BrushSpec = renderer.spec
        self.rng = np.random.default_rng(self.config.seed)
        # the rasterizer is used only for its brush-specific initial-stroke ranges (``initial_params_at``)
        self.sampler = StrokeSampler(ProceduralRasterizer(self.spec, renderer.out_size, "black", rng=self.rng))
        self.pixel_loss = PixelLoss("l1")
        c = self.config
        self.sinkhorn = (
            SinkhornLoss(c.ot_epsilon, c.ot_iters, c.ot_size, channel=c.ot_channel, lse=c.ot_lse, seed=c.seed) if c.beta_ot > 0 else None
        )

    # ----- one step -----------------------------------------------------------------------
    def _clamp(self, x_shape: torch.Tensor, x_color: torch.Tensor, x_alpha: torch.Tensor) -> None:
        lo, hi = self.config.shape_range
        with torch.no_grad():
            x_shape.clamp_(lo, hi)
            x_color.clamp_(0.0, 1.0)
            x_alpha.clamp_(0.0, 1.0)

    def render(self, base: torch.Tensor, x_shape: torch.Tensor, x_color: torch.Tensor, x_alpha: torch.Tensor, anchor: int) -> torch.Tensor:
        """Canvas after compositing strokes ``0..anchor`` of every block onto ``base``; differentiable."""
        b, _, s, _ = base.shape
        x = torch.cat([x_shape, x_color, x_alpha], dim=-1)[:, : anchor + 1]
        fg, alpha = self.renderer(x.reshape(b * (anchor + 1), -1))
        if self.config.morphology:
            fg, alpha = dilation2d(fg, 1), erosion2d(alpha, 1)
        fg = fg.reshape(b, anchor + 1, 3, s, s)
        alpha = alpha.reshape(b, anchor + 1, 3, s, s)
        return composite_sequence(base, fg, alpha)

    # ----- the loop -----------------------------------------------------------------------
    def paint(
        self,
        targets: torch.Tensor,
        canvas: torch.Tensor | None = None,
        canvas_color: str | None = None,
        callback: Callable[[int, int, float, float], None] | None = None,
        probe: Callable[[int, int, torch.Tensor, torch.Tensor, list[torch.Tensor]], None] | None = None,
    ) -> PaintResult:
        """Place ``strokes_per_block`` strokes on each of the ``B`` target blocks ``(B, 3, S, S)``.

        ``canvas`` is the starting canvas (default: a blank one in ``canvas_color``, the brush's default when ``None``).
        ``callback(anchor, step, loss, psnr)`` is called after every anchor, once the strokes of that anchor are optimized.
        ``probe(anchor, step, canvas, targets, params)`` is called at every step, after the canvas is rendered and before the
        loss is back-propagated; ``canvas`` is differentiable and ``params`` are the leaf tensors, so a probe can take
        gradients of any loss with ``torch.autograd.grad(..., retain_graph=True)`` (it must not change them). It is a
        diagnostic hook: it costs nothing when it is ``None``.
        """
        cfg, spec, dev = self.config, self.spec, self.device
        targets = targets.to(dev)
        b, _, s, _ = targets.shape
        if s != self.renderer.out_size:
            raise ValueError(f"targets are {s} px but the renderer draws {self.renderer.out_size} px blocks")
        base = canvas.to(dev) if canvas is not None else blank_canvas(b, s, canvas_color or spec.default_canvas_color, dev)
        n, iters = cfg.strokes_per_block, cfg.resolved_iters()

        def leaf(dim: int) -> torch.Tensor:  # uniform start, as the original; real strokes overwrite it anchor by anchor
            return torch.tensor(self.rng.random((b, n, dim), dtype=np.float32), device=dev, requires_grad=True)

        x_shape, x_color, x_alpha = leaf(spec.shape_dim), leaf(spec.color_dim), leaf(spec.alpha_dim)
        params = [x_shape, x_color, x_alpha]
        opt = (
            torch.optim.RMSprop(params, lr=cfg.lr, centered=cfg.centered)
            if cfg.optimizer == "rmsprop" else torch.optim.Adam(params, lr=cfg.lr)
        )
        current = base  # the canvas the sampler looks at: what the previous anchor left behind
        losses: list[torch.Tensor] = []
        psnrs: list[torch.Tensor] = []
        started = time.perf_counter()
        for anchor in range(n):
            new = torch.from_numpy(self.sampler.sample(targets, current, uniform=cfg.init == "uniform")).to(dev)
            with torch.no_grad():
                x_shape[:, anchor] = new[:, : spec.shape_dim]
                x_color[:, anchor] = new[:, spec.shape_dim : spec.shape_dim + spec.color_dim]
                x_alpha[:, anchor] = new[:, spec.shape_dim + spec.color_dim :]
            for _ in range(iters):
                opt.zero_grad()
                self._clamp(x_shape, x_color, x_alpha)
                out = self.render(base, x_shape, x_color, x_alpha, anchor)
                if probe is not None:
                    probe(anchor, len(losses), out, targets, params)
                loss = cfg.beta_l1 * self.pixel_loss(out, targets)
                if self.sinkhorn is not None:
                    loss = loss + cfg.beta_ot * self.sinkhorn(out, targets)
                loss.backward()
                self._clamp(x_shape, x_color, x_alpha)
                opt.step()
                losses.append(loss.detach())
                psnrs.append(psnr(out.detach(), targets))
            current = out.detach()
            if callback is not None:
                callback(anchor, len(losses), float(losses[-1]), float(psnrs[-1]))
        synchronize(dev)
        elapsed = time.perf_counter() - started
        with torch.no_grad():
            self._clamp(x_shape, x_color, x_alpha)
            final = torch.cat([x_shape, x_color, x_alpha], dim=-1).cpu().numpy()
        return PaintResult(
            params=final, canvas=current, loss=torch.stack(losses).cpu().numpy(), psnr=torch.stack(psnrs).cpu().numpy(),
            anchor=np.repeat(np.arange(n), iters), seconds=elapsed, config=cfg,
        )


# ----- whole images on a fixed grid ---------------------------------------------------------
@dataclass
class GridPaint:
    image: np.ndarray  # (canvas_size, canvas_size, 3) float32, the rasterized painting
    strokes: np.ndarray  # (M * M * N, d) in whole-image coordinates, in drawing order
    result: PaintResult
    render_seconds: float


def paint_fixed_grid(
    image: np.ndarray,
    renderer: NeuralRenderer,
    config: PainterConfig | None = None,
    grid: int = 5,
    canvas_size: int = 512,
    canvas_color: str | None = None,
    device: str | torch.device | None = None,
    engine: PainterEngine | None = None,
) -> GridPaint:
    """Paint ``image`` ((H, W, 3) float RGB in [0, 1]) with ``grid x grid`` blocks (``grid = 1`` is full image mode).

    The image is stretched to ``grid * S`` pixels (``S`` = the renderer's block size) and cut into blocks, every block is
    painted by the engine, and the strokes are rendered at ``canvas_size`` by the procedural rasterizer in the order
    "stroke ``j`` of every block (blocks shuffled with the seeded generator) before stroke ``j + 1``", like the original.
    """
    engine = engine or PainterEngine(renderer, config, device)
    spec = engine.spec
    color = canvas_color or spec.default_canvas_color
    targets = img2patches(image, grid, renderer.out_size, engine.device, interpolation=RESIZES[engine.config.resize])
    result = engine.paint(targets, canvas_color=color)
    order = engine.rng.permutation(grid * grid)
    strokes = flatten_blocks(block_to_global(result.params, spec, grid), order)
    t0 = time.perf_counter()
    final = ProceduralRasterizer(spec, canvas_size, color).render_sequence(strokes)
    return GridPaint(image=final, strokes=strokes, result=result, render_seconds=time.perf_counter() - t0)


def save_strokes_npz(path: str | Path, strokes: np.ndarray, spec: BrushSpec, **extra) -> Path:
    """Write strokes in the layout of the original ``<name>_strokes.npz`` (``x_ctt``, ``x_color``, ``x_alpha`` of shape
    ``(1, N, .)``) plus the brush name and any ``extra`` metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    v = np.asarray(strokes, dtype=np.float32).reshape(1, -1, spec.dim)
    shape, color, alpha = spec.split(v)
    np.savez(path, x_ctt=shape, x_color=color, x_alpha=alpha, brush=spec.name, **extra)
    return path


def config_dict(config: PainterConfig) -> dict:
    """Plain ``dict`` of a config for experiment logs (tuples become lists)."""
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(config).items()}
