"""Whole images painted at one or several grid sizes (Week 7).

:func:`paint_image` paints an image at the grid sizes of ``core.grid.grid_levels``, one level after the other:

* ``progressive`` (the original ``demo_prog.py``): ``1 x 1, 2 x 2, ..., M x M``. A level cuts the target into ``m x m`` blocks,
  optimizes its strokes with :class:`PainterEngine` starting from the canvas the previous levels left (their strokes
  rasterized at the output size, then cut into the level's blocks), maps the strokes to image coordinates and draws them
  on top of the painting;
* ``fixed`` (``demo.py``): the single level ``M x M``; with the equal allocation it is :func:`paint_fixed_grid` (same random
  draws, same strokes; checked in the tests);
* ``full``: the single level ``1 x 1``.

How many strokes a block gets (``PainterConfig.allocation``):

* ``equal``, the original: every block of every level gets ``total // sum(m^2)`` strokes (the original's ``stroke_parser``: 9
  for 500 strokes and ``M = 5``, so 495 strokes are painted);
* ``error``: each level gets its share of the budget in proportion to its number of blocks, exactly ``total`` in all
  (``core.grid.level_budgets``), and the share goes to the blocks in proportion to their mean absolute error on the canvas
  the level starts from (:func:`allocate`, at most twice the level's mean per block), so blocks that are already right, such
  as a flat background, get fewer strokes and detailed ones more.

The canvas is rasterized incrementally: each level draws its strokes on top of the previous levels' canvas, which is what the
original's re-rendering of all strokes so far computes, with one pass over the strokes instead of one per level.

Differences from the original, besides those of the engine: the target of every level is shrunk from the source image (the
original shrinks an already shrunk ``M S`` pixel copy), by area averaging unless ``PainterConfig.resize = "linear"``, and so is
the canvas handed to the next level; the block shuffle is drawn from the engine's seeded generator.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from ..core.differentiable_canvas import blank_canvas
from ..core.grid import block_to_global, flatten_blocks, grid_levels, img2patches, level_budgets, patches2img, strokes_per_block
from ..core.procedural_rasterizer import ProceduralRasterizer
from ..models.neural_renderer import NeuralRenderer
from .painter_engine import RESIZES, PainterConfig, PainterEngine, PaintResult

ALLOCATION_CAP = 2.0  # allocation "error": a block gets at most this many times the level's mean strokes per block

# The pipeline's settings, tuned in Week 7 (report Section 4.4.3), shared by run_paint.py and the web interface.
# PainterConfig keeps the reference's defaults so that older experiments stay reproducible from their records.
PIPELINE_SETTINGS = {"lr": 0.004, "lr_schedule": "cosine"}
FAST_SETTINGS = {"allocation": "error", "iters_per_stroke": 11}  # the same PSNR in less than half the time (oil, Table 4.28)


def allocate(weights: np.ndarray, budget: int, cap: int) -> np.ndarray:
    """Split ``budget`` strokes among blocks in proportion to ``weights`` (D'Hondt's method), at most ``cap`` per block.

    Strokes are handed out one at a time to the block with the largest ``weight / (count + 1)`` among those under the cap,
    so the counts are as proportional as integers allow. All-zero weights (a perfect canvas) split the budget evenly.
    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if budget > cap * w.size:
        raise ValueError(f"{budget} strokes do not fit in {w.size} blocks of at most {cap}")
    if not (w > 0).any():
        w = np.ones_like(w)
    counts = np.zeros(w.size, dtype=np.int64)
    for _ in range(budget):
        counts[int(np.argmax(np.where(counts < cap, w / (counts + 1), -np.inf)))] += 1
    return counts


@dataclass
class Level:
    m: int
    counts: np.ndarray  # (m * m,) strokes given to each block
    result: PaintResult
    strokes: np.ndarray  # (n, d) this level's strokes in image coordinates, in drawing order


@dataclass
class Painting:
    image: np.ndarray  # (canvas_size, canvas_size, 3) float32, the rasterized painting
    strokes: np.ndarray  # (N, d) every stroke in image coordinates, in drawing order
    levels: list[Level]
    render_seconds: float  # rasterizing the strokes (the canvas handed to each level and the final image are the same pass)

    @property
    def result(self) -> PaintResult:
        """The finest level's optimization (its neural canvas is the one a fixed grid of the same size has)."""
        return self.levels[-1].result

    @property
    def optimize_seconds(self) -> float:
        return sum(level.result.seconds for level in self.levels)

    @property
    def steps(self) -> int:
        return sum(len(level.result.loss) for level in self.levels)


@dataclass
class Progress:
    """What a progress callback receives after every new stroke of every block has been optimized."""

    level: int  # 1-based index of the level being painted
    levels: int
    m: int  # its grid size
    strokes_done: int  # over all levels
    strokes_total: int
    step: int  # optimizer steps so far, over all levels
    loss: float
    psnr: float  # dB, the level's neural canvas against its target blocks
    elapsed: float  # seconds since the painting started
    canvas: torch.Tensor  # (m * m, 3, S, S) the level's neural canvas, detached

    def preview(self) -> np.ndarray:
        """The neural canvas as one ``(m S, m S, 3)`` image, for a live preview."""
        return patches2img(self.canvas, self.m).clip(0.0, 1.0)


def paint_image(
    image: np.ndarray,
    renderer: NeuralRenderer,
    config: PainterConfig | None = None,
    mode: str = "progressive",
    max_divide: int = 5,
    strokes: int = 500,
    canvas_size: int = 512,
    canvas_color: str | None = None,
    device: str | torch.device | None = None,
    engine: PainterEngine | None = None,
    callback: Callable[[Progress], None] | None = None,
    on_stroke: Callable[[int, np.ndarray], None] | None = None,
) -> Painting:
    """Paint ``image`` ((H, W, 3) float RGB in [0, 1], stretched to a square like the original) with about ``strokes`` strokes.

    ``mode`` and ``max_divide`` choose the grids (``core.grid.grid_levels``); ``config.allocation`` how the strokes are shared
    (``config.strokes_per_block`` is not used: the budget is ``strokes``). Returns the painting rasterized at ``canvas_size``
    pixels, its strokes in drawing order and every level's optimization history. ``on_stroke(i, canvas)`` is called after the
    ``i``-th stroke (counted over all levels) is drawn on the ``canvas_size`` painting, for a replay or a time-lapse.
    """
    engine = engine or PainterEngine(renderer, config, device)
    cfg, spec, dev, size = engine.config, engine.spec, engine.device, renderer.out_size
    color = canvas_color or spec.default_canvas_color
    grids = grid_levels(mode, max_divide)
    if cfg.allocation == "equal":
        per_block = strokes_per_block(strokes, grids)
        if per_block < 1:
            raise ValueError(f"{strokes} strokes are too few for the grids {grids} (an equal split needs at least {sum(m * m for m in grids)})")
        budgets = [per_block * m * m for m in grids]
    else:
        budgets = level_budgets(strokes, grids)
    total = sum(budgets)
    raster = ProceduralRasterizer(spec, canvas_size, color)
    interpolation = RESIZES[cfg.resize]
    levels: list[Level] = []
    done, steps, render_seconds, started = 0, 0, 0.0, time.perf_counter()
    for index, (m, budget) in enumerate(zip(grids, budgets), start=1):
        if budget == 0:  # a tiny budget leaves nothing to the coarsest levels
            continue
        targets = img2patches(image, m, size, dev, interpolation=interpolation)
        base = img2patches(raster.canvas, m, size, dev, interpolation=interpolation) if levels else blank_canvas(m * m, size, color, dev)
        if cfg.allocation == "equal":
            counts = np.full(m * m, budget // (m * m), dtype=np.int64)
        else:
            error = (targets - base).abs().mean(dim=(1, 2, 3)).cpu().numpy()
            counts = allocate(error, budget, max(1, math.ceil(ALLOCATION_CAP * budget / (m * m))))

        def report(anchor: int, step: int, loss: float, psnr: float, canvas: torch.Tensor, m=m, counts=counts, index=index) -> None:
            callback(Progress(
                level=index, levels=len(grids), m=m, strokes_done=done + int(np.minimum(counts, anchor + 1).sum()), strokes_total=total,
                step=steps + step, loss=loss, psnr=psnr, elapsed=time.perf_counter() - started, canvas=canvas,
            ))

        result = engine.paint(targets, canvas=base, canvas_color=color, strokes=counts, callback=report if callback else None)
        order = engine.rng.permutation(m * m)
        new = flatten_blocks(block_to_global(result.params, spec, m), order, result.active)
        t0 = time.perf_counter()
        offset = done  # strokes of the earlier levels
        raster.render_sequence(new, reset=False, on_stroke=(lambda i, canvas: on_stroke(offset + i, canvas)) if on_stroke else None)
        render_seconds += time.perf_counter() - t0
        levels.append(Level(m=m, counts=counts, result=result, strokes=new))
        done += len(new)
        steps += len(result.loss)
    return Painting(image=raster.canvas, strokes=np.concatenate([lv.strokes for lv in levels]), levels=levels, render_seconds=render_seconds)
