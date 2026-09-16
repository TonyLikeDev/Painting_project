"""Coarse-to-fine progressive painting (Week 7).

Planned interface: :class:`ProgressivePainter(engine, schedule)`` iterates the
levels of ``core.grid.progressive_schedule`` (``1x1 -> 2x2 -> ... -> MxM``): split
the target with ``core.grid.split_grid``, optimize the level's strokes with
``PainterEngine`` starting from the previous canvas, convert block coordinates to
global ones with ``core.grid.block_to_global``, re-render the accumulated strokes
with the procedural rasterizer at the next level's resolution, and emit
``(strokes_done, loss, elapsed)`` progress events for the UI. Dynamic stroke
allocation and learning-rate scheduling are the modernization items of Week 7.
"""
from __future__ import annotations


class ProgressivePainter:  # pragma: no cover - stub
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("ProgressivePainter is scheduled for Week 7 (see ROADMAP.md)")
