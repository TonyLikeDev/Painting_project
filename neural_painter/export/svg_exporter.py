"""SVG vector export (Week 9, stretch).

Planned interface: :func:`strokes_to_svg(strokes, brush, size, canvas_color, path)``.
Bezier brushes become ``<path d="M x0 y0 Q x1 y1 x2 y2">`` with stroke width from
the radii (variable width approximated by a filled outline); tape strokes become
rotated ``<rect>``; oil strokes become an ``<image>`` of the brush texture inside a
``<g transform="translate rotate scale">``, with the colour gradient as a
``<linearGradient>`` mask.
"""
from __future__ import annotations


def strokes_to_svg(strokes, brush, size: int, canvas_color, path):  # pragma: no cover - stub
    raise NotImplementedError("SVG export is scheduled for Week 9 (stretch goal, see ROADMAP.md)")
