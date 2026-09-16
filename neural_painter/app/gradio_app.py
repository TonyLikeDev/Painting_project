"""Gradio web interface (Week 8).

Planned layout (proposal section 3, RESEARCH_PLAN Week 8): image upload or sample
picker (``core.image_io.list_sample_images``), brush selector (four brushes),
stroke-count slider, grid mode (full image / fixed grid / progressive), canvas
colour, start button, live progress bar and intermediate canvas preview fed by the
``ProgressivePainter`` progress callback, before/after comparison, stroke-by-stroke
playback slider, and download buttons for PNG/JPG, MP4 and GIF (Week 9).
Run with ``python -m neural_painter.app.gradio_app``.
"""
from __future__ import annotations


def build_app():  # pragma: no cover - stub
    raise NotImplementedError("the Gradio app is scheduled for Week 8 (see ROADMAP.md)")


def main() -> int:  # pragma: no cover - stub
    build_app().launch()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
