"""MP4 / GIF time-lapse export (Week 9).

Planned interface: :class:`VideoRecorder(path, fps=40, size=(w, h), stride=1)``
used as the ``on_stroke`` callback of ``ProceduralRasterizer.render_sequence``;
writes MP4 through ``imageio-ffmpeg`` (H.264, browser-playable, unlike the original
``MP4V`` fourcc) and GIF through ``imageio`` with palette quantization. ``stride``
keeps every n-th frame to bound file size.
"""
from __future__ import annotations


class VideoRecorder:  # pragma: no cover - stub
    def __init__(self, path, fps: int = 40, size=None, stride: int = 1) -> None:
        raise NotImplementedError("VideoRecorder is scheduled for Week 9 (see ROADMAP.md)")
