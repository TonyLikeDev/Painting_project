"""MP4 / GIF time-lapse of a painting, stroke by stroke from the blank canvas (Week 9).

:func:`write_timelapse` draws a painting's strokes again with the procedural rasterizer (about 1.4 s for 500 strokes at 512 pixels,
Table 4.31), starting from the blank canvas, and writes one frame of the blank canvas, one after every ``stride``-th stroke and one
after the last, then holds the finished painting for ``hold`` seconds. The format follows the file's suffix:

* ``.mp4``: H.264 in the ``yuv420p`` pixel format through the ffmpeg bundled with ``imageio-ffmpeg``, which browsers and phones
  play (the original wrote MPEG-4 part 2, ``MP4V``, which they do not); the frames are streamed to ffmpeg;
* ``.gif``: Pillow, every frame mapped without dithering onto one 256-colour palette taken from the finished painting and the
  blank canvas, so that the colours do not flicker from frame to frame; the frames are kept in memory until the file is written.

Strokes have no resolution, so ``size`` draws the frames at any size; ``shape`` then resizes them to ``(width, height)``, e.g. back
to the photograph's aspect ratio as ``final.png`` is. :func:`write_timelapses` writes the MP4 and the GIF that the web app and
``run_paint.py --timelapse`` offer, with the settings below.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator

import imageio_ffmpeg
import numpy as np
from PIL import Image

from ..core.image_io import resize, to_uint8
from ..core.procedural_rasterizer import ProceduralRasterizer
from ..core.stroke_models import get_brush

FORMATS = (".mp4", ".gif")
MP4_QUALITY = 6  # imageio-ffmpeg's 1 to 10 scale; 6 is x264's constant rate factor 20

# The time-lapses of the app and run_paint.py, both at the painting's size: the MP4 at 40 frames a second like the original's video,
# at most 500 frames (12.5 s); the GIF shorter, about 100 frames at 20 a second (about 0.5 MB for 500 oil strokes at 512 pixels).
MP4_SETTINGS = {"fps": 40.0, "frames": 500, "hold": 1.5}
GIF_SETTINGS = {"fps": 20.0, "frames": 100, "hold": 1.5}


def frames(strokes: np.ndarray, brush, size: int = 512, canvas_color=None, shape: tuple[int, int] | None = None,
           stride: int = 1) -> Iterator[np.ndarray]:
    """The ``uint8`` frames of the time-lapse: the blank canvas, then the canvas after every ``stride``-th stroke and the last."""
    spec = get_brush(brush)
    raster = ProceduralRasterizer(spec, size, canvas_color or spec.default_canvas_color)
    arr = np.asarray(strokes, dtype=np.float32).reshape(-1, spec.dim)
    width, height = shape or (size, size)

    def frame(canvas: np.ndarray) -> np.ndarray:
        return to_uint8(canvas if (width, height) == (size, size) else resize(canvas, (width, height)))

    yield frame(raster.canvas)
    for i, p in enumerate(arr):  # render_sequence's loop, as a generator
        if raster.is_visible(p):
            raster.draw(p)
        if (i + 1) % stride == 0 or i == len(arr) - 1:
            yield frame(raster.canvas)


def write_timelapse(path: str | Path, strokes: np.ndarray, brush, *, size: int = 512, canvas_color=None,
                    shape: tuple[int, int] | None = None, fps: float = 40.0, stride: int = 1, hold: float = 1.0) -> int:
    """Write the time-lapse of ``strokes`` (``(N, d)`` in drawing order) to ``path`` (.mp4 or .gif); returns the number of frames
    drawn (the blank canvas, every ``stride``-th stroke and the last), not counting the ``hold`` seconds at the end."""
    path = Path(path)
    if path.suffix.lower() not in FORMATS:
        raise ValueError(f"{path.name}: a time-lapse is an .mp4 or a .gif file")
    if stride < 1 or fps <= 0 or hold < 0:
        raise ValueError(f"need stride >= 1, fps > 0 and hold >= 0 (got {stride}, {fps}, {hold})")
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = shape or (size, size)
    source = frames(strokes, brush, size, canvas_color, (width, height), stride)
    if path.suffix.lower() == ".mp4":
        writer = imageio_ffmpeg.write_frames(str(path), (width, height), fps=fps, quality=MP4_QUALITY, macro_block_size=2,
                                             output_params=["-movflags", "+faststart"])  # odd sides are scaled to even ones
        writer.send(None)
        count = 0
        for count, image in enumerate(source, start=1):
            writer.send(image.tobytes())
        for _ in range(round(hold * fps)):
            writer.send(image.tobytes())
        writer.close()
        return count
    images = list(source)
    palette = Image.fromarray(np.concatenate([images[-1], images[0]])).quantize(256)  # the painting's colours and the canvas's
    gif = [Image.fromarray(image).quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    step = 1000.0 / fps
    gif[0].save(path, save_all=True, append_images=gif[1:], duration=[step] * (len(gif) - 1) + [step + 1000.0 * hold], loop=0)
    return len(images)


def write_timelapses(folder: str | Path, strokes: np.ndarray, brush, canvas_color=None, shape: tuple[int, int] | None = None,
                     size: int = 512) -> tuple[Path, Path]:
    """``timelapse.mp4`` and ``timelapse.gif`` in ``folder``, with :data:`MP4_SETTINGS` and :data:`GIF_SETTINGS`; ``shape`` is the
    painting's ``(width, height)`` (default square)."""
    folder = Path(folder)
    paths = folder / "timelapse.mp4", folder / "timelapse.gif"
    for path, settings in zip(paths, (MP4_SETTINGS, GIF_SETTINGS)):
        write_timelapse(path, strokes, brush, size=size, canvas_color=canvas_color, shape=shape, fps=settings["fps"],
                        stride=max(1, math.ceil(len(strokes) / settings["frames"])), hold=settings["hold"])
    return paths
