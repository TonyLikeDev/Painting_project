"""Time-lapse export (Week 9): MP4 and GIF frames from the blank canvas to the finished painting, and the stroke files they read."""
from __future__ import annotations

import imageio_ffmpeg
import numpy as np
import pytest
from PIL import Image

from neural_painter.core.image_io import to_uint8
from neural_painter.core.procedural_rasterizer import ProceduralRasterizer
from neural_painter.core.stroke_models import get_brush
from neural_painter.export.video_recorder import write_timelapse, write_timelapses
from neural_painter.pipeline.painter_engine import load_strokes_npz, save_strokes_npz

SIZE = 64


def strokes_for(brush: str, n: int = 12) -> np.ndarray:
    raster = ProceduralRasterizer(brush, SIZE, rng=0)
    return np.stack([raster.initial_params_at(*raster.rng.uniform(0.2, 0.8, 2), raster.rng.random(3)) for _ in range(n)])


def painted(brush: str, strokes: np.ndarray, color: str) -> np.ndarray:
    return to_uint8(ProceduralRasterizer(brush, SIZE, color).render_sequence(strokes)).astype(int)


def mp4_frames(path) -> tuple[list[np.ndarray], dict]:
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    width, height = meta["size"]
    return [np.frombuffer(f, np.uint8).reshape(height, width, 3).astype(int) for f in reader], meta


def test_mp4_runs_from_the_blank_canvas_to_the_painting_in_h264(tmp_path):
    strokes = strokes_for("watercolor")
    n = write_timelapse(tmp_path / "t.mp4", strokes, "watercolor", size=SIZE, canvas_color="white", fps=10, stride=5, hold=0.5)
    assert n == 1 + 3  # blank, after 5 and 10 strokes, and after the 12th (the last)
    frames, meta = mp4_frames(tmp_path / "t.mp4")
    assert meta["codec"].startswith("h264") and meta["pix_fmt"].startswith("yuv420p") and meta["fps"] == 10
    assert len(frames) == n + 5  # half a second of the finished painting at 10 frames a second
    assert np.abs(frames[0] - 255).mean() < 4  # the white canvas (the macOS ffmpeg build decodes it as 253)
    assert np.abs(frames[-1] - painted("watercolor", strokes, "white")).mean() < 4  # H.264 is lossy


def test_gif_shares_one_palette_and_holds_the_end(tmp_path):
    strokes = strokes_for("oilpaintbrush")
    n = write_timelapse(tmp_path / "t.gif", strokes, "oil", size=SIZE, canvas_color="black", fps=20, stride=4, hold=1.0)
    gif = Image.open(tmp_path / "t.gif")
    assert n == gif.n_frames == 4  # blank, after 4, 8 and 12 strokes
    durations, first, last = [], None, None
    for i in range(gif.n_frames):
        gif.seek(i)
        durations.append(gif.info["duration"])
        first = np.array(gif.convert("RGB")).astype(int) if i == 0 else first
        last = np.array(gif.convert("RGB")).astype(int)
    assert durations[0] == 50 and durations[-1] == 1050  # 20 frames a second, one second more on the painting
    assert first.max() <= 2  # the black canvas
    assert np.abs(last - painted("oilpaintbrush", strokes, "black")).mean() < 6  # 256 colours, no dithering


def test_frames_take_the_photo_shape_and_bad_requests_are_refused(tmp_path):
    strokes = strokes_for("rectangle", 3)
    write_timelapse(tmp_path / "wide.mp4", strokes, "rectangle", size=SIZE, shape=(101, 50), fps=5, hold=0)
    frames, meta = mp4_frames(tmp_path / "wide.mp4")
    assert meta["size"] == (102, 50) and len(frames) == 4  # H.264 in 4:2:0 needs even sides
    mp4, gif = write_timelapses(tmp_path / "both", strokes, "rectangle", "white", shape=(40, 30), size=SIZE)
    assert Image.open(gif).size == (40, 30) and mp4.stat().st_size > 0
    with pytest.raises(ValueError, match="mp4"):
        write_timelapse(tmp_path / "t.avi", strokes, "rectangle")
    with pytest.raises(ValueError, match="stride"):
        write_timelapse(tmp_path / "t.gif", strokes, "rectangle", stride=0)


def test_stroke_files_read_back_with_their_brush_and_metadata(tmp_path):
    spec = get_brush("markerpen")
    strokes = np.random.default_rng(1).random((5, spec.dim), dtype=np.float32)
    path = save_strokes_npz(tmp_path / "strokes.npz", strokes, spec, seed=4, canvas="#102030")
    back, back_spec, extra = load_strokes_npz(path)
    np.testing.assert_array_equal(back, strokes)
    assert back_spec.name == "markerpen" and extra == {"seed": 4, "canvas": "#102030"}
    with pytest.raises(ValueError, match="drawn with markerpen"):
        load_strokes_npz(path, "oil")
    original = tmp_path / "apple_strokes.npz"  # the original code's file does not name its brush
    np.savez(original, x_ctt=strokes[None, :, :8], x_color=strokes[None, :, 8:11], x_alpha=strokes[None, :, 11:])
    with pytest.raises(ValueError, match="name the brush"):
        load_strokes_npz(original)
    np.testing.assert_array_equal(load_strokes_npz(original, "marker")[0], strokes)
