"""The web interface (Weeks 8 and 9): the page builds, a painting runs through it to its time-lapse, bad photos are refused,
Stop stops."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

gr = pytest.importorskip("gradio")

from neural_painter.app import gradio_app as app  # noqa: E402
from neural_painter.core.image_io import ImageTooLargeError, UnsupportedImageError, save_image  # noqa: E402
from neural_painter.models.neural_renderer import original_checkpoint_path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class FakeRequest:
    def __init__(self, language: str) -> None:
        self.headers = {"accept-language": language}


def run(generator) -> list[list]:
    """Every update the Paint generator yields, as dicts keyed by output name."""
    return [dict(zip(app.OUTPUTS, update)) for update in generator]


def test_the_page_builds_and_both_languages_have_every_text():
    demo = app.build_app()
    assert isinstance(demo, gr.Blocks)
    assert set(app.TEXT["en"]) == set(app.TEXT["vi"])
    for lang in ("en", "vi"):
        assert "{done}" in app.TEXT[lang]["running"] and "{folder}" in app.TEXT[lang]["done"]
    assert app.language(FakeRequest("vi-VN,vi;q=0.9,en;q=0.8")) == "vi"
    assert app.language(FakeRequest("en-US,en")) == "en" and app.language(None) == "en"
    assert app.language(FakeRequest("en-US,en;q=0.9,vi;q=0.8")) == "vi"  # Vietnamese anywhere in the list
    for lang in ("vi", "en"):  # every label of the page has its text in both languages
        labels = demo.relabel(lang)
        assert len(labels) > 20 and all(isinstance(u, dict) for u in labels)
    assert app.TEXT["vi"]["brush"] in [u.get("label") for u in demo.relabel("vi")]
    assert all(Path(p[0]).is_file() for p in app.samples()) and len(app.samples()) == len(app.SAMPLES)


def test_photo_errors_and_uploads():
    assert app.photo_error("en", ImageTooLargeError("x")) == app.TEXT["en"]["too_large"]
    assert app.photo_error("vi", UnsupportedImageError("unsupported image format '.gif'")) == app.TEXT["vi"]["bad_format"]
    assert app.photo_error("en", UnsupportedImageError("cannot decode x.png")) == app.TEXT["en"]["not_image"]
    assert app.uploaded({"path": "/tmp/abc/photo.jpg", "orig_name": "My photo.jpg"}) == ("/tmp/abc/photo.jpg", "My photo")
    assert app.uploaded("C:/x/apple.png") == ("C:/x/apple.png", "apple") and app.uploaded(None) == (None, "")


def test_bad_photos_are_refused_with_a_message(tmp_path):
    with pytest.raises(gr.Error, match="Choose a photo first"):
        next(app.paint(None, "oilpaintbrush", 100, "progressive", "best", "default", "stretch", 0, "en"))
    junk = tmp_path / "junk.png"
    junk.write_bytes(b"not an image at all")
    with pytest.raises(gr.Error, match="không phải là ảnh"):
        next(app.paint({"path": str(junk), "orig_name": "junk.png"}, "oilpaintbrush", 100, "progressive", "best", "default",
                       "stretch", 0, "vi"))
    gif = tmp_path / "a.gif"
    gif.write_bytes(b"GIF89a")
    with pytest.raises(gr.Error, match="cannot be painted"):
        next(app.paint(str(gif), "oilpaintbrush", 100, "progressive", "best", "default", "stretch", 0, "en"))


def test_a_painting_runs_through_the_page_function(tmp_path, monkeypatch):
    if not original_checkpoint_path("oilpaintbrush", light=True).is_file():
        pytest.skip("the pretrained light oil renderer has not been downloaded")
    monkeypatch.setattr(app, "OUTPUT_ROOT", tmp_path / "output")
    monkeypatch.setattr(app, "FAST_SETTINGS", {"iters_per_stroke": 2})  # a quick painting for the test
    monkeypatch.setattr(app, "DEVICE", "cpu")
    photo = tmp_path / "wide.png"
    save_image(photo, np.random.default_rng(0).random((60, 120, 3)).astype(np.float32))  # 2:1
    out = run(app.paint({"path": str(photo), "orig_name": "wide.png"}, "oilpaintbrush", 100, "progressive", "fast", "default",
                        "stretch", 3, "en"))
    assert out[0]["status"] == "Loading the brush..." and any("Painting: grid" in str(u["status"]) for u in out[1:-1])
    last, video = out[-2], out[-1]  # the painting, then its time-lapse
    assert last["status"].startswith("Done: 55 strokes") and " on CPU." in last["status"]  # 100 // 55 = 1 per block, 5 levels
    assert last["status"].endswith("Making the time-lapse video...") and video["status"] == last["status"].rsplit(" Making", 1)[0]
    assert all(u["live"].shape == (512, 512, 3) for u in out[1:] if not isinstance(u["live"], dict))  # the preview, enlarged
    assert last["tabs"]["selected"] == "compare"  # the page shows the photo next to the painting when it is done
    before, after = last["compare"]["value"]
    assert before.shape == after.shape == (256, 512, 3) and last["painting"]["value"].shape == (256, 512, 3)  # the photo's 2:1
    frames = last["frames"]["frames"]
    assert len(frames) == 1 + 55 and frames[-1].shape == (512, 512, 3)  # the blank canvas, then one frame a stroke
    assert app.replay(0, last["frames"]).shape == (256, 512, 3) and app.replay(10_000, last["frames"]) is not None
    assert (app.replay(0, last["frames"]) == 255).all()  # 0 strokes: the oil brush's white canvas
    assert np.abs(app.replay(55, last["frames"]).astype(int) - last["painting"]["value"]).max() <= 1  # the slider counts strokes: 55 is the end
    assert last["replay"]["maximum"] == 55 and last["replay"]["value"] == 55
    folder = next((tmp_path / "output").glob("*_app_wide_oilpaintbrush"))
    assert {p.name for p in folder.iterdir()} == {"final.png", "final.jpg", "input.png", "strokes.npz", "timelapse.mp4", "timelapse.gif"}
    assert video["timelapse"] == str(folder / "timelapse.mp4") and video["download_gif"]["value"] == str(folder / "timelapse.gif")
    assert last["download_jpg"]["value"] == str(folder / "final.jpg") and "selected" not in video["tabs"]  # the tab stays
    from PIL import Image

    assert Image.open(folder / "timelapse.gif").size == (512, 256)  # in the photo's shape, like the painting


def test_stop_ends_the_worker(monkeypatch, tmp_path):
    """Closing the generator (what the Stop button does) makes the worker stop at its next progress event."""
    stopped = threading.Event()

    def endless(image, renderer, config, callback=None, on_stroke=None, **kwargs):
        canvas = np.zeros((4, 3, 8, 8), np.float32)
        try:
            for k in range(10_000):
                callback(app.Progress(level=1, levels=1, m=2, strokes_done=k, strokes_total=10_000, step=k, loss=0.1, psnr=20.0,
                                      elapsed=0.0, canvas=__import__("torch").from_numpy(canvas)))
                time.sleep(0.01)
        except app.Stopped:
            stopped.set()
            raise

    monkeypatch.setattr(app, "paint_image", endless)
    monkeypatch.setattr(app, "renderer_for", lambda brush: None)
    photo = tmp_path / "p.png"
    save_image(photo, np.full((32, 32, 3), 0.5, np.float32))
    job = app.paint(str(photo), "oilpaintbrush", 100, "progressive", "best", "default", "stretch", 0, "en")
    next(job)  # loading
    update = dict(zip(app.OUTPUTS, next(job)))
    assert "Painting: grid 2 x 2" in update["status"]
    job.close()
    assert stopped.wait(5), "the worker did not stop"
