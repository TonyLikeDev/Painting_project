"""Web interface (Week 8): a photograph in, a painting out, in the browser.

Run ``python -m neural_painter.app.gradio_app`` (or double-click ``Start_App.bat`` on Windows, ``start_app.command`` on macOS); the
page opens at http://127.0.0.1:7860. The painting is ``pipeline.progressive_painter.paint_image`` with the pipeline's settings
(``PIPELINE_SETTINGS``, plus ``FAST_SETTINGS`` for the fast option), run in a worker thread whose ``Progress`` events update a live
preview and a progress bar. The finished painting is shown next to the photograph (a before / after slider), can be replayed
stroke by stroke and watched as a time-lapse video (Week 9, ``export.video_recorder``, made right after the painting), and is saved
with its strokes, the MP4 and the GIF under ``output/`` like ``scripts/run_paint.py --timelapse``; every file has a download button. Photos that cannot
be painted (not an image, an unsupported format, too many pixels) are refused with a message on the page: the Paint event takes
the uploaded file unprocessed, so that ``image_io`` and not Gradio decides. The page speaks Vietnamese or English: it starts in
Vietnamese when the browser lists Vietnamese among its languages, and a switch at the top changes every label. (Gradio's own
translation mechanism only activates the languages it ships, which do not include Vietnamese, so the labels are set from Python.)
One painting runs at a time; others wait in the queue.
"""
from __future__ import annotations

import queue
import threading
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from ..core.device import disable_power_throttling, get_device
from ..core.image_io import ImageTooLargeError, UnsupportedImageError, preprocess, resize, save_image, to_uint8
from ..core.procedural_rasterizer import canvas_rgb
from ..core.stroke_models import get_brush
from ..export.video_recorder import write_timelapses
from ..models.neural_renderer import load_original_checkpoint, original_checkpoint_path
from ..pipeline.painter_engine import PainterConfig, save_strokes_npz
from ..pipeline.progressive_painter import FAST_SETTINGS, PIPELINE_SETTINGS, Progress, paint_image
from ..pipeline.train_renderer import new_experiment_dir

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = ROOT / "data" / "eval_set"
SAMPLES = ("apple", "iceland", "jay", "sunflowers", "alien", "51e3d4424d53b10ff3d8992cc12c30771037dbf85254784b7d297ed7934b_640")
OUTPUT_ROOT = ROOT / "output"
SIZE = 512  # the photo is fitted to and the painting drawn at SIZE x SIZE pixels
REPLAY_FRAMES = 100  # frames kept for the replay slider
PREVIEW_EVERY = 0.3  # seconds between two updates of the live preview
MAX_UPLOAD = "40mb"
DEVICE = None  # None: cuda, then mps, then cpu (core.device.get_device)

TEXT = {
    "en": {
        "title": "## Neural painter\nTurn a photo into a painting made of brush strokes. Choose a photo and a brush, then press **Paint**.",
        "photo": "Photo", "samples": "Or pick a sample photo", "brush": "Brush", "oil": "Oil paint", "watercolor": "Watercolour",
        "marker": "Marker pen", "tape": "Coloured tape", "strokes": "Number of strokes",
        "strokes_info": "More strokes give more detail but take longer.", "mode": "How to paint",
        "progressive": "Coarse to fine (recommended)", "fixed": "Fixed 5 x 5 grid", "full": "Whole image at once", "speed": "Speed",
        "best": "Best quality", "fast": "Fast (about twice as fast)", "canvas": "Canvas colour", "canvas_default": "Brush default",
        "white": "White", "black": "Black", "advanced": "More options", "fit": "Photo shape", "stretch": "Keep the photo's shape",
        "crop": "Crop to a square", "pad": "Add borders to make a square", "seed": "Random seed (the same seed gives the same painting)",
        "paint": "Paint", "stop": "Stop", "live": "Painting in progress", "compare": "Photo and painting",
        "painting": "Painting", "replay": "Replay", "replay_slider": "Strokes painted: drag to replay the painting stroke by stroke",
        "timelapse": "Time-lapse", "download": "**Download**", "download_png": "Painting (PNG)", "download_jpg": "Painting (JPG)",
        "download_mp4": "Time-lapse (MP4)", "download_gif": "Time-lapse (GIF)", "download_npz": "Strokes (.npz)",
        "idle": "Choose a photo, then press **Paint**.",
        "loading": "Loading the brush...",
        "running": "Painting: grid {m} x {m} ({level} of {levels}), {done} of {total} strokes, {elapsed:.0f} s",
        "done": "Done: {strokes} strokes in {seconds:.0f} s on {device}. Saved in `{folder}`.",
        "making": "Making the time-lapse video...",
        "stopped": "Stopped. Press **Paint** to start again.",
        "no_photo": "Choose a photo first.",
        "bad_format": "This file type cannot be painted. Use a JPG, PNG, BMP, WebP or TIFF photo.",
        "not_image": "This file is not an image, or it is damaged.",
        "too_large": "This photo is too large (more than 40 million pixels). Use a smaller copy.",
        "no_renderer": "The renderer for this brush is not installed ({path}).",
        "failed": "Painting failed: {error}",
        "error_title": "Cannot paint",
    },
    "vi": {
        "title": "## Vẽ tranh từ ảnh\nBiến một bức ảnh thành tranh vẽ bằng các nét cọ. Chọn ảnh và loại cọ, rồi bấm **Vẽ**.",
        "photo": "Ảnh", "samples": "Hoặc chọn một ảnh mẫu", "brush": "Loại cọ", "oil": "Sơn dầu", "watercolor": "Màu nước",
        "marker": "Bút dạ", "tape": "Băng dính màu", "strokes": "Số nét cọ",
        "strokes_info": "Nhiều nét hơn thì chi tiết hơn nhưng vẽ lâu hơn.", "mode": "Cách vẽ",
        "progressive": "Từ thô đến mịn (khuyên dùng)", "fixed": "Lưới cố định 5 x 5", "full": "Cả ảnh một lần", "speed": "Tốc độ",
        "best": "Chất lượng tốt nhất", "fast": "Nhanh (nhanh gấp khoảng 2 lần)", "canvas": "Màu nền", "canvas_default": "Mặc định của cọ",
        "white": "Trắng", "black": "Đen", "advanced": "Tùy chọn khác", "fit": "Hình dạng ảnh", "stretch": "Giữ tỉ lệ của ảnh",
        "crop": "Cắt thành hình vuông", "pad": "Thêm viền thành hình vuông", "seed": "Số ngẫu nhiên (cùng số thì cùng bức tranh)",
        "paint": "Vẽ", "stop": "Dừng", "live": "Đang vẽ", "compare": "Ảnh và tranh",
        "painting": "Tranh", "replay": "Xem lại", "replay_slider": "Số nét đã vẽ: kéo để xem lại từng nét cọ",
        "timelapse": "Video quá trình vẽ", "download": "**Tải về**", "download_png": "Tranh (PNG)", "download_jpg": "Tranh (JPG)",
        "download_mp4": "Video (MP4)", "download_gif": "Ảnh động (GIF)", "download_npz": "Dữ liệu nét cọ (.npz)",
        "idle": "Chọn ảnh rồi bấm **Vẽ**.",
        "loading": "Đang tải cọ...",
        "running": "Đang vẽ: lưới {m} x {m} ({level}/{levels}), {done}/{total} nét, {elapsed:.0f} giây",
        "done": "Xong: {strokes} nét trong {seconds:.0f} giây trên {device}. Đã lưu vào `{folder}`.",
        "making": "Đang tạo video quá trình vẽ...",
        "stopped": "Đã dừng. Bấm **Vẽ** để vẽ lại.",
        "no_photo": "Hãy chọn ảnh trước.",
        "bad_format": "Không vẽ được loại tệp này. Hãy dùng ảnh JPG, PNG, BMP, WebP hoặc TIFF.",
        "not_image": "Tệp này không phải là ảnh, hoặc đã bị hỏng.",
        "too_large": "Ảnh quá lớn (hơn 40 triệu điểm ảnh). Hãy dùng bản nhỏ hơn.",
        "no_renderer": "Chưa có bộ renderer cho loại cọ này ({path}).",
        "failed": "Vẽ không thành công: {error}",
        "error_title": "Không vẽ được",
    },
}
LANGUAGES = [("Tiếng Việt", "vi"), ("English", "en")]
SAMPLES_LABEL = "Ảnh mẫu / Sample photos"  # gr.Examples cannot be relabelled, so it says both

# the choices of the radio buttons: (text key, value)
BRUSHES = (("oil", "oilpaintbrush"), ("watercolor", "watercolor"), ("marker", "markerpen"), ("tape", "rectangle"))
MODES = (("progressive", "progressive"), ("fixed", "fixed"), ("full", "full"))
SPEEDS = (("best", "best"), ("fast", "fast"))
CANVASES = (("canvas_default", "default"), ("white", "white"), ("black", "black"))
FITS = (("stretch", "stretch"), ("crop", "crop"), ("pad", "pad"))

# the outputs of the Paint event, in order; paint() yields one value per name (gr.skip() leaves an output as it is)
OUTPUTS = ("status", "live", "compare", "painting", "replay", "replay_image", "frames", "timelapse", "download_png", "download_jpg",
           "download_mp4", "download_gif", "download_npz", "tabs")


def updates(**values) -> list:
    return [values.get(name, gr.skip()) for name in OUTPUTS]


def language(request: gr.Request | None) -> str:
    """The page's first language: Vietnamese when the browser lists it at all, else English."""
    header = (request.headers.get("accept-language", "") if request is not None else "") or ""
    return "vi" if any(tag.strip().lower().startswith("vi") for tag in header.split(",")) else "en"


def first_language(request: gr.Request) -> str:
    return language(request)


def say(lang: str, key: str, **values) -> str:
    return TEXT[lang][key].format(**values)


def problem(lang: str, message: str) -> gr.Error:
    """The error box shown on the page, titled in the page's language; ``print_exception=False``: a refused photo is no crash."""
    return gr.Error(message, title=say(lang, "error_title"), print_exception=False)


def device_name(device) -> str:
    """What the status line calls the device: GPU or CPU, not ``cuda:0``."""
    return {"cuda": "GPU", "mps": "GPU (Apple)"}.get(getattr(device, "type", str(device)), "CPU")


def photo_error(lang: str, exc: Exception) -> str:
    """The message for a photo that ``image_io`` refused."""
    if isinstance(exc, ImageTooLargeError):
        return say(lang, "too_large")
    if isinstance(exc, UnsupportedImageError) and str(exc).startswith("unsupported"):
        return say(lang, "bad_format")
    return say(lang, "not_image")


def uploaded(payload) -> tuple[str | None, str]:
    """``(path, name)`` of the unprocessed image payload (a dict or FileData from the page, or a plain path in tests)."""
    if payload is None:
        return None, ""
    if isinstance(payload, (str, Path)):
        return str(payload), Path(payload).stem
    data = payload if isinstance(payload, dict) else vars(payload)
    path = data.get("path")
    return path, Path(data.get("orig_name") or path or "photo").stem


_RENDERERS: dict = {}


def renderer_for(brush: str):
    """The authors' light renderer of ``brush``, loaded once."""
    if brush not in _RENDERERS:
        path = original_checkpoint_path(brush, light=True)
        if not path.is_file():
            raise FileNotFoundError(path)
        _RENDERERS[brush] = load_original_checkpoint(path, brush, True)
    return _RENDERERS[brush]


class Stopped(Exception):
    """Raised in the worker thread when the page stopped the painting."""


def paint(photo, brush, strokes, mode, speed, canvas, fit, seed, lang="en", progress=gr.Progress()):
    """Generator behind the Paint button: yields live updates, then the finished painting (lists in the order of OUTPUTS)."""
    lang = lang if lang in TEXT else "en"
    path, name = uploaded(photo)
    if not path:
        raise problem(lang, say(lang, "no_photo"))
    try:
        fitted = preprocess(path, size=SIZE, mode=fit)
    except (FileNotFoundError, UnsupportedImageError, ImageTooLargeError) as exc:
        raise problem(lang, photo_error(lang, exc)) from None
    yield updates(status=say(lang, "loading"))
    try:
        renderer = renderer_for(brush)
    except FileNotFoundError as exc:
        raise problem(lang, say(lang, "no_renderer", path=exc)) from None
    device, seed, strokes = get_device(DEVICE), int(seed or 0), int(strokes)
    config = PainterConfig(seed=seed, **{**PIPELINE_SETTINGS, **(FAST_SETTINGS if speed == "fast" else {})})
    spec = get_brush(brush)
    color = spec.default_canvas_color if canvas == "default" else canvas
    events: queue.Queue = queue.Queue()
    stop, result = threading.Event(), {}
    every = max(1, strokes // REPLAY_FRAMES)
    counts, frames = [0], [to_uint8(np.ones((SIZE, SIZE, 3), np.float32) * canvas_rgb(color))]  # the replay starts on the blank canvas

    def on_progress(event: Progress) -> None:
        if stop.is_set():
            raise Stopped
        events.put(event)

    def on_stroke(i: int, canvas_now: np.ndarray) -> None:
        if (i + 1) % every == 0:
            counts.append(i + 1)
            frames.append(to_uint8(canvas_now))

    def work() -> None:
        try:
            result["paint"] = paint_image(fitted.image, renderer, config, mode=mode, strokes=strokes, canvas_size=SIZE,
                                          canvas_color=color, device=device,
                                          callback=on_progress, on_stroke=on_stroke)
        except Stopped:
            result["stopped"] = True
        except Exception as exc:  # shown on the page instead of ending the worker silently
            result["error"] = exc
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True).start()
    try:
        finished = False
        while not finished:
            latest = None
            try:
                item = events.get(timeout=PREVIEW_EVERY)
                while True:  # keep only the newest event
                    if item is None:
                        finished = True
                        break
                    latest = item
                    item = events.get_nowait()
            except queue.Empty:
                pass
            if latest is not None:
                progress(latest.strokes_done / latest.strokes_total)
                yield updates(status=say(lang, "running", m=latest.m, level=latest.level, levels=latest.levels, done=latest.strokes_done,
                                         total=latest.strokes_total, elapsed=latest.elapsed),
                              live=to_uint8(resize(latest.preview(), SIZE, interpolation=cv2.INTER_NEAREST)))  # the 32 to 160 px canvas, enlarged
    finally:
        stop.set()  # the page stopped the event (Stop, a closed tab): the worker stops at its next progress event
    if result.get("stopped"):
        yield updates(status=say(lang, "stopped"))
        return
    if "error" in result:
        raise problem(lang, say(lang, "failed", error=result["error"]))

    painting = result["paint"]
    shape = fitted.output_size(SIZE, keep_aspect_ratio=fit == "stretch")  # (width, height)
    final = painting.image if shape == (SIZE, SIZE) else resize(painting.image, shape)
    photo_view = fitted.image if shape == (SIZE, SIZE) else resize(fitted.image, shape)
    out = new_experiment_dir(OUTPUT_ROOT, f"{name}_{brush}", kind="app")
    save_image(out / "final.png", final)
    save_image(out / "final.jpg", final)
    save_image(out / "input.png", fitted.image)
    save_strokes_npz(out / "strokes.npz", painting.strokes, spec, seed=seed, canvas=color)
    if counts[-1] != len(painting.strokes):
        counts.append(len(painting.strokes))
        frames.append(to_uint8(painting.image))
    folder = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    done = say(lang, "done", strokes=len(painting.strokes), seconds=painting.optimize_seconds + painting.render_seconds,
               device=device_name(device), folder=folder.as_posix())
    # images go as gr.update(value=...): Gradio 6.27 sends the next yield's gr.skip() as a diff that blanks a raw image value
    # in a tab not opened yet (the Painting and Replay tabs stayed empty)
    yield updates(
        status=f"{done} {say(lang, 'making')}",
        live=gr.update(value=to_uint8(painting.image)),
        compare=gr.update(value=(to_uint8(photo_view), to_uint8(final))),
        painting=gr.update(value=to_uint8(final)),
        replay=gr.update(minimum=0, maximum=len(painting.strokes), step=every, value=len(painting.strokes), interactive=True),
        replay_image=gr.update(value=to_uint8(final)),
        frames={"frames": frames, "counts": counts, "shape": shape},
        download_png=gr.update(value=str(out / "final.png"), interactive=True),
        download_jpg=gr.update(value=str(out / "final.jpg"), interactive=True),
        download_npz=gr.update(value=str(out / "strokes.npz"), interactive=True),
        tabs=gr.update(selected="compare"),  # show the photo next to the painting as soon as it is done
    )
    mp4, gif = write_timelapses(out, painting.strokes, spec, color, shape=shape, size=SIZE)  # a few seconds (500 strokes: about 4 s)
    yield updates(status=done, timelapse=str(mp4), download_mp4=gr.update(value=str(mp4), interactive=True),
                  download_gif=gr.update(value=str(gif), interactive=True))


def replay(strokes, state):
    """The painting after about ``strokes`` strokes (the nearest kept frame; 0 is the blank canvas), in the photo's shape."""
    if not state or not state.get("frames"):
        return None
    frame = state["frames"][int(np.argmin(np.abs(np.asarray(state["counts"]) - float(strokes))))]
    shape = tuple(state["shape"])
    return frame if shape == (frame.shape[1], frame.shape[0]) else to_uint8(resize(frame, shape))


def samples() -> list[list[str]]:
    return [[str(SAMPLE_DIR / f"{name}.png")] for name in SAMPLES if (SAMPLE_DIR / f"{name}.png").is_file()]


def build_app() -> gr.Blocks:
    t = TEXT["en"]
    labelled = []  # (component, {property: text key or [(text key, value), ...]}) relabelled when the language changes

    def label(component, **texts):
        labelled.append((component, texts))
        return component

    with gr.Blocks(title="Neural painter") as demo:
        with gr.Row():
            title = label(gr.Markdown(t["title"]), value="title")
            lang = gr.Radio(LANGUAGES, value="en", show_label=False, container=False, scale=0, min_width=220)
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                photo = label(gr.Image(type="filepath", image_mode=None, sources=["upload", "clipboard"], label=t["photo"], height=300),
                              label="photo")
                if samples():
                    gr.Examples(samples(), inputs=photo, label=SAMPLES_LABEL, examples_per_page=6)
                brush = label(gr.Radio([(t[k], v) for k, v in BRUSHES], value="oilpaintbrush", label=t["brush"]),
                              label="brush", choices=BRUSHES)
                strokes = label(gr.Slider(100, 1000, value=500, step=50, label=t["strokes"], info=t["strokes_info"]),
                                label="strokes", info="strokes_info")
                mode = label(gr.Radio([(t[k], v) for k, v in MODES], value="progressive", label=t["mode"]), label="mode", choices=MODES)
                speed = label(gr.Radio([(t[k], v) for k, v in SPEEDS], value="best", label=t["speed"]), label="speed", choices=SPEEDS)
                with label(gr.Accordion(t["advanced"], open=False), label="advanced"):
                    canvas = label(gr.Radio([(t[k], v) for k, v in CANVASES], value="default", label=t["canvas"]),
                                   label="canvas", choices=CANVASES)
                    fit = label(gr.Radio([(t[k], v) for k, v in FITS], value="stretch", label=t["fit"]), label="fit", choices=FITS)
                    seed = label(gr.Number(value=0, precision=0, minimum=0, label=t["seed"]), label="seed")
                with gr.Row():
                    start = label(gr.Button(t["paint"], variant="primary", size="lg"), value="paint")
                    stop = label(gr.Button(t["stop"], variant="stop", size="lg"), value="stop")
            with gr.Column(scale=2, min_width=420):
                status = label(gr.Markdown(t["idle"]), value="idle")
                with gr.Tabs(selected="live") as tabs:
                    with label(gr.Tab(t["live"], id="live"), label="live"):
                        live = label(gr.Image(label=t["live"], interactive=False, height=460), label="live")
                    with label(gr.Tab(t["compare"], id="compare"), label="compare"):
                        compare = label(gr.ImageSlider(label=t["compare"], interactive=False, max_height=520), label="compare")
                    with label(gr.Tab(t["painting"], id="painting"), label="painting"):
                        painting = label(gr.Image(label=t["painting"], interactive=False, height=460), label="painting")
                    with label(gr.Tab(t["replay"], id="replay"), label="replay"):
                        replay_image = label(gr.Image(label=t["replay"], interactive=False, height=460), label="replay")
                        replay_slider = label(gr.Slider(0, 1, value=0, step=1, label=t["replay_slider"], interactive=False),
                                              label="replay_slider")
                    with label(gr.Tab(t["timelapse"], id="timelapse"), label="timelapse"):
                        timelapse = label(gr.Video(label=t["timelapse"], interactive=False, autoplay=True, loop=True, height=460),
                                          label="timelapse")
                label(gr.Markdown(t["download"]), value="download")
                with gr.Row():
                    downloads = {name: label(gr.DownloadButton(t[name], interactive=False, size="sm", min_width=120), label=name)
                                 for name in ("download_png", "download_jpg", "download_mp4", "download_gif", "download_npz")}
                frames = gr.State(None)

        def relabel(lang_now: str) -> list:
            words = TEXT[lang_now]
            return [gr.update(**{prop: [(words[k], v) for k, v in key] if isinstance(key, tuple) else words[key]
                                 for prop, key in texts.items()}) for _, texts in labelled]

        components = {"status": status, "live": live, "compare": compare, "painting": painting, "replay": replay_slider,
                      "replay_image": replay_image, "frames": frames, "timelapse": timelapse, "tabs": tabs, **downloads}
        job = start.click(paint, inputs=[photo, brush, strokes, mode, speed, canvas, fit, seed, lang],
                          outputs=[components[n] for n in OUTPUTS], preprocess=False, show_progress_on=[status],
                          concurrency_limit=1, concurrency_id="painter")
        stop.click(None, cancels=[job])
        replay_slider.input(replay, inputs=[replay_slider, frames], outputs=replay_image, show_progress="hidden")
        lang.change(relabel, inputs=lang, outputs=[c for c, _ in labelled], queue=False, show_progress="hidden")
        demo.load(first_language, inputs=None, outputs=lang, queue=False, show_progress="hidden")
    demo.relabel = relabel  # for the tests
    return demo


def main() -> int:
    disable_power_throttling()
    demo = build_app()
    demo.queue(default_concurrency_limit=1).launch(inbrowser=True, max_file_size=MAX_UPLOAD)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
