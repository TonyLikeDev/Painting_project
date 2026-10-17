"""MP4 and GIF time-lapse of a finished painting (Week 9).

    venv/Scripts/python.exe scripts/make_timelapse.py SOURCE [--out FOLDER] [--brush oil] [--canvas white] [--size 512]
        [--fps 40] [--stride 1] [--hold 1.5] [--format mp4 gif] [--strip 6]

``SOURCE`` is a ``strokes.npz`` or the folder holding it: a ``run_paint.py`` or web-app result under ``output/``, a run folder of an
experiment, or the original code's ``<name>_strokes.npz`` (name its ``--brush``). The strokes are drawn again from the blank canvas
(``export.video_recorder.write_timelapse``) on the canvas colour the file records (``--canvas`` otherwise; the brush's default
when neither says), in the shape of the ``final.png`` beside it. Without ``--fps`` / ``--stride`` / ``--hold`` the files are those of
``run_paint.py --timelapse``. Writes ``timelapse.mp4`` and ``timelapse.gif`` into ``--out`` (default: the source folder); ``--strip N``
also writes ``timelapse_strip.png``, N frames side by side at evenly spaced stroke counts, for a printed page, and ``--strip 0 9 45``
the frames after those numbers of strokes (``--format`` with no value writes the strip alone).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neural_painter.core.device import disable_power_throttling  # noqa: E402
from neural_painter.export.video_recorder import GIF_SETTINGS, MP4_SETTINGS, frames, write_timelapse  # noqa: E402
from neural_painter.pipeline.painter_engine import load_strokes_npz  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MP4 / GIF time-lapse of a painting's strokes")
    p.add_argument("source", type=Path, help="strokes.npz or a folder holding one")
    p.add_argument("--out", type=Path, default=None, help="folder for timelapse.mp4 / .gif (default: the source's)")
    p.add_argument("--brush", default=None, help="the brush, for an original .npz that does not record it")
    p.add_argument("--canvas", default=None, help="black | white | #rrggbb, when the file does not record it")
    p.add_argument("--size", type=int, default=512, help="side the strokes are drawn at")
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--stride", type=int, default=None, help="one frame every this many strokes")
    p.add_argument("--hold", type=float, default=None, help="seconds the finished painting stays at the end")
    p.add_argument("--format", nargs="*", default=["mp4", "gif"], choices=["mp4", "gif"])
    p.add_argument("--strip", type=int, nargs="+", default=[], help="N frames side by side (timelapse_strip.png), or the stroke counts")
    args = p.parse_args(argv)
    disable_power_throttling()
    npz = args.source / "strokes.npz" if args.source.is_dir() else args.source
    if not npz.is_file():
        raise SystemExit(f"no stroke file at {npz}")
    strokes, spec, extra = load_strokes_npz(npz, args.brush)
    canvas = str(extra.get("canvas", args.canvas or spec.default_canvas_color))
    final = npz.parent / "final.png"
    shape = Image.open(final).size if final.is_file() else None
    out = args.out or npz.parent
    for kind in args.format:
        settings = MP4_SETTINGS if kind == "mp4" else GIF_SETTINGS
        path = out / f"timelapse.{kind}"
        n = write_timelapse(path, strokes, spec, size=args.size, canvas_color=canvas, shape=shape, fps=args.fps or settings["fps"],
                            stride=args.stride or max(1, math.ceil(len(strokes) / settings["frames"])),
                            hold=settings["hold"] if args.hold is None else args.hold)
        print(f"wrote {path} ({n} frames, {path.stat().st_size / 1e6:.2f} MB)")
    if args.strip:
        counts = np.linspace(0, len(strokes), args.strip[0]) if len(args.strip) == 1 else np.clip(args.strip, 0, len(strokes))
        picks = sorted(set(np.asarray(counts).round().astype(int).tolist()))
        shots = [f for k, f in enumerate(frames(strokes, spec, args.size, canvas, shape)) if k in picks]  # frame k: k strokes drawn
        gap = np.full((shots[0].shape[0], 8, 3), 255, np.uint8)
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.concatenate([x for shot in shots for x in (shot, gap)][:-1], axis=1)).save(out / "timelapse_strip.png")
        print(f"wrote {out / 'timelapse_strip.png'} (after {', '.join(map(str, picks))} strokes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
