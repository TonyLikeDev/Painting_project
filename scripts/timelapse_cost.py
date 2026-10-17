"""What the time-lapse costs (Week 9): seconds and megabytes of the MP4 and the GIF of finished paintings, per brush.

    venv/Scripts/python.exe scripts/timelapse_cost.py "Oil=experiments/<folder>#prog_cos4" "Tape=experiments/<folder>#prog_cos4" ...

Each argument names an experiment folder of ``run_ablation.py`` and one of its configs; the ``strokes.npz`` of seed 0 of every image
in it is turned into the app's ``timelapse.mp4`` and ``timelapse.gif`` (``export.video_recorder.write_timelapses``, 512 pixels,
the brush's canvas) in a temporary folder. Writes ``results.csv`` (one row per painting), ``summary.md`` (means per brush) and
``config.yaml`` into ``experiments/<date>_timelapse_cost``. Run it alone: it is a timing.
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neural_painter.core.device import device_summary, disable_power_throttling, get_device  # noqa: E402
from neural_painter.export.video_recorder import GIF_SETTINGS, MP4_SETTINGS, write_timelapse  # noqa: E402
from neural_painter.pipeline.painter_engine import load_strokes_npz  # noqa: E402
from neural_painter.pipeline.train_renderer import commit_info, new_experiment_dir  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("sources", nargs="+", help='"Label=experiments/<folder>#<config>"')
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default="timelapse_cost")
    args = p.parse_args(argv)
    disable_power_throttling()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for source in args.sources:
            label, rest = source.split("=", 1)
            folder, config = rest.split("#", 1)
            for npz in sorted(Path(folder).glob(f"runs/*/{config}_s{args.seed}/strokes.npz")):
                strokes, spec, _ = load_strokes_npz(npz)
                row = {"brush": label, "image": npz.parent.parent.name, "strokes": len(strokes)}
                for kind, settings in (("mp4", MP4_SETTINGS), ("gif", GIF_SETTINGS)):
                    path = Path(tmp) / f"t.{kind}"
                    t0 = time.perf_counter()
                    frames = write_timelapse(path, strokes, spec, fps=settings["fps"], hold=settings["hold"],
                                             stride=max(1, -(-len(strokes) // settings["frames"])))
                    row |= {f"{kind}_s": time.perf_counter() - t0, f"{kind}_mb": path.stat().st_size / 1e6, f"{kind}_frames": frames}
                rows.append(row)
                print(f"{label} {row['image'][:12]}: mp4 {row['mp4_s']:.2f} s {row['mp4_mb']:.2f} MB, gif {row['gif_s']:.2f} s "
                      f"{row['gif_mb']:.2f} MB", flush=True)
    out = new_experiment_dir(ROOT / "experiments", args.name, kind="measure")
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: f"{v:.4g}" if isinstance(v, float) else v for k, v in row.items()} for row in rows)
    lines = ["| Brush | Paintings | Strokes | MP4 s | MP4 MB | GIF s | GIF MB |", "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label in dict.fromkeys(r["brush"] for r in rows):
        group = [r for r in rows if r["brush"] == label]
        mean = {k: sum(r[k] for r in group) / len(group) for k in ("strokes", "mp4_s", "mp4_mb", "gif_s", "gif_mb")}
        lines.append(f"| {label} | {len(group)} | {mean['strokes']:.0f} | {mean['mp4_s']:.2f} | {mean['mp4_mb']:.2f} | {mean['gif_s']:.2f} | "
                     f"{mean['gif_mb']:.2f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    info = {"created": datetime.now().isoformat(timespec="seconds"), **commit_info(), **device_summary(get_device("cpu")), "sources": args.sources,
            "seed": args.seed, "size": 512, "mp4": MP4_SETTINGS, "gif": GIF_SETTINGS}
    (out / "config.yaml").write_text(yaml.safe_dump(info, sort_keys=False), encoding="utf-8")
    print("\n".join(lines) + f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
