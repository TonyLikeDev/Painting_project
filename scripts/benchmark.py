"""Hardware benchmark (Week 7, research question RQ4): the same paintings on one device profile, timed, with their peak memory.

    python scripts/benchmark.py --profile cuda|mps|cpu [--brushes oil watercolor marker tape] [--modes progressive fixed]
        [--renderer original-light] [--strokes 500] [--every 10] [--seeds 0] [--quick] [--max-minutes 0]

For each brush, ``run_ablation.py`` paints the evaluation images (every ``--every``-th of the sorted set: 3 of 30 by default) with
every seed (one by default) in every mode on the profile's device, and writes ``experiments/<date>_benchmark_<profile>_<brush>/``: ``results.csv``
(optimization and rasterization time, peak working memory, optimizer steps, PSNR / SSIM per painting) and ``config.yaml`` (device,
CPU threads, commit). Run it with the same commit on every machine and commit the folders. ``--quick`` paints one image with one
seed per brush and mode, to check a machine in a few minutes. An interrupted benchmark is resumed with
``run_ablation.py --dir <folder>`` and the same arguments (they are recorded in ``config.yaml``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_ablation  # noqa: E402

PROFILES = ("cuda", "mps", "cpu")
PIPELINE = ",lr_schedule=cosine,lr=0.004"  # run_paint.py's defaults (Week 7); the fixed grid keeps the reference's demo.py settings
BRUSHES = ("oil", "watercolor", "marker", "tape")


def ablation_args(args: argparse.Namespace, brush: str) -> list[str]:
    """The ``run_ablation.py`` command line that paints ``brush`` for this benchmark."""
    configs = [f"{mode}=0@{args.max_divide},mode={mode}" + (PIPELINE if mode == "progressive" else "") for mode in args.modes]
    every, seeds = (30, ["0"]) if args.quick else (args.every, [str(s) for s in args.seeds])
    return ["--kind", "benchmark", "--name", f"{args.profile}_{brush}" + ("_quick" if args.quick else ""), "--brush", brush, "--device",
            args.profile, "--renderer", args.renderer, "--strokes", str(args.strokes), "--grid", str(args.max_divide), "--every", str(every),
            "--seeds", *seeds, "--configs", *configs, "--deterministic", "--max-minutes", str(args.max_minutes)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", required=True, choices=PROFILES)
    ap.add_argument("--brushes", nargs="+", default=list(BRUSHES), choices=BRUSHES)
    ap.add_argument("--modes", nargs="+", default=["progressive", "fixed"], choices=["progressive", "fixed", "full"])
    ap.add_argument("--renderer", default="original-light")
    ap.add_argument("--strokes", type=int, default=500)
    ap.add_argument("--max-divide", type=int, default=5)
    ap.add_argument("--every", type=int, default=10, help="every K-th evaluation image (10: 3 of the 30)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0], help="time and memory hardly depend on the seed")
    ap.add_argument("--quick", action="store_true", help="one image, one seed per brush and mode")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="per brush: start no new painting after this many minutes")
    args = ap.parse_args(argv)
    for brush in args.brushes:
        print(f"== {args.profile}: {brush} ==", flush=True)
        run_ablation.main(ablation_args(args, brush))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
