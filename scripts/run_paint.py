"""One command from photo to painting (Week 7).

Planned usage::

    venv/Scripts/python.exe scripts/run_paint.py IMAGE --brush oil --strokes 500 --grid progressive \
        --max-divide 5 --canvas white --device auto --seed 0 --out experiments/<date>_<name>/

Prints strokes done, current loss and elapsed time while running; writes the final
PNG, the stroke ``.npz``, ``config.yaml`` and ``results.csv`` (RESEARCH_PLAN 3.6).
"""
from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paint an image with the neural painter (Week 7)")
    p.add_argument("image")
    p.add_argument("--brush", default="oil", help="oil | watercolor | marker | tape")
    p.add_argument("--strokes", type=int, default=500)
    p.add_argument("--grid", default="progressive", choices=["full", "fixed", "progressive"])
    p.add_argument("--max-divide", type=int, default=5)
    p.add_argument("--canvas", default=None, help="black | white | #rrggbb (default: brush default)")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    print("run_paint.py is scheduled for Week 7; see ROADMAP.md", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
