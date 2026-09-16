"""Hardware benchmark runner (Week 7).

Planned usage::

    venv/Scripts/python.exe scripts/benchmark.py --profile cuda --set data/eval_set --strokes 500 \
        --brush oil --seeds 0 1 2 --out experiments/<date>_benchmark_cuda/

Runs ``run_paint`` on every frozen image, records wall time (with device
synchronization), peak memory (``core.device.peak_memory_gb``), PSNR / SSIM / LPIPS,
and writes ``results.csv`` plus the commit hash and ``device_summary()``.
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    print("benchmark.py is scheduled for Week 7; see ROADMAP.md", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
