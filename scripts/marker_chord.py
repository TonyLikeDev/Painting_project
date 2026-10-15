"""How long must a new marker-pen stroke be for the renderers to draw it? (Week 6 diagnostic.)

The original initialization puts the three Bezier points of a new marker stroke on one pixel. The rasterizer skips a marker
stroke whose end points are less than 4 pixels apart (L1, at the renderer's resolution), the neural renderers learnt that from
it, and a stroke that renders nothing has no gradient. This script draws straight strokes of a fixed chord length (in block units)
in evenly spread directions around random centres and records, per chord length and renderer:

* ``visible``: the fraction of strokes whose largest opacity exceeds 0.5;
* ``alpha_max``: the mean of the largest opacity;
* ``grad``: the mean L2 norm of the gradient of the summed opacity map with respect to the stroke's shape parameters
  (neural renderers only), i.e. whether the optimizer gets any signal to move or grow the stroke.

Writes ``experiments/<date>_diagnostic_marker_chord/`` with ``results.csv``, ``summary.md`` and ``config.yaml``.

Usage: venv/Scripts/python.exe scripts/marker_chord.py [--chords 0 0.02 0.05 0.075 0.1 0.125 0.15 0.2] [--n 512] [--device auto]
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neural_painter.core.device import device_summary, get_device  # noqa: E402
from neural_painter.core.procedural_rasterizer import ProceduralRasterizer  # noqa: E402
from neural_painter.models.neural_renderer import load_original_checkpoint, original_checkpoint_path  # noqa: E402
from neural_painter.pipeline.train_renderer import EXPERIMENT_ROOT, commit_info, new_experiment_dir  # noqa: E402

FIELDS = ["chord", "renderer", "size", "visible", "alpha_max", "grad"]


def strokes(chord: float, n: int, seed: int = 0) -> np.ndarray:
    """``(n, 12)`` straight marker strokes of length ``chord``: centres in [0.25, 0.75], directions evenly spread over 180 degrees."""
    rng = np.random.default_rng(seed)
    phi = np.pi * (np.arange(n) % 16) / 16
    c = rng.uniform(0.25, 0.75, (n, 2))
    d = 0.5 * chord * np.stack([np.cos(phi), np.sin(phi)], 1)
    p0, p2 = c - d, c + d
    radii = rng.uniform(0.1, 0.25, (n, 2))
    rgb = rng.uniform(0, 1, (n, 3))
    alpha = rng.uniform(0.8, 0.98, (n, 1))
    mid = np.full((n, 2), 0.5)  # control point in the middle of the chord: a straight stroke
    return np.concatenate([p0, mid, p2, radii, rgb, alpha], 1).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--chords", nargs="+", type=float, default=[0, 0.02, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2])
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--name", default="marker_chord")
    args = ap.parse_args(argv)
    device = get_device(args.device)
    nets = {light: load_original_checkpoint(original_checkpoint_path("markerpen", light), "markerpen", light).to(device).eval()
            for light in (True, False)}
    rows = []
    for chord in args.chords:
        p = strokes(chord, args.n)
        for size in (32, 128):
            r = ProceduralRasterizer("markerpen", size, train=True)
            amax = np.array([r.render_stroke(s)[1].max() for s in p])
            rows.append({"chord": chord, "renderer": "rasterizer", "size": size, "visible": float((amax > 0.5).mean()),
                         "alpha_max": float(amax.mean()), "grad": None})
        for light, net in nets.items():
            amax, grad = [], []
            for chunk in np.array_split(p, max(1, len(p) // 64)):  # 64 strokes at a time: the 128 px renderer's graph is large
                x = torch.tensor(chunk, device=device, requires_grad=True)
                _, alpha = net(x)
                grad.append(torch.autograd.grad(alpha.sum(), x)[0][:, :8].norm(dim=1).cpu())
                amax.append(alpha.detach().flatten(1).amax(1).cpu())
            amax, grad = torch.cat(amax).numpy(), torch.cat(grad)
            rows.append({"chord": chord, "renderer": "neural light" if light else "neural full", "size": net.out_size,
                         "visible": float((amax > 0.5).mean()), "alpha_max": float(amax.mean()), "grad": float(grad.mean())})
        print(f"chord {chord:5.3f}: " + "  ".join(f"{r['renderer']} {r['size']}px vis {r['visible']:.2f}" for r in rows[-4:]), flush=True)

    out = new_experiment_dir(EXPERIMENT_ROOT, args.name, kind="diagnostic")
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: ("" if v is None else (f"{v:.6g}" if isinstance(v, float) else v)) for k, v in r.items()} for r in rows)
    lines = ["# Marker-pen strokes of a given chord length: are they drawn?", "",
             f"{args.n} straight strokes per chord length (block units), 16 directions over 180 degrees, centres in [0.25, 0.75]. "
             "visible = share with largest opacity > 0.5; grad = mean norm of d(sum of opacity) / d(shape parameters).", "",
             "| Chord | Renderer | Size | Visible | Mean largest opacity | Gradient |", "| ---: | :--- | ---: | ---: | ---: | ---: |"]
    lines += [f"| {r['chord']:g} | {r['renderer']} | {r['size']} | {r['visible']:.2f} | {r['alpha_max']:.3f} | "
              f"{'-' if r['grad'] is None else format(r['grad'], '.3g')} |" for r in rows]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "config.yaml").write_text(yaml.safe_dump({"created": datetime.now().isoformat(timespec="seconds"), **commit_info(),
                                                     **device_summary(device), "chords": args.chords, "n": args.n}, sort_keys=False),
                                     encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
