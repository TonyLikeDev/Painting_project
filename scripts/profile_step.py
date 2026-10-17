"""Where a painting's time goes, and what the Week 7 speed options do (report Section 4.4.7).

Two measurements, written to ``experiments/<date>_profile_<name>/`` (``results.csv``, ``summary.md``, ``config.yaml``):

* ``variants`` (default device ``cuda``): one fixed-grid painting of ``--image`` (500 strokes, light oil renderer, seed 0) timed
  ``--reps`` times as it is, with the strokes composited in one closed-form product instead of one at a time
  (``C = sum_i a_i f_i prod_{j>i} (1 - a_j) + C_0 prod_j (1 - a_j)``), and under autocast to float16 and to bfloat16 (``cuda``
  only); then one painting under the PyTorch profiler, which counts the kernel launches and the CPU and GPU time.
* ``paintings``: one fixed-grid and one progressive painting (the engine's defaults), twice each, on ``--device`` in a fresh process
  with Windows' power throttling forced off and in one with it forced on: the optimization and the rasterization time.
* ``cpu-steps``: the CPU time of one forward and backward pass of the light renderer for 128, 256, 384 and 512 strokes, then 128
  and 256 again, in a fresh process with Windows' power throttling forced off and in one with it forced on
  (``core.device.set_power_throttling``; on other systems both runs are the same).

Every number of the speed paragraphs of Section 4.4.7 comes from here. Run it alone: anything else on the machine changes it.

Usage: venv/Scripts/python.exe scripts/profile_step.py [--name step] [--device cuda] [--reps 2] [--image apple]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neural_painter.core.device import device_summary, get_device, set_deterministic, set_power_throttling  # noqa: E402
from neural_painter.core.image_io import load_image  # noqa: E402
from neural_painter.models.neural_renderer import load_original_checkpoint, original_checkpoint_path  # noqa: E402
from neural_painter.pipeline import metrics, painter_engine  # noqa: E402
from neural_painter.pipeline.train_renderer import EXPERIMENT_ROOT, commit_info, new_experiment_dir  # noqa: E402

SIZES = (128, 256, 384, 512, 128, 256)


def closed_form(canvas: torch.Tensor, foregrounds: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
    """``composite_sequence`` with one reversed cumulative product instead of a loop over the strokes."""
    keep = 1.0 - alphas  # (B, N, C, H, W)
    after = torch.flip(torch.cumprod(torch.flip(keep, [1]), 1), [1])  # prod over j >= i
    after = torch.cat([after[:, 1:], torch.ones_like(after[:, :1])], 1)  # prod over j > i
    return (alphas * foregrounds * after).sum(1) + canvas * after[:, 0] * keep[:, 0]


def renderer():
    return load_original_checkpoint(original_checkpoint_path("oilpaintbrush", True), "oilpaintbrush", True)


def variants(device: torch.device, image: np.ndarray, reps: int) -> list[dict]:
    model, rows = renderer(), []
    options = [("loop composite (as built)", None, "float32"), ("closed-form composite", closed_form, "float32")]
    if device.type == "cuda":  # PainterConfig.precision: the engine runs the renderer under autocast
        options += [("autocast float16", None, "float16"), ("autocast bfloat16", None, "bfloat16")]
    loop = painter_engine.composite_sequence
    for name, composite, precision in options:
        painter_engine.composite_sequence = composite or loop
        try:
            for rep in range(reps):
                cfg = painter_engine.PainterConfig(strokes_per_block=20, seed=0, precision=precision)
                paint = painter_engine.paint_fixed_grid(image, model, cfg, grid=5, device=device)
                rows.append({"kind": "variant", "name": name, "rep": rep, "optimize_s": paint.result.seconds,
                             "psnr": metrics.psnr(paint.image, image)})
                print(f"{name:28s} rep {rep}: {paint.result.seconds:.2f} s, psnr {rows[-1]['psnr']:.2f}", flush=True)
        except RuntimeError as exc:  # an option the backend cannot run is recorded, not fatal
            rows.append({"kind": "variant", "name": name, "rep": -1, "error": str(exc).splitlines()[0]})
            print(f"{name}: failed ({rows[-1]['error']})", flush=True)
        finally:
            painter_engine.composite_sequence = loop
    activities = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if device.type == "cuda" else [])
    with torch.profiler.profile(activities=activities) as prof:
        t0 = time.perf_counter()
        painter_engine.paint_fixed_grid(image, model, painter_engine.PainterConfig(strokes_per_block=20, seed=0), grid=5, device=device)
        wall = time.perf_counter() - t0
    events = prof.events()  # the raw events: summing key_averages() would count a kernel's time for its operator too
    launches = sum(1 for e in events if e.name in ("cudaLaunchKernel", "cudaLaunchKernelExC"))
    gpu = sum(e.device_time for e in events if e.device_type == torch.autograd.DeviceType.CUDA) / 1e6
    rows.append({"kind": "profile", "name": "loop composite (as built)", "kernel_launches": launches, "wall_s": wall, "gpu_s": gpu})
    print(f"profile: {launches} kernel launches, {wall:.2f} s under the profiler, GPU kernels busy {gpu:.2f} s", flush=True)
    return rows


def cpu_steps(throttled: bool) -> list[dict]:
    """Run in a fresh process: the throttling state, once a large step has run, persists."""
    set_power_throttling(throttled)
    model, rows = renderer(), []
    for i, n in enumerate(SIZES):
        x = torch.rand(n, model.spec.dim, requires_grad=True)
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            fg, alpha = model(x)
            (fg.mean() + alpha.mean()).backward()
            best = min(best, time.perf_counter() - t0)
        rows.append({"kind": "cpu-step", "name": "throttled" if throttled else "not throttled", "order": i, "strokes": n, "seconds": best})
    return rows


def paint_times(throttled: bool, device_name: str, image_name: str, reps: int = 2) -> list[dict]:
    """Run in a fresh process: one fixed-grid and one progressive painting, ``reps`` times each, with throttling on or off."""
    from neural_painter.pipeline.progressive_painter import paint_image

    set_power_throttling(throttled)
    set_deterministic(True)
    device, model = get_device(device_name), renderer()
    image = load_image(ROOT / "data" / "eval_set" / f"{image_name}.png")
    rows = []
    for rep in range(reps):
        for mode in ("fixed", "progressive"):
            paint = paint_image(image, model, painter_engine.PainterConfig(seed=0), mode=mode, device=device)
            rows.append({"kind": "painting", "name": f"{mode}, {'throttled' if throttled else 'not throttled'}", "rep": rep,
                         "optimize_s": paint.optimize_seconds, "seconds": paint.render_seconds, "psnr": metrics.psnr(paint.image, image)})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="step")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--image", default="apple")
    ap.add_argument("--cpu-steps-child", choices=["on", "off"], help=argparse.SUPPRESS)
    ap.add_argument("--paint-child", choices=["on", "off"], help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.cpu_steps_child:  # the subprocesses: print their rows as JSON
        print(json.dumps(cpu_steps(args.cpu_steps_child == "on")))
        return 0
    if args.paint_child:
        print(json.dumps(paint_times(args.paint_child == "on", args.device, args.image)))
        return 0

    set_power_throttling(False)
    set_deterministic(True)
    device = get_device(args.device)
    image = load_image(ROOT / "data" / "eval_set" / f"{args.image}.png")
    rows = variants(device, image, args.reps)
    for state in ("off", "on"):
        out = subprocess.run([sys.executable, __file__, "--cpu-steps-child", state], capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        rows += json.loads(out.stdout.strip().splitlines()[-1])
        print(f"cpu steps, throttling {state}: " + ", ".join(f"{r['strokes']}: {r['seconds']:.3f} s" for r in rows if r['kind'] == 'cpu-step' and r['name'] == ('throttled' if state == 'on' else 'not throttled')), flush=True)
    for state in ("off", "on"):  # whole paintings on the device, in fresh processes
        out = subprocess.run([sys.executable, __file__, "--paint-child", state, "--device", args.device, "--image", args.image], capture_output=True, text=True)
        new = json.loads(out.stdout.strip().splitlines()[-1])
        rows += new
        print(f"paintings, throttling {state}: " + ", ".join(f"{r['name'].split(',')[0]} {r['optimize_s']:.1f} + {r['seconds']:.1f} s" for r in new), flush=True)

    folder = new_experiment_dir(EXPERIMENT_ROOT, args.name, kind="profile")
    order = ["kind", "name", "rep", "order", "strokes", "optimize_s", "psnr", "seconds", "kernel_launches", "wall_s", "gpu_s", "error"]
    fields = [k for k in order if any(k in r for r in rows)]
    with open(folder / "results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"# Painting step profile ({device})", "", "| Variant | Optimize (s), mean of reps | PSNR (dB) |", "| :--- | ---: | ---: |"]
    for name in dict.fromkeys(r["name"] for r in rows if r["kind"] == "variant"):
        mine = [r for r in rows if r["kind"] == "variant" and r["name"] == name and r.get("rep", -1) >= 0]
        lines.append(f"| {name} | {np.mean([r['optimize_s'] for r in mine]):.2f} | {np.mean([r['psnr'] for r in mine]):.2f} |" if mine else f"| {name} | failed | |")
    p = next(r for r in rows if r["kind"] == "profile")
    lines += ["", f"Profile of one painting: {p['kernel_launches']} kernel launches; {p['wall_s']:.2f} s under the profiler, of which the GPU "
              f"ran kernels for {p['gpu_s']:.2f} s.", "",
              "| CPU step, strokes | " + " | ".join(str(n) for n in SIZES) + " |", "| :--- | " + " | ".join("---:" for _ in SIZES) + " |"]
    for name in ("not throttled", "throttled"):
        lines.append(f"| {name} | " + " | ".join(f"{r['seconds']:.3f}" for r in rows if r["kind"] == "cpu-step" and r["name"] == name) + " |")
    lines += ["", f"| Painting on {device} | Optimize (s), mean | Rasterize (s), mean | PSNR (dB) |", "| :--- | ---: | ---: | ---: |"]
    for name in dict.fromkeys(r["name"] for r in rows if r["kind"] == "painting"):
        mine = [r for r in rows if r["kind"] == "painting" and r["name"] == name]
        lines.append(f"| {name} | {np.mean([r['optimize_s'] for r in mine]):.2f} | {np.mean([r['seconds'] for r in mine]):.2f} | {mine[0]['psnr']:.2f} |")
    (folder / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (folder / "config.yaml").write_text(yaml.safe_dump({"created": datetime.now().isoformat(timespec="seconds"), **commit_info(),
                                                        **device_summary(device), "image": args.image, "reps": args.reps}, sort_keys=False), encoding="utf-8")
    print("\n".join(lines) + f"\nwrote {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
