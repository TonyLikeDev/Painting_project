"""One-command check that this machine can run the project, and how fast (the Friday MacBook check).

Runs every step even when an earlier one fails, so one problem does not hide the others:

1. environment: Python, OS, torch, available backends, git commit, and whether the reference checkout / checkpoints exist;
2. dependencies: each package from ``pyproject.toml`` is imported and its version recorded (plus the ffmpeg binary);
3. test suite: ``pytest`` in a subprocess, with failures, skips and the reason for each skip;
4. renderers: forward and forward + backward time and peak memory of the pretrained renderers on this device and on the CPU;
5. training smoke test: two short epochs of the light oil renderer, to prove that training runs on this backend;
6. painting smoke test: one evaluation image painted with 100 oil strokes on a 5 x 5 grid, with the pixel loss and with the
   Sinkhorn term added, to prove that the painting engine and the transport loss run on this backend (and how long they take).

Writes ``experiments/<date>_check_<os>_<arch>_<device>/`` with ``result.json`` and ``report.md``; commit and push that folder
so the numbers reach the repository. Exit code 0 when no step failed.

Usage: python scripts/device_check.py [--device auto] [--skip-tests] [--skip-training] [--skip-painting] [--with-lpips]
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
import platform
import re
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# (distribution name, import name, required?)
DEPENDENCIES = [
    ("torch", "torch", True), ("torchvision", "torchvision", True), ("numpy", "numpy", True), ("scipy", "scipy", True),
    ("opencv-python", "cv2", True), ("pillow", "PIL", True), ("pyyaml", "yaml", True), ("einops", "einops", True),
    ("matplotlib", "matplotlib", True), ("scikit-image", "skimage", True), ("torchmetrics", "torchmetrics", True),
    ("lpips", "lpips", True), ("imageio", "imageio", True), ("imageio-ffmpeg", "imageio_ffmpeg", True), ("psutil", "psutil", True),
    ("pytest", "pytest", False), ("gradio", "gradio", False),
]


def step(results: dict, name: str, fn):
    """Run ``fn``; store its return value, or the error, under ``results[name]``."""
    t0 = time.perf_counter()
    try:
        results[name] = {"ok": True, "seconds": None, **(fn() or {})}
    except Exception as exc:  # the point of this script is to report, not to stop
        results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=6)}
    results[name]["seconds"] = round(time.perf_counter() - t0, 1)
    print(f"[{'ok' if results[name]['ok'] else 'FAILED'}] {name} ({results[name]['seconds']} s)" + ("" if results[name]["ok"] else f": {results[name]['error']}"), flush=True)


def check_environment(device_name: str) -> dict:
    import torch

    from neural_painter.core.device import device_summary, get_device
    from neural_painter.models.neural_renderer import original_checkpoint_path
    from neural_painter.pipeline.train_renderer import commit_info

    device = get_device(device_name)
    mps = getattr(torch.backends, "mps", None)
    checkpoints = {f"{b}{'_light' if l else ''}": original_checkpoint_path(b, l).is_file()
                   for b in ("oilpaintbrush", "watercolor", "markerpen", "rectangle") for l in (False, True)}
    return {
        "device": str(device), "summary": device_summary(device), **commit_info(), "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(mps is not None and mps.is_available()), "cpu_threads": torch.get_num_threads(),
        "reference_checkout": (ROOT / "stylized-neural-painting" / "renderer.py").is_file(), "pretrained_checkpoints": checkpoints,
    }


def check_dependencies() -> dict:
    found, missing = {}, []
    for dist, module, required in DEPENDENCIES:
        try:
            mod = importlib.import_module(module)
            try:
                version = metadata.version(dist)
            except metadata.PackageNotFoundError:
                version = getattr(mod, "__version__", "?")
            found[dist] = str(version)
        except Exception as exc:
            found[dist] = f"MISSING ({type(exc).__name__}: {exc})"
            if required:
                missing.append(dist)
    out: dict = {"versions": found, "missing_required": missing, "ok": not missing}
    if missing:
        out["error"] = "required packages missing or not importable: " + ", ".join(missing)
    try:
        import imageio_ffmpeg

        out["ffmpeg"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        out["ffmpeg"] = f"unavailable ({type(exc).__name__}: {exc})"
    return out


def parse_pytest_output(text: str) -> tuple[dict[str, int], list[str]]:
    """Counts from pytest's last line (``81 passed, 3 skipped in 45.1s``) and the reason of every ``-rs`` skip."""
    last = next((l for l in reversed(text.splitlines()) if l.strip()), "")
    counts = {k: int(n) for n, k in re.findall(r"(\d+) (passed|failed|skipped|errors?|xfailed|xpassed)", last)}
    skips = re.findall(r"^SKIPPED \[\d+\] (.+)$", text, flags=re.MULTILINE)
    return counts, sorted(set(skips))


def check_tests() -> dict:
    proc = subprocess.run(
        # pyproject.toml already adds -q, and a second -q suppresses the summary line: clear addopts, then ask for one -q
        [sys.executable, "-m", "pytest", "-o", "addopts=", "-q", "-p", "no:cacheprovider", "-rs", "-W", "ignore"], cwd=ROOT,
        capture_output=True, text=True,
    )
    text = proc.stdout + proc.stderr
    counts, skips = parse_pytest_output(text)
    out = {"ok": proc.returncode == 0, "returncode": proc.returncode, "counts": counts, "skipped_reasons": skips,
           "tail": "\n".join(text.splitlines()[-15:])}
    if proc.returncode != 0:
        out["error"] = f"pytest exited with {proc.returncode}: {counts}"
    return out


def check_renderers(device_name: str) -> dict:
    import torch

    from neural_painter.core.device import get_device
    from neural_painter.models.neural_renderer import benchmark_renderer, load_original_checkpoint, original_checkpoint_path

    rows = []
    for device in dict.fromkeys([get_device(device_name), torch.device("cpu")]):
        for brush, light in (("oilpaintbrush", True), ("oilpaintbrush", False)):
            path = original_checkpoint_path(brush, light)
            if not path.is_file():
                rows.append({"device": str(device), "renderer": f"{brush} {'light' if light else 'full'}", "skipped": "checkpoint not downloaded"})
                continue
            model = load_original_checkpoint(path, brush, light)
            rows.append({"device": str(device), "renderer": f"{brush} {'light' if light else 'full'}", **benchmark_renderer(model, device)})
    return {"rows": rows}


def check_training(device_name: str) -> dict:
    from neural_painter.pipeline.train_renderer import TrainConfig, train

    with tempfile.TemporaryDirectory() as tmp:
        cfg = TrainConfig(brush="oilpaintbrush", light=True, epochs=2, samples_per_epoch=3200, batch_size=64, threads=4, device=device_name,
                          val_size=200, vis_every=100, lr_step=100, checkpoint_dir=f"{tmp}/ckpt", experiments_dir=f"{tmp}/exp")
        lines: list[str] = []
        result = train(cfg, log=lines.append)
    epoch2 = next(l for l in lines if l.startswith("epoch   2/2"))
    return {"device": result["device"], "psnr_mean_after_2_short_epochs": result["last"]["psnr_mean"], "epoch2_line": epoch2}


def check_painting(device_name: str, strokes_per_block: int = 4, iters_per_stroke: int = 0) -> dict:
    """Paint ``apple`` of the evaluation set on a 5 x 5 grid (100 strokes by default) with and without the Sinkhorn term."""
    from neural_painter.core.device import get_device
    from neural_painter.core.image_io import load_image
    from neural_painter.models.neural_renderer import load_original_checkpoint, original_checkpoint_path
    from neural_painter.pipeline import metrics
    from neural_painter.pipeline.painter_engine import PainterConfig, paint_fixed_grid

    checkpoint, image_path = original_checkpoint_path("oilpaintbrush", True), ROOT / "data" / "eval_set" / "apple.png"
    if not checkpoint.is_file() or not image_path.is_file():
        return {"skipped": "the pretrained light oil renderer or data/eval_set/apple.png is missing"}
    device = get_device(device_name)
    renderer, image = load_original_checkpoint(checkpoint, "oilpaintbrush", True), load_image(image_path)
    rows = []
    for name, beta in (("pixel", 0.0), ("pixel + Sinkhorn", 0.1)):
        cfg = PainterConfig(strokes_per_block=strokes_per_block, iters_per_stroke=iters_per_stroke, beta_ot=beta, seed=0)
        paint = paint_fixed_grid(image, renderer, cfg, grid=5, device=device)
        m = metrics.image_metrics(paint.image, image)
        rows.append({"loss": name, "device": str(device), "strokes": len(paint.strokes), "steps": len(paint.result.loss),
                     "optimize_s": round(paint.result.seconds, 1), "render_s": round(paint.render_seconds, 1),
                     "psnr": round(m["psnr"], 2), "ssim": round(m["ssim"], 3)})
        if not (m["psnr"] > 0 and all(v == v for v in paint.result.loss)):  # a finite loss and a picture that is not blank noise
            raise RuntimeError(f"the painting with {name} is degenerate: {rows[-1]}")
    return {"rows": rows}


def check_lpips(device_name: str) -> dict:
    import torch

    import lpips
    from neural_painter.core.device import get_device

    device = get_device(device_name)
    model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()  # downloads the AlexNet weights on first use
    x = torch.rand(2, 3, 64, 64, device=device) * 2 - 1
    with torch.no_grad():
        value = float(model(x, x.flip(0)).mean())
    return {"device": str(device), "lpips_between_two_random_images": value}


def report_markdown(results: dict) -> str:
    env = results.get("environment", {})
    lines = ["# Device check", "", f"Run on {results['created']} (commit `{env.get('commit', '?')}`, working tree dirty: {env.get('dirty')}).", ""]
    summary = env.get("summary", {})
    lines += [f"- device: `{env.get('device')}`; Python {summary.get('python')}; torch {summary.get('torch')}; {summary.get('platform')}",
              f"- backends: cuda {env.get('cuda_available')}, mps {env.get('mps_available')}; CPU threads {env.get('cpu_threads')}",
              f"- reference checkout present: {env.get('reference_checkout')}; pretrained checkpoints: "
              + ", ".join(k for k, v in env.get("pretrained_checkpoints", {}).items() if v) , ""]
    lines += ["| Step | Result | Seconds |", "| :--- | :--- | ---: |"]
    for name, r in results.items():
        if isinstance(r, dict) and "ok" in r:
            lines.append(f"| {name} | {'ok' if r['ok'] else 'FAILED: ' + str(r.get('error', ''))[:120].splitlines()[0]} | {r['seconds']} |")
    deps = results.get("dependencies", {}).get("versions")
    if deps:
        lines += ["", "## Dependencies", "", ", ".join(f"{k} {v}" for k, v in deps.items()), "", f"ffmpeg: `{results['dependencies'].get('ffmpeg')}`"]
    tests = results.get("tests", {})
    if tests.get("counts") is not None:
        lines += ["", "## Tests", "", f"{tests['counts']}"]
        if tests.get("skipped_reasons"):
            lines += ["", "Skipped:"] + [f"- {s}" for s in tests["skipped_reasons"]]
    rend = results.get("renderers", {}).get("rows")
    if rend:
        lines += ["", "## Renderer speed (64 strokes per call)", "", "| Device | Renderer | Forward (ms) | Forward + backward (ms) | Peak memory (GB) |", "| :--- | :--- | ---: | ---: | ---: |"]
        for r in rend:
            if "skipped" in r:
                lines.append(f"| {r['device']} | {r['renderer']} | - | - | {r['skipped']} |")
            else:
                mem = "-" if r.get("peak_memory_gb") is None else f"{r['peak_memory_gb']:.2f}"
                lines.append(f"| {r['device']} | {r['renderer']} | {r['forward_ms']:.1f} | {r['forward_backward_ms']:.1f} | {mem} |")
    train = results.get("training", {})
    if train.get("ok"):
        lines += ["", "## Training smoke test", "", f"`{train['epoch2_line']}`"]
    painting = results.get("painting", {})
    if painting.get("rows"):
        lines += ["", "## Painting smoke test (apple, 5 x 5 grid)", "", "| Loss | Device | Strokes | Steps | Optimize (s) | Final 512 px render (s) | PSNR (dB) | SSIM |",
                  "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        lines += [f"| {r['loss']} | {r['device']} | {r['strokes']} | {r['steps']} | {r['optimize_s']} | {r['render_s']} | {r['psnr']} | {r['ssim']} |"
                  for r in painting["rows"]]
    elif painting.get("skipped"):
        lines += ["", "## Painting smoke test", "", f"Skipped: {painting['skipped']}."]
    for name, r in results.items():
        if isinstance(r, dict) and r.get("ok") is False:
            detail = r.get("traceback") or "\n".join(filter(None, [r.get("error"), r.get("tail")]))
            lines += ["", f"## Failure in `{name}`", "", "```", detail.strip(), "```"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--skip-painting", action="store_true")
    ap.add_argument("--with-lpips", action="store_true", help="also build the LPIPS metric (downloads the AlexNet weights once)")
    args = ap.parse_args(argv)

    results: dict = {"created": datetime.now().isoformat(timespec="seconds")}
    step(results, "environment", lambda: check_environment(args.device))
    step(results, "dependencies", check_dependencies)
    if not args.skip_tests:
        step(results, "tests", check_tests)
    step(results, "renderers", lambda: check_renderers(args.device))
    if not args.skip_training:
        step(results, "training", lambda: check_training(args.device))
    if not args.skip_painting:
        step(results, "painting", lambda: check_painting(args.device))
    if args.with_lpips:
        step(results, "lpips", lambda: check_lpips(args.device))

    from neural_painter.pipeline.train_renderer import EXPERIMENT_ROOT, new_experiment_dir

    device = results.get("environment", {}).get("device", args.device).split(":")[0]
    out = new_experiment_dir(EXPERIMENT_ROOT, f"{platform.system().lower()}_{platform.machine().lower()}_{device}", kind="check")
    (out / "result.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    (out / "report.md").write_text(report_markdown(results), encoding="utf-8")
    failed = [n for n, r in results.items() if isinstance(r, dict) and r.get("ok") is False]
    print(f"\nwrote {out}\n" + ("ALL STEPS OK" if not failed else "FAILED STEPS: " + ", ".join(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
