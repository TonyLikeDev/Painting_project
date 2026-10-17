"""End-to-end test of the one-command pipeline on many kinds of input (Week 7).

Paints every photograph of the reference's ``test_images/`` (JPGs of 377 to 1021 pixels, none of them square) with
``run_paint.py``, cycling through the four brushes and the three fits, plus generated edge cases: a grey-scale PNG, a PNG with
an alpha channel, a 4:1 panorama, a 1:3 portrait, a 40 x 30 pixel thumbnail and a 16-bit TIFF; and three inputs that must be
refused with a message (a file that is not an image, an unsupported format, an image above the pixel limit). Every case gets a
row in ``experiments/<date>_check_end_to_end/results.csv``: the input's size, the painting's size, strokes, PSNR / SSIM,
times and whether the outcome was the expected one. The paintings are in the folder's git-ignored ``runs/``.

Usage: venv/Scripts/python.exe scripts/check_end_to_end.py [--strokes 500] [--device auto] [--name end_to_end]
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_paint  # noqa: E402
from neural_painter.core.image_io import MAX_PIXELS  # noqa: E402
from neural_painter.pipeline.train_renderer import EXPERIMENT_ROOT, commit_info, new_experiment_dir  # noqa: E402

TEST_IMAGES = ROOT / "stylized-neural-painting" / "test_images"
BRUSHES = ("oil", "watercolor", "marker", "tape")
FITS = ("stretch", "crop", "pad")
FIELDS = ["case", "expect", "outcome", "ok", "brush", "fit", "input_w", "input_h", "output_w", "output_h", "strokes", "psnr", "ssim",
          "optimize_s", "render_s", "wall_s", "message"]


def edge_cases(folder: Path) -> list[tuple[str, Path, str]]:
    """Generated inputs: ``(case, path, expect)`` with expect ``ok`` or ``refused``."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:300, 0:400]
    smooth = np.stack([xx / 399, yy / 299, 0.5 + 0.5 * np.sin(xx / 40)], -1)
    cases = []

    def save(name: str, array: np.ndarray, expect: str = "ok") -> None:
        path = folder / name
        Image.fromarray(array).save(path)  # the mode follows from the array: L, RGB, RGBA, I;16
        cases.append((name, path, expect))

    save("grey.png", (smooth[..., 0] * 255).astype(np.uint8))
    rgba = np.concatenate([(smooth * 255).astype(np.uint8), np.full((300, 400, 1), 128, np.uint8)], -1)
    save("alpha.png", rgba)
    save("panorama_4x1.jpg", (np.tile(smooth, (1, 4, 1))[:200] * 255).astype(np.uint8))
    save("portrait_1x3.jpg", (np.tile(smooth.transpose(1, 0, 2), (3, 1, 1))[:, :200] * 255).astype(np.uint8))
    save("thumbnail_40x30.png", (smooth[::10, ::10] * 255).astype(np.uint8))
    save("deep_16bit.tif", (smooth[..., 1] * 65535).astype(np.uint16))
    (folder / "not_an_image.png").write_bytes(rng.bytes(2000))
    cases.append(("not_an_image.png", folder / "not_an_image.png", "refused"))
    (folder / "animation.gif").write_bytes(b"GIF89a")
    cases.append(("animation.gif", folder / "animation.gif", "refused"))
    side = int(np.sqrt(MAX_PIXELS)) + 8
    Image.new("RGB", (side, side), (90, 120, 150)).save(folder / "too_large.png")
    cases.append(("too_large.png", folder / "too_large.png", "refused"))
    return cases


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--strokes", type=int, default=500)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--name", default="end_to_end")
    args = ap.parse_args(argv)
    out = new_experiment_dir(EXPERIMENT_ROOT, args.name, kind="check")
    photos = [(p.name, p, "ok") for p in sorted(TEST_IMAGES.glob("*.jpg"))]
    cases = photos + edge_cases(out / "runs" / "inputs")
    rows = []
    for i, (case, path, expect) in enumerate(cases):
        brush, fit = BRUSHES[i % len(BRUSHES)], FITS[i % len(FITS)]
        target = out / "runs" / Path(case).stem
        row = {"case": case, "expect": expect, "brush": brush, "fit": fit}
        try:
            with Image.open(path) as im:
                row["input_w"], row["input_h"] = im.size
        except Exception:  # the cases that are not images
            pass
        try:
            run_paint.main([str(path), "--brush", brush, "--fit", fit, "--strokes", str(args.strokes), "--device", args.device,
                            "--out", str(target), "--deterministic", "--quiet"])
            result = next(csv.DictReader(open(target / "results.csv", encoding="utf-8")))
            with Image.open(target / "final.png") as im:
                row["output_w"], row["output_h"] = im.size
            row.update({k: result[k] for k in ("strokes", "psnr", "ssim", "optimize_s", "render_s", "wall_s")}, outcome="ok")
        except SystemExit as exc:
            row.update(outcome="refused", message=str(exc))
        except Exception as exc:  # an unexpected failure is exactly what this check is for: record it and go on
            row.update(outcome="error", message=f"{type(exc).__name__}: {exc}")
        row["ok"] = row["outcome"] == expect
        rows.append(row)
        print(f"{case:24s} {brush:10s} {fit:7s} {row['outcome']:8s} {'as expected' if row['ok'] else 'UNEXPECTED'}"
              + (f"  {row.get('message', '')[:90]}" if row["outcome"] != "ok" else f"  {row['output_w']}x{row['output_h']}, {float(row['psnr']):.2f} dB"), flush=True)
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (out / "config.yaml").write_text(yaml.safe_dump({"created": datetime.now().isoformat(timespec="seconds"), **commit_info(),
                                                     "strokes": args.strokes, "device": args.device, "cases": len(rows)}, sort_keys=False), encoding="utf-8")
    bad = [r["case"] for r in rows if not r["ok"]]
    print(f"{len(rows) - len(bad)} of {len(rows)} cases as expected" + (f"; UNEXPECTED: {bad}" if bad else "") + f"\nwrote {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
