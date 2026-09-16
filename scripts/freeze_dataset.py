"""Freeze the evaluation image set (Week 3, RESEARCH_PLAN.md section 3.5).

Every source image is centre-cropped to a square and resized to ``SIZE x SIZE``
with the same ``INTER_AREA`` resampling the painter uses, then written as a
lossless PNG to ``data/eval_set/``. A manifest with SHA-256 hashes is written next
to the images and rendered into ``experiments/dataset.md``. Once frozen the set
must not change; rerunning without ``--force`` refuses to overwrite an existing
manifest.

Sources: the 11 images shipped with the original repository
(``stylized-neural-painting/test_images``) plus any photo dropped into
``data/raw_photos/`` (self-collected set; add 10 to 20 covering portraits,
landscapes, still life and high-texture scenes, then rerun with ``--force``
*before* Week 5 experiments start).

Usage: ``venv/Scripts/python.exe scripts/freeze_dataset.py [--size 512] [--force]``
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neural_painter.core import image_io  # noqa: E402

REPO_IMAGES = ROOT / "stylized-neural-painting" / "test_images"
RAW_PHOTOS = ROOT / "data" / "raw_photos"
EVAL_SET = ROOT / "data" / "eval_set"
MANIFEST = EVAL_SET / "manifest.csv"
DATASET_MD = ROOT / "experiments" / "dataset.md"

# short content tags for the report table (repo images only; user photos get "user photo")
TAGS = {
    "alien": "toy figure, saturated colours, studio background",
    "apple": "still life, single object on white",
    "cube2": "Rubik's cube, flat colour regions, sharp edges",
    "diamond": "gem on dark background, specular highlights",
    "diamond2": "gem, second view, dark background",
    "fire": "flames, high-frequency texture, dark background",
    "iceland": "landscape, sky and water, low-frequency regions",
    "jay": "bird, fine feather texture",
    "joker": "face / portrait, painted make-up",
    "sunflowers": "painting (van Gogh), dense texture",
    "yosemite": "landscape, rock and trees",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--force", action="store_true", help="overwrite an existing frozen set")
    args = ap.parse_args(argv)

    if MANIFEST.exists() and not args.force:
        print(f"{MANIFEST} exists; the set is frozen. Use --force only to add the self-collected photos.", file=sys.stderr)
        return 1
    EVAL_SET.mkdir(parents=True, exist_ok=True)

    sources = [(p, "original repo test_images") for p in image_io.list_sample_images(REPO_IMAGES)]
    sources += [(p, "self-collected") for p in image_io.list_sample_images(RAW_PHOTOS)]
    if not sources:
        print("no source images found", file=sys.stderr)
        return 1

    rows = []
    for src, origin in sources:
        img = image_io.load_image(src)
        h, w = img.shape[:2]
        fitted, box = image_io.fit(img, args.size, "crop")
        out = image_io.save_image(EVAL_SET / f"{src.stem}.png", fitted)
        rows.append(
            {
                "name": src.stem,
                "file": out.name,
                "origin": origin,
                "source_file": src.name,
                "source_width": w,
                "source_height": h,
                "crop_box": " ".join(str(v) for v in box),
                "size": args.size,
                "sha256": sha256(out),
                "tags": TAGS.get(src.stem, "user photo"),
            }
        )
        print(f"{src.name:18s} {w}x{h} -> crop {box} -> {out.name}")

    with open(MANIFEST, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Evaluation image set (frozen)",
        "",
        f"Frozen on {date.today().isoformat()} at commit `{commit_hash()}` by `scripts/freeze_dataset.py`.",
        f"Resolution: {args.size} x {args.size}, centre crop to square then `INTER_AREA` resize, lossless PNG in `data/eval_set/`.",
        "The manifest with SHA-256 hashes is `data/eval_set/manifest.csv`. **Do not modify these files**; every table in the",
        "report is computed on exactly this set (RESEARCH_PLAN.md section 3.5).",
        "",
        "Rules: the painter receives these square PNGs directly, so the original code's stretch-to-square resize is the",
        "identity and both implementations see identical pixels. Metrics (PSNR, SSIM, LPIPS) are computed at",
        f"{args.size} x {args.size} against these files.",
        "",
        f"| # | Name | Origin | Source size (w x h) | Crop box (l, t, r, b) | Content | SHA-256 (first 12) |",
        "| ---: | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | `{r['file']}` | {r['origin']} | {r['source_width']} x {r['source_height']} | "
            f"{r['crop_box'].replace(' ', ', ')} | {r['tags']} | `{r['sha256'][:12]}` |"
        )
    n_user = sum(1 for r in rows if r["origin"] == "self-collected")
    lines += [
        "",
        f"Total: {len(rows)} images ({len(rows) - n_user} from the original repository, {n_user} self-collected).",
        "",
        "## Self-collected photos",
        "",
        "Target: 10 to 20 photos covering portraits, landscapes, still life and high-texture scenes, taken or owned by the",
        "student (no licensing issues in the report). Drop the originals into `data/raw_photos/` and rerun",
        "`scripts/freeze_dataset.py --force`; the repo images are re-generated byte-identically, so their hashes stay valid.",
        "This must happen before the Week 5 ablations start; after that the set is closed.",
        "",
        "## Preprocessing figure",
        "",
        "`report/figures/preprocessing_pipeline.png` (from `scripts/make_preprocessing_figure.py`) shows input, crop,",
        "normalized tensor and grid split for one image of this set.",
    ]
    DATASET_MD.parent.mkdir(parents=True, exist_ok=True)
    DATASET_MD.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {MANIFEST} and {DATASET_MD} ({len(rows)} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
