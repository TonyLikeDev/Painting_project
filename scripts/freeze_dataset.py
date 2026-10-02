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
RAW_MAX_PIXELS = 100_000_000  # camera originals exceed image_io's 40 MP upload guard; this script runs offline

# short content tags for the report table, by file stem (long hash-style names by their first 12 characters);
# a photo without an entry gets the generic tag below
TAGS = {
    # the 11 images of the original repository
    "alien": "creature figure, dark glossy surfaces, white background",
    "apple": "still life, red apples in a wooden crate, wood texture",
    "cube2": "translucent coloured cubes, flat colour regions, sharp edges, white background",
    "diamond": "faceted gem on dark background, specular highlights",
    "diamond2": "glass block with iridescent edges, black background",
    "fire": "flames, high-frequency texture, dark background",
    "iceland": "landscape, waterfall and mountain at sunset, sky and water",
    "jay": "portrait, face and hand, dark hair, shallow depth of field",
    "joker": "face / portrait, painted make-up",
    "sunflowers": "sunflowers in a basket, saturated yellow, blurred background",
    "yosemite": "landscape, rock face, waterfall and trees",
    # the self-collected photos
    "51e3d4424d53": "minimal still life, pink balloon and white chair, flat pale background",
    "53e2d3464b5b": "urban scene, cafe chairs and tables, repeated geometric forms",
    "53e8d44b4253": "colour pencils in a radial pattern, saturated colours, sharp tips",
    "54e2d3424c54": "macro still life, daisy and pencils, shallow depth of field",
    "54e4dd474257": "stacked crates, dense saturated colour blocks, high-frequency texture",
    "54e9d14b4e52": "animal, kitten in grass and daisies, shallow depth of field",
    "55e4d4414a50": "dandelion seed on dark background, thin structures, reflection",
    "57e3d6464e55": "black-and-white checkerboard with a drain and water, high contrast",
    "57e4d0404d5b": "cactus, repeated spines, green high-frequency texture",
    "57e4d1474e5b": "food still life, cereal bowl and spoon, warm colours",
    "57e5dd464d54": "people on a vintage motorbike, faded retro colour grading",
    "57e8d7414852": "abstract stacked paper layers, green and yellow diagonal bands",
    "ca-si-bui-truong-linh-1": "studio portrait, person in a green jacket, flat yellow background",
    "emily-lau-NVi2yab124g-unsplash": "outdoor portrait, tinted glasses, blurred background",
    "ice-cream-cone-1274894_640": "still life, ice-cream cone, saturated blue background",
    "maria-lysenko-3Bh0hy-yOcA-unsplash": "outdoor portrait, park background, shallow depth of field",
    "pexels-beratorer-30650522": "macro flowers, yellow daisies, dark background",
    "pexels-christina99999-38524143": "macro flowers, pink petals, fine detail",
    "pexels-molnartamasphotography-29202983": "landscape, hazy hills and water, low contrast, low-frequency regions",
}
GENERIC_TAG = "self-collected photo"


def tag_for(stem: str) -> str:
    """The content tag of an image: exact stem, else a key of 12 or more characters that the stem starts with, else the generic tag."""
    if stem in TAGS:
        return TAGS[stem]
    return next((tag for key, tag in TAGS.items() if len(key) >= 12 and stem.startswith(key)), GENERIC_TAG)


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
        img = image_io.load_image(src, max_pixels=RAW_MAX_PIXELS)
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
                "tags": tag_for(src.stem),
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
        "Photographs gathered by the student to cover portraits, landscapes, still life and high-texture scenes (target: 10 to",
        "20). They are not part of the original repository; each keeps the file name it was collected under (`source_file` in",
        "the manifest). The originals are in `data/raw_photos/`; the frozen 512 x 512 PNGs in `data/eval_set/` are the dataset,",
        "so results do not need the originals. To add photos, drop them into `data/raw_photos/` and rerun",
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
