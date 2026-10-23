"""Make the aesthetic survey of Week 9: four brushes on five images, rated 1 to 5 by at least 10 people (RESEARCH_PLAN 3.3).

Takes the 20 paintings from the brush comparison of Week 7 (``experiments/*_ablation_brushes_prog_<brush>``, config ``prog_cos4``,
seed 0: the pipeline's settings, 495 strokes), shuffles them once with a fixed seed so that no two neighbours share an image or a
brush, and writes to ``--out`` (default ``report/notes/aesthetic_survey``):

* ``paintings/01.png`` to ``20.png``: the paintings under their survey numbers only (for a screen, or a Google Form);
* ``paintings.pdf``: the same, four to an A4 page with large numbers (print one colour copy, or show it on a screen);
* ``answer_sheet_vi.pdf``: the one-page Vietnamese answer sheet, one per rater (black and white is enough);
* ``key.csv``: survey number -> image, brush and the painting's PSNR / SSIM / LPIPS (not for the raters).

and the empty ratings file ``experiments/aesthetic_survey_template/ratings.csv``. ``scripts/survey_summary.py`` scores the filled one.
The PDFs are printed by headless Chrome or Edge (the same browsers ``app_screenshots.py`` uses).

Usage: venv/Scripts/python.exe scripts/make_survey.py [--out report/notes/aesthetic_survey] [--seed 9]
"""
from __future__ import annotations

import argparse
import csv
import html
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app_screenshots import CHROME_CANDIDATES  # noqa: E402

# portrait, landscape, still life, flowers, minimal scene: five of the ten images the brush comparison kept paintings of
IMAGES = ("jay", "yosemite", "57e4d1474e5ba914f1dc8460962e33791c3ad6e04e5074417d2f7dd49f4ec6_640", "pexels-beratorer-30650522",
          "51e3d4424d53b10ff3d8992cc12c30771037dbf85254784b7d297ed7934b_640")
BRUSHES = ("oilpaintbrush", "watercolor", "markerpen", "rectangle")
CONFIG, SEED = "prog_cos4", 0
SCALE = ("Không đẹp chút nào", "Không đẹp", "Bình thường", "Đẹp", "Rất đẹp")


def survey_order(seed: int) -> list[tuple[str, str]]:
    """The 20 (image, brush) pairs in survey order: shuffled, no two neighbours with the same image or the same brush."""
    rng, items = random.Random(seed), [(i, b) for i in IMAGES for b in BRUSHES]
    while True:
        rng.shuffle(items)
        if all(a[0] != b[0] and a[1] != b[1] for a, b in zip(items, items[1:])):
            return list(items)


def source_folder(brush: str) -> Path:
    found = sorted((ROOT / "experiments").glob(f"*_ablation_brushes_prog_{brush}"))
    if not found:
        raise SystemExit(f"no experiments/*_ablation_brushes_prog_{brush} folder")
    return found[-1]


def scores(folder: Path, image: str) -> dict:
    with open(folder / "results.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["image"] == image and row["config"] == CONFIG and int(row["seed"]) == SEED:
                return {k: row.get(k, "") for k in ("psnr", "ssim", "lpips")}
    raise SystemExit(f"{folder.name} has no {CONFIG} seed {SEED} row for {image}")


PAGE_CSS = """@page { size: A4; margin: 12mm; } body { font-family: Arial, sans-serif; margin: 0; color: #000; }
h1 { font-size: 20pt; margin: 0 0 4mm; } p { font-size: 11pt; margin: 2mm 0; line-height: 1.35; }"""


def paintings_html(order: list, folder: Path) -> str:
    cells = [f'<div class="cell"><div class="num">Tranh {n}</div><img src="{(folder / f"{n:02d}.png").as_uri()}"></div>'
             for n in range(1, len(order) + 1)]
    pages = "".join(f'<div class="page">{"".join(cells[i:i + 4])}</div>' for i in range(0, len(cells), 4))
    return f"""<!doctype html><html lang="vi"><meta charset="utf-8"><style>{PAGE_CSS}
.page {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6mm; page-break-after: always; }}
.cell {{ text-align: center; }} .num {{ font-size: 18pt; font-weight: bold; margin-bottom: 2mm; }}
img {{ width: 88mm; height: 118mm; object-fit: contain; }}</style><body>{pages}</body></html>"""


def answer_sheet_html(n: int) -> str:
    circles = "".join(f"<td>{k}</td>" for k in range(1, 6))
    rows = [f"<tr><th>Tranh {i}</th>{circles}</tr>" for i in range(1, n + 1)]
    half = (n + 1) // 2
    table = lambda part: f'<table><tr><th></th>{"".join(f"<th>{k}</th>" for k in range(1, 6))}</tr>{"".join(part)}</table>'
    scale = ", ".join(f"<b>{k}</b> = {html.escape(t)}" for k, t in enumerate(SCALE, 1))
    return f"""<!doctype html><html lang="vi"><meta charset="utf-8"><style>{PAGE_CSS}
.tables {{ display: flex; gap: 10mm; margin-top: 5mm; }} table {{ border-collapse: collapse; font-size: 12pt; }}
th, td {{ border: 1px solid #000; padding: 2.2mm 4mm; text-align: center; }} th {{ text-align: left; }}</style><body>
<h1>Phiếu đánh giá tranh</h1>
<p>Mã người đánh giá: R______ &nbsp;&nbsp; Ngày: ____ / ____ / 2026</p>
<p>Bạn đã từng học vẽ, hoặc làm công việc liên quan đến mỹ thuật? &nbsp; ☐ Có &nbsp;&nbsp; ☐ Không</p>
<p>Bạn sẽ xem lần lượt {n} bức tranh, đánh số từ 1 đến {n}. Các bức tranh do máy tính vẽ bằng các nét cọ. Với mỗi bức, hãy
<b>khoanh tròn một số</b>: bạn thấy bức tranh <b>đẹp</b> đến mức nào? Hãy đánh giá bức tranh như một tác phẩm hội họa, không cần
nghĩ nó giống ảnh gốc đến đâu. Không có câu trả lời đúng hay sai; hãy trả lời theo cảm nhận đầu tiên của bạn.</p>
<p>{scale}</p>
<div class="tables">{table(rows[:half])}{table(rows[half:])}</div>
<p style="margin-top: 6mm">Cảm ơn bạn đã tham gia!</p></body></html>"""


def print_pdf(page: str, pdf: Path) -> None:
    chrome = next((c for c in CHROME_CANDIDATES if Path(c).is_file()), None)
    if chrome is None:
        raise SystemExit("no Chrome or Edge found to print the PDF")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        source = Path(tmp) / "page.html"
        source.write_text(page, encoding="utf-8")
        subprocess.run([chrome, "--headless=new", "--disable-gpu", f"--user-data-dir={tmp}", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf}", source.as_uri()], check=True, capture_output=True, timeout=120)
    if not pdf.is_file():
        raise RuntimeError(f"the browser did not write {pdf}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "report" / "notes" / "aesthetic_survey")
    ap.add_argument("--template", type=Path, default=ROOT / "experiments" / "aesthetic_survey_template" / "ratings.csv")
    ap.add_argument("--seed", type=int, default=9, help="seed of the shuffle (fixed, so the order can be made again)")
    args = ap.parse_args(argv)
    order = survey_order(args.seed)
    folders = {b: source_folder(b) for b in BRUSHES}
    paintings = args.out / "paintings"
    paintings.mkdir(parents=True, exist_ok=True)
    with open(args.out / "key.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["item", "image", "brush", "psnr", "ssim", "lpips", "source"])
        for n, (image, brush) in enumerate(order, 1):
            source = folders[brush] / "runs" / image / f"{CONFIG}_s{SEED}" / "final.png"
            shutil.copyfile(source, paintings / f"{n:02d}.png")
            m = scores(folders[brush], image)
            writer.writerow([n, image, brush, m["psnr"], m["ssim"], m["lpips"], source.relative_to(ROOT).as_posix()])
    print_pdf(paintings_html(order, paintings), args.out / "paintings.pdf")
    print_pdf(answer_sheet_html(len(order)), args.out / "answer_sheet_vi.pdf")
    args.template.parent.mkdir(parents=True, exist_ok=True)
    args.template.write_text(",".join(["rater", "art"] + [str(n) for n in range(1, len(order) + 1)]) + "\n", encoding="utf-8")
    print(f"wrote {len(order)} paintings, paintings.pdf, answer_sheet_vi.pdf and key.csv to {args.out}; template {args.template}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
