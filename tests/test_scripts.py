"""The measurement scripts: the pieces that must not rot because their real exercise is on another machine."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest
import torch

from neural_painter.models.neural_renderer import NeuralRenderer, save_checkpoint

ROOT = Path(__file__).resolve().parent.parent


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_pytest_output_reads_counts_and_skip_reasons():
    device_check = load_script("device_check")
    text = (
        "..s.F\n=========== short test summary info ===========\n"
        "SKIPPED [2] tests\\test_renderer.py:178: checkpoints_G_markerpen has not been downloaded\n"
        "SKIPPED [1] tests/conftest.py:23: original stylized-neural-painting checkout not present\n"
        "1 failed, 81 passed, 3 skipped in 45.12s\n"
    )
    counts, skips = device_check.parse_pytest_output(text)
    assert counts == {"failed": 1, "passed": 81, "skipped": 3}
    assert skips == [
        "tests/conftest.py:23: original stylized-neural-painting checkout not present",
        "tests\\test_renderer.py:178: checkpoints_G_markerpen has not been downloaded",
    ]
    assert device_check.parse_pytest_output("") == ({}, [])


def test_device_check_report_shows_failures_and_skips():
    device_check = load_script("device_check")
    results = {
        "created": "2026-10-09T17:00:00",
        "environment": {"commit": "abc1234", "dirty": False, "device": "mps", "cuda_available": False, "mps_available": True, "cpu_threads": 10,
                        "summary": {"python": "3.14.4", "torch": "2.14.0", "platform": "macOS-15"}, "reference_checkout": True,
                        "pretrained_checkpoints": {"oilpaintbrush_light": True, "watercolor": False}},
        "dependencies": {"ok": True, "seconds": 3.0, "versions": {"torch": "2.14.0", "lpips": "0.1.4"}, "ffmpeg": "/x/ffmpeg"},
        "tests": {"ok": False, "seconds": 90.0, "error": "pytest exited with 1", "counts": {"failed": 1, "passed": 80}, "skipped_reasons": ["a.py:1: why"],
                  "tail": "FAILED tests/test_x.py::test_y"},
        "renderers": {"ok": True, "seconds": 5.0, "rows": [
            {"device": "mps", "renderer": "oilpaintbrush light", "forward_ms": 2.5, "forward_backward_ms": 5.0, "peak_memory_gb": 0.1},
            {"device": "cpu", "renderer": "oilpaintbrush full", "skipped": "checkpoint not downloaded"}]},
        "training": {"ok": True, "seconds": 60.0, "epoch2_line": "epoch   2/2 | loss 1.0"},
        "painting": {"ok": True, "seconds": 40.0, "rows": [
            {"loss": "pixel", "device": "mps", "strokes": 100, "steps": 500, "optimize_s": 12.3, "render_s": 1.5, "psnr": 17.5, "ssim": 0.41},
            {"loss": "pixel + Sinkhorn", "device": "mps", "strokes": 100, "steps": 500, "optimize_s": 30.1, "render_s": 1.5, "psnr": 17.6, "ssim": 0.42}]},
    }
    md = device_check.report_markdown(results)
    assert "device: `mps`; Python 3.14.4; torch 2.14.0" in md
    assert "| tests | FAILED: pytest exited with 1 | 90.0 |" in md and "FAILED tests/test_x.py::test_y" in md
    assert "- a.py:1: why" in md and "| mps | oilpaintbrush light | 2.5 | 5.0 | 0.10 |" in md and "checkpoint not downloaded" in md
    assert "epoch   2/2" in md and "lpips 0.1.4" in md
    assert "| pixel + Sinkhorn | mps | 100 | 500 | 30.1 | 1.5 | 17.6 | 0.42 |" in md and "Painting smoke test (apple, 5 x 5 grid)" in md
    results["painting"] = {"ok": True, "seconds": 0.1, "skipped": "the pretrained light oil renderer or data/eval_set/apple.png is missing"}
    assert "Skipped: the pretrained light oil renderer" in device_check.report_markdown(results)


def test_device_check_paints_a_small_picture_with_both_losses():
    device_check = load_script("device_check")
    from neural_painter.models.neural_renderer import original_checkpoint_path

    if not original_checkpoint_path("oilpaintbrush", light=True).is_file() or not (ROOT / "data" / "eval_set" / "apple.png").is_file():
        pytest.skip("needs the pretrained light oil renderer and the evaluation image")
    result = device_check.check_painting("cpu", strokes_per_block=2, iters_per_stroke=3)
    assert [r["loss"] for r in result["rows"]] == ["pixel", "pixel + Sinkhorn"]
    assert all(r["strokes"] == 50 and r["steps"] == 6 and r["device"] == "cpu" and r["psnr"] > 0 for r in result["rows"])  # 25 blocks x 2 strokes, 3 steps each


def test_training_curve_plot_marks_learning_rate_drops(tmp_path):
    plot_script = load_script("plot_training_curve")
    fields = ["epoch", "train_loss", "psnr_fg", "psnr_alpha", "psnr_mean", "ssim_fg", "ssim_alpha", "lr", "epoch_seconds", "strokes_per_second"]
    with open(tmp_path / "curve.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for e in range(1, 7):
            writer.writerow({"epoch": e, "train_loss": 5.0 / e, "psnr_fg": 20 + e, "psnr_alpha": 15 + e, "psnr_mean": 17.5 + e, "ssim_fg": 0.9,
                             "ssim_alpha": 0.9, "lr": 2e-4 if e <= 3 else 2e-5, "epoch_seconds": 30, "strokes_per_second": 1600})
    curve = plot_script.read_curve(tmp_path)
    fig = plot_script.plot(curve, "unit", reference=24.4)
    assert len(fig.axes) == 2 and fig.axes[0].get_yscale() == "log"
    vlines = [line for line in fig.axes[1].lines if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1]]
    assert [float(line.get_xdata()[0]) for line in vlines] == [4.0]  # the first epoch trained at the lower rate
    out = tmp_path / "plots" / "curve.png"
    assert plot_script.main([str(tmp_path), "--reference", "24.4", "--out", str(out)]) == 0 and out.stat().st_size > 5000
    with pytest.raises(ValueError, match="no rows"):
        (tmp_path / "empty").mkdir()
        (tmp_path / "empty" / "curve.csv").write_text(",".join(fields) + "\n", encoding="utf-8")
        plot_script.read_curve(tmp_path / "empty")


def test_content_tags_cover_the_frozen_evaluation_set():
    freeze = load_script("freeze_dataset")
    assert freeze.tag_for("alien") == freeze.TAGS["alien"]
    long_name = "51e3d4424d53b10ff3d8992cc12c30771037dbf85254784b7d297ed7934b_640"
    assert freeze.tag_for(long_name) == freeze.TAGS["51e3d4424d53"]  # hash-style names match by their first 12 characters
    assert freeze.tag_for("fire_2") == freeze.GENERIC_TAG  # short keys only match exactly
    assert freeze.tag_for("unlisted-photo") == freeze.GENERIC_TAG
    with open(ROOT / "data" / "eval_set" / "manifest.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and {r["origin"] for r in rows} <= {"original repo test_images", "self-collected"}
    untagged = [r["name"] for r in rows if r["tags"] == freeze.GENERIC_TAG]
    assert not untagged, f"images in the frozen manifest without a content description: {untagged}"


def test_fidelity_table_formatting():
    fidelity = load_script("renderer_fidelity")
    row = {f: None for f in fidelity.FIELDS}
    row.update(renderer="original light", brush="oilpaintbrush", size=32, params_m=5.3669, epochs=400, psnr_fg=27.38, psnr_alpha=21.38,
               psnr_mean=24.38, ssim_fg=0.9441, ssim_alpha=0.92, forward_ms=1.84, forward_backward_ms=3.6)
    table = fidelity.markdown_table([row])
    assert "| original light | oilpaintbrush | 32 | 5.37 | 400 | - | 27.38 | 21.38 | 24.38 | 0.944 | 0.920 | 1.8 | 3.6 | - |" in table


def test_fidelity_script_scores_a_checkpoint_end_to_end(tmp_path, monkeypatch, capsys):
    fidelity = load_script("renderer_fidelity")
    monkeypatch.setattr(fidelity, "EXPERIMENT_ROOT", tmp_path)
    torch.manual_seed(0)
    path = save_checkpoint(tmp_path / "tape_light.pt", NeuralRenderer("rectangle", light=True), epoch=1)
    code = fidelity.main(["--device", "cpu", "--n", "16", "--brush", "tape", "--no-original", "--ours", f"untrained={path}", "--figure", "--name", "unit"])
    assert code == 0
    out = next(tmp_path.glob("*_fidelity_unit"))
    figure = out / "comparison_untrained.png"
    assert figure.is_file() and figure.stat().st_size > 1000  # rasterizer + untrained rows, foreground and opacity
    rows = list(csv.DictReader(open(out / "results.csv", encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["renderer"] == "untrained" and rows[0]["brush"] == "rectangle" and rows[0]["size"] == "32"
    assert float(rows[0]["psnr_mean"]) < 20.0  # an untrained net is far from the rasterizer
    assert float(rows[0]["forward_backward_ms"]) > 0.0
    assert (out / "results.md").read_text(encoding="utf-8").startswith("# Renderer fidelity") and (out / "config.yaml").is_file()
    assert "untrained" in capsys.readouterr().out
    assert fidelity.main(["--device", "cpu", "--no-original", "--name", "none"]) == 1  # nothing to score


def test_run_paint_writes_a_painting_in_the_photo_aspect_ratio(tmp_path, capsys):
    from neural_painter.models.neural_renderer import original_checkpoint_path

    if not original_checkpoint_path("oilpaintbrush", light=True).is_file():
        pytest.skip("the pretrained light oil renderer has not been downloaded")
    import numpy as np

    from neural_painter.core.image_io import load_image, save_image

    photo = tmp_path / "wide.png"
    save_image(photo, np.random.default_rng(0).random((60, 120, 3)).astype(np.float32))  # 2:1
    run_paint = load_script("run_paint")
    out = tmp_path / "out"
    assert run_paint.main([str(photo), "--strokes", "14", "--max-divide", "3", "--size", "96", "--device", "cpu", "--out", str(out), "--quiet",
                           "--timelapse"]) == 0
    assert load_image(out / "final.png").shape == (48, 96, 3)  # stretched back to the photo's 2:1
    assert load_image(out / "input.png").shape == (96, 96, 3)
    rows = list(csv.DictReader(open(out / "results.csv", encoding="utf-8")))
    assert rows[0]["strokes"] == "14" and rows[0]["grid"] == "progressive" and float(rows[0]["peak_memory_gb"]) >= 0
    assert "levels" in (out / "config.yaml").read_text(encoding="utf-8") and "14 strokes" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="not empty"):
        run_paint.main([str(photo), "--out", str(out)])
    from PIL import Image

    assert Image.open(out / "timelapse.gif").size == (96, 48) and (out / "timelapse.mp4").stat().st_size > 0  # Week 9
    load_script("make_timelapse").main([str(out), "--out", str(tmp_path / "again"), "--format", "gif", "--stride", "7", "--hold", "0"])
    again = Image.open(tmp_path / "again" / "timelapse.gif")
    assert again.size == (96, 48) and again.n_frames == 3  # blank, after 7 and 14 strokes, on the canvas the file records
    assert "3 frames" in capsys.readouterr().out


def test_benchmark_builds_one_runner_call_per_brush():
    benchmark = load_script("benchmark")
    args = benchmark.main.__globals__["argparse"].Namespace(
        profile="mps", modes=["progressive", "fixed"], max_divide=5, quick=False, every=6, seeds=[0, 1], renderer="original-light",
        strokes=500, max_minutes=0.0)
    line = benchmark.ablation_args(args, "marker")
    assert line[:4] == ["--kind", "benchmark", "--name", "mps_marker"] and line[line.index("--device") + 1] == "mps"
    assert line[line.index("--configs") + 1 : line.index("--configs") + 3] == [
        "progressive=0@5,mode=progressive,lr_schedule=cosine,lr=0.004", "fixed=0@5,mode=fixed"]
    args.quick = True
    quick = benchmark.ablation_args(args, "oil")
    assert quick[3] == "cpu_oil_quick".replace("cpu", "mps") and quick[quick.index("--seeds") + 1 : quick.index("--configs")] == ["0"]


def test_closed_form_composite_equals_the_loop():
    from neural_painter.core.differentiable_canvas import composite_sequence

    profile = load_script("profile_step")
    g = torch.Generator().manual_seed(0)
    canvas, fg, alpha = torch.rand(2, 3, 8, 8, generator=g), torch.rand(2, 5, 3, 8, 8, generator=g), torch.rand(2, 5, 3, 8, 8, generator=g)
    torch.testing.assert_close(profile.closed_form(canvas, fg, alpha), composite_sequence(canvas, fg, alpha))


def test_usability_summary_scores_sus_and_ranks_issues(tmp_path):
    summary = load_script("usability_summary")
    assert summary.sus_score([3] * 10) == 50.0
    assert summary.sus_score([5, 1] * 5) == 100.0 and summary.sus_score([1, 5] * 5) == 0.0
    with pytest.raises(ValueError):
        summary.sus_score([3] * 9)
    (tmp_path / "tasks.csv").write_text("participant,task,outcome,seconds,notes\nP1,T1,done,80,\nP2,T1,help,120,needed a hint\n"
                                        "P1,T2,done,30,\nP2,T2,failed,,\n", encoding="utf-8")
    (tmp_path / "sus.csv").write_text("participant,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10\nP1,5,1,5,1,5,1,5,1,5,1\nP2,3,3,3,3,3,3,3,3,3,3\n",
                                      encoding="utf-8-sig")
    (tmp_path / "issues.csv").write_text("participant,issue,severity\nP1,Stop button unclear,1\nP2,Stop button unclear,2\n"
                                         "P2,Did not find the replay,3\n", encoding="utf-8")
    text = summary.summarize(tmp_path)
    assert "| T1 | 2 | 1 | 1 | 0 | 100 |" in text and "| T2 | 2 | 1 | 0 | 1 | 30 |" in text
    assert "Mean 75.0" in text and "| P2 | 50.0 |" in text
    assert text.index("Stop button unclear") < text.index("Did not find the replay")  # 2 participants x 2 before 1 x 3
