"""The ablation runner and its summary: parsing, statistics, and a tiny resumable end-to-end run."""
from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from neural_painter.models.neural_renderer import original_checkpoint_path

ROOT = Path(__file__).resolve().parent.parent


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_parsing():
    run = load_script("run_ablation")
    assert run.parse_config("pixel=0") == ("pixel", {"beta_ot": 0.0})
    assert run.parse_config("ot=0.1") == ("ot", {"beta_ot": 0.1})
    assert run.parse_config("sweep=1:0.005:10") == ("sweep", {"beta_ot": 1.0, "ot_epsilon": 0.005, "ot_iters": 10})
    assert run.parse_config("full=0@1") == ("full", {"beta_ot": 0.0, "grid": 1})
    assert run.parse_config("g3=0.1:0.02@3") == ("g3", {"beta_ot": 0.1, "ot_epsilon": 0.02, "grid": 3})
    assert run.parse_config("flat=0,init=uniform") == ("flat", {"beta_ot": 0.0, "init": "uniform"})
    assert run.parse_config("x=1@2,init=uniform,centered=true,lr=0.01") == ("x", {"beta_ot": 1.0, "grid": 2, "init": "uniform", "centered": True, "lr": 0.01})
    assert run.parse_config("x=0,centered=no")[1]["centered"] is False
    assert run.parse_config("bare=0,morphology=false") == ("bare", {"beta_ot": 0.0, "morphology": False})
    for bad in ("pixel", "=0.1", "x=", "x=0@", "x=@2", "x=0,bogus=1", "x=0,init", "x=0,"):
        with pytest.raises(argparse.ArgumentTypeError):
            run.parse_config(bad)


def synthetic_rows(gain: float = 0.2) -> list[dict]:
    """4 images x 2 configs x 3 seeds; ``ot`` is ``gain`` dB better than ``pixel`` on every pair, plus noise."""
    rng = np.random.default_rng(0)
    rows = []
    for image in ("a", "b", "c", "d"):
        base = 20 + 3 * rng.random()
        for seed in (0, 1, 2):
            noise = rng.normal(0, 0.05)
            for config, offset in (("pixel", 0.0), ("ot", gain)):
                rows.append({"image": image, "config": config, "seed": str(seed), "psnr": f"{base + noise + offset:.4f}", "ssim": f"{0.5 + noise / 10:.4f}",
                             "lpips": "", "neural_psnr": f"{25 + offset:.3f}", "optimize_s": "10" if config == "pixel" else "13",
                             "steps_to_22db": "100", "steps_to_24db": ""})
    return rows


def test_summary_reports_means_over_seeds_and_paired_differences():
    summary = load_script("summarize_ablation")
    rows = synthetic_rows(0.2)
    text, csv_rows = summary.summarize(rows, "pixel")
    assert [r["config"] for r in csv_rows] == ["pixel", "ot"]
    assert csv_rows[1]["psnr"] - csv_rows[0]["psnr"] == pytest.approx(0.2, abs=1e-3)
    assert "| ot | PSNR (dB) | +0.200 |" in text and "100% of 4" in text  # a paired gain of 0.2 dB, in every one of the 4 images
    assert "LPIPS" not in text and "24 dB" not in text  # empty columns are not shown
    assert "100 (12/12)" in text and csv_rows[0]["steps_to_22db"] == 100.0  # median steps over the runs that reach 22 dB, and how many do
    partial, _ = summary.summarize([dict(r, steps_to_22db="" if r["seed"] == "2" else "80") for r in rows], "pixel")
    assert "| pixel | " in partial and partial.count("80 (8/12)") == 2  # a threshold missed by a third of the runs says so
    seeds = summary.per_seed_means(rows, "pixel", "psnr")
    assert seeds.shape == (3,) and seeds.mean() == pytest.approx(csv_rows[0]["psnr"])
    d = summary.paired(rows, "pixel", "ot", "psnr")
    assert d.shape == (4,) and np.allclose(d, 0.2, atol=1e-6)  # the unit is the image: its three seeds are averaged first
    assert summary.paired(rows, "pixel", "ot", "psnr", per="run").shape == (12,)
    assert "p" in text.splitlines()[-1] or "Wilcoxon" in text


def test_paired_difference_averages_the_seeds_of_each_image():
    summary = load_script("summarize_ablation")
    rows = [{"image": "a", "config": c, "seed": str(s), "psnr": f"{v}"}
            for c, vals in (("pixel", (20.0, 20.0, 20.0)), ("ot", (21.0, 22.0, 20.0))) for s, v in enumerate(vals)]
    rows += [{"image": "b", "config": c, "seed": "0", "psnr": f"{v}"} for c, v in (("pixel", 18.0), ("ot", 17.0))]
    assert summary.paired(rows, "pixel", "ot", "psnr").tolist() == pytest.approx([1.0, -1.0])  # image a: mean(+1, +2, 0); image b: -1
    assert sorted(summary.paired(rows, "pixel", "ot", "psnr", per="run").tolist()) == pytest.approx([-1.0, 0.0, 1.0, 2.0])
    rows = [r for r in rows if not (r["image"] == "a" and r["seed"] == "1" and r["config"] == "ot")]  # a missing run drops its pair only
    assert summary.paired(rows, "pixel", "ot", "psnr").tolist() == pytest.approx([0.5, -1.0])


def fake_ablation_folder(root: Path, gains: dict[str, float], seeds=(0, 1), steps: int = 12) -> Path:
    """A folder shaped like ``run_ablation.py``'s output: results.csv, per-run final.png and history.npz, and an eval set."""
    import cv2

    folder, eval_dir = root / "2026-10-03_ablation_fake", root / "eval"
    eval_dir.mkdir()
    rows = []
    for k, (image, gain) in enumerate(gains.items()):
        cv2.imwrite(str(eval_dir / f"{image}.png"), np.full((16, 16, 3), (40 * k + 40) % 256, np.uint8))
        for config, offset in (("pixel", 0.0), ("ot", gain)):
            for seed in seeds:
                run = folder / "runs" / image / f"{config}_s{seed}"
                run.mkdir(parents=True)
                cv2.imwrite(str(run / "final.png"), np.full((16, 16, 3), (60 * k + 20 * seed + 30) % 256, np.uint8))
                np.savez(run / "history.npz", psnr=20 + offset + np.linspace(0, 4, steps), loss=np.zeros(steps), anchor=np.zeros(steps))
                rows.append({"image": image, "config": config, "seed": str(seed), "psnr": f"{22 + offset:.3f}"})
    with open(folder / "results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return folder


def test_figure_images_are_picked_by_rule_not_by_eye(tmp_path):
    plot = load_script("plot_ablation")
    folder = fake_ablation_folder(tmp_path, {"a": 0.5, "b": -0.4, "c": 0.1, "d": 0.3, "e": -0.1})
    picks = plot.pick_images(plot.read_rows(folder), "pixel", "ot")
    assert [(k, i) for k, i, _ in picks] == [("best", "a"), ("median", "c"), ("worst", "b")]  # sorted gains: a .5, d .3, c .1, e -.1, b -.4
    assert picks[0][2] == pytest.approx(0.5) and picks[2][2] == pytest.approx(-0.4)
    (tmp_path / "one").mkdir()
    only = fake_ablation_folder(tmp_path / "one", {"a": 0.2})
    assert [k for k, _, _ in plot.pick_images(plot.read_rows(only), "pixel", "ot")] == ["best"]  # one image: no duplicate pictures
    assert plot.pick_images([], "pixel", "ot") == []


def test_ablation_figures_are_written(tmp_path):
    plot = load_script("plot_ablation")
    folder = fake_ablation_folder(tmp_path, {"a": 0.5, "b": -0.4, "c": 0.1})
    out = tmp_path / "figures"
    assert plot.main([str(folder), "--baseline", "pixel", "--other", "ot", "--out-dir", str(out), "--eval-dir", str(tmp_path / "eval")]) == 0
    assert (out / "ablation_fake_qualitative.png").stat().st_size > 1000 and (out / "ablation_fake_convergence.png").stat().st_size > 1000
    assert plot.main([str(folder), "--out-dir", str(out), "--tag", "t", "--skip", "qualitative"]) == 0  # baseline and other default to first and last
    assert (out / "t_convergence.png").is_file() and not (out / "t_qualitative.png").exists()
    curves = plot.history_matrix(folder, "ot", ["a", "b"], [0, 1])
    assert curves.shape == (2, 12) and curves[0, 0] == pytest.approx(20.5) and curves[1, 0] == pytest.approx(19.6)
    mean, half = plot.band(curves)
    assert mean[0] == pytest.approx((20.5 + 19.6) / 2) and half.shape == (12,) and np.all(half > 0)


def test_gradient_ratio_summary_takes_the_median_of_the_per_probe_ratios():
    grads = load_script("loss_gradients")
    rows = [{"phase": "first", "l1_all": 2.0, "ot_all": 1.0, "l1_new": 1.0, "ot_new": 3.0, "l1_pos": 1.0, "ot_pos": 0.5},
            {"phase": "first", "l1_all": 1.0, "ot_all": 1.0, "l1_new": 1.0, "ot_new": 1.0, "l1_pos": 1.0, "ot_pos": 0.1},
            {"phase": "first", "l1_all": 4.0, "ot_all": 1.0, "l1_new": 1.0, "ot_new": 2.0, "l1_pos": 1.0, "ot_pos": 0.3},
            {"phase": "last", "l1_all": 1.0, "ot_all": 0.2, "l1_new": 1.0, "ot_new": 0.2, "l1_pos": 1.0, "ot_pos": 0.2}]
    text = grads.summarize(rows)
    assert "| first step of a new stroke | 0.5 | 2 | 0.3 |" in text  # medians of (0.5, 1, 0.25), (3, 1, 2) and (0.5, 0.1, 0.3)
    assert "| last step of a stroke | 0.2 | 0.2 | 0.2 |" in text


def test_gradient_probe_runs_end_to_end_on_a_tiny_painting(tmp_path, monkeypatch):
    if not original_checkpoint_path("oilpaintbrush", light=True).is_file():
        pytest.skip("the pretrained light oil renderer has not been downloaded")
    grads = load_script("loss_gradients")
    monkeypatch.setattr(grads, "EXPERIMENT_ROOT", tmp_path)
    assert grads.main(["--images", "apple", "--every", "1", "--grid", "2", "--strokes", "8", "--iters", "4", "--device", "cpu", "--name", "t"]) == 0
    folder = next(tmp_path.glob("*_diagnostic_t"))
    rows = list(csv.DictReader(open(folder / "gradients.csv", encoding="utf-8")))
    assert [(r["anchor"], r["phase"]) for r in rows] == [("0", "first"), ("0", "last"), ("1", "first"), ("1", "last")]  # 2 strokes x (first, last step)
    for r in rows:
        assert all(np.isfinite(float(r[k])) and float(r[k]) >= 0 for k in ("l1_all", "ot_all", "l1_new", "ot_new", "l1_pos", "ot_pos"))
    assert float(rows[0]["l1_all"]) > 0 and float(rows[0]["ot_all"]) > 0  # both losses have a gradient on the first stroke
    assert (folder / "summary.md").read_text(encoding="utf-8").startswith("# Gradient of the Sinkhorn cost")


def test_lpips_is_added_after_the_fact_and_only_where_missing(tmp_path):
    score = load_script("score_lpips")
    folder = fake_ablation_folder(tmp_path, {"a": 0.5, "b": -0.4})
    rows = list(csv.DictReader(open(folder / "results.csv", encoding="utf-8")))
    for r in rows:
        r["lpips"] = ""
    rows[0]["lpips"] = "0.5"  # a value that must survive
    with open(folder / "results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    seen = []

    def metric(image, target):
        seen.append(image.shape)
        return float(np.abs(image - target).mean())

    scored, kept = score.score_folder(folder, metric, tmp_path / "eval")
    assert (scored, kept) == (len(rows) - 1, 1) and len(seen) == len(rows) - 1
    after = list(csv.DictReader(open(folder / "results.csv", encoding="utf-8")))
    assert after[0]["lpips"] == "0.5" and all(0 <= float(r["lpips"]) <= 1 for r in after)
    assert score.score_folder(folder, metric, tmp_path / "eval") == (0, len(rows))  # repeating changes nothing
    original = folder / "runs" / "a" / "original_s0"  # the original demo names its picture <image>_final.png
    original.mkdir(parents=True)
    (folder / "runs" / "a" / "pixel_s0" / "final.png").replace(original / "a_final.png")
    assert score.painting_path(folder, {"image": "a", "config": "original", "seed": "0"}) == original / "a_final.png"
    with pytest.raises(SystemExit, match="no lpips column"):
        (tmp_path / "bad").mkdir()
        (tmp_path / "bad" / "results.csv").write_text("image,config\na,x\n", encoding="utf-8")
        score.score_folder(tmp_path / "bad", metric, tmp_path / "eval")


def test_sweep_figure_draws_one_point_per_config_at_the_swept_value(tmp_path):
    plot = load_script("plot_sweep")
    rng = np.random.default_rng(1)
    rows = []
    for image in "abcde":
        for seed in (0, 1):
            base = 20 + rng.random()
            rows.append({"image": image, "config": "pixel", "seed": str(seed), "psnr": f"{base:.4f}", "ssim": "0.5", "beta_ot": "0", "ot_epsilon": "0.01"})
            for name, beta, gain in (("w1", 1.0, 0.1), ("w10", 10.0, -0.5), ("w100", 100.0, -2.0)):
                rows.append({"image": image, "config": name, "seed": str(seed), "psnr": f"{base + gain:.4f}", "ssim": f"{0.5 + gain / 10:.4f}",
                             "beta_ot": str(beta), "ot_epsilon": "0.01"})
    points = plot.sweep_points(rows, "pixel", "beta_ot")
    assert [p["config"] for p in points] == ["w1", "w10", "w100"] and [p["x"] for p in points] == [1.0, 10.0, 100.0]  # sorted by the swept value
    assert [round(p["psnr"]["mean"], 3) for p in points] == [0.1, -0.5, -2.0] and all(p["psnr"]["diffs"].shape == (5,) for p in points)
    assert [p["config"] for p in plot.sweep_points(rows, "pixel", "beta_ot", configs=["w10"])] == ["w10"]
    folder = tmp_path / "2026-10-03_ablation_fake_sweep"
    folder.mkdir()
    with open(folder / "results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path / "sweep.png"
    assert plot.main([str(folder), "--baseline", "pixel", "--x", "beta_ot", "--out", str(out)]) == 0 and out.stat().st_size > 1000
    with pytest.raises(SystemExit, match="no config besides"):
        plot.main([str(folder), "--baseline", "pixel", "--configs", "nope", "--out", str(out)])


def test_labels_rename_configs_in_the_markdown_only():
    summary = load_script("summarize_ablation")
    text, csv_rows = summary.summarize(synthetic_rows(0.2), "pixel", {"pixel": "Pixel loss", "ot": "Pixel + Sinkhorn"})
    assert "| Pixel loss |" in text and "| Pixel + Sinkhorn |" in text and "Paired against `Pixel loss`" in text
    assert "| Pixel + Sinkhorn | PSNR (dB) | +0.200 |" in text
    assert [r["config"] for r in csv_rows] == ["pixel", "ot"]  # the machine-readable record keeps the names of the runs


def test_summary_cli_writes_files(tmp_path, capsys):
    summary = load_script("summarize_ablation")
    with open(tmp_path / "results.csv", "w", newline="", encoding="utf-8") as fh:
        rows = synthetic_rows(-0.1)
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert summary.main([str(tmp_path), "--baseline", "pixel"]) == 0
    assert (tmp_path / "summary.md").read_text(encoding="utf-8").startswith("# Ablation summary")
    assert "config" in (tmp_path / "summary.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "-0.100" in capsys.readouterr().out  # a loss shows as a negative paired difference


def test_each_config_can_set_its_own_grid_at_equal_stroke_budget(tmp_path, monkeypatch):
    if not original_checkpoint_path("oilpaintbrush", light=True).is_file():
        pytest.skip("the pretrained light oil renderer has not been downloaded")
    import torch

    run = load_script("run_ablation")
    monkeypatch.setattr(run, "EXPERIMENT_ROOT", tmp_path)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", torch.backends.cudnn.deterministic)  # restored after the test
    assert run.main(["--images", "apple", "--configs", "full=0@1", "grid2=0@2", "--seeds", "0", "--strokes", "8", "--iters", "2",
                     "--device", "cpu", "--name", "grids", "--deterministic"]) == 0
    assert torch.backends.cudnn.deterministic  # the flag reached the backend
    folder = next(tmp_path.glob("*_ablation_grids"))
    rows = {r["config"]: r for r in csv.DictReader(open(folder / "results.csv", encoding="utf-8"))}
    assert (rows["full"]["grid"], rows["full"]["strokes"]) == ("1", "8") and (rows["grid2"]["grid"], rows["grid2"]["strokes"]) == ("2", "8")
    assert np.load(folder / "runs" / "apple" / "full_s0" / "history.npz")["loss"].shape == (16,)  # 8 strokes x 2 steps in one block
    assert np.load(folder / "runs" / "apple" / "grid2_s0" / "history.npz")["loss"].shape == (4,)  # 2 strokes x 2 steps in each of 4 blocks
    text = (folder / "config.yaml").read_text(encoding="utf-8")
    assert "total_strokes: 8" in text and "grid: 2" in text and "cudnn_deterministic: true" in text


def test_every_selects_an_evenly_spread_subset_of_the_images(tmp_path, monkeypatch):
    if not original_checkpoint_path("oilpaintbrush", light=True).is_file():
        pytest.skip("the pretrained light oil renderer has not been downloaded")
    import cv2

    run = load_script("run_ablation")
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    for k in range(6):
        cv2.imwrite(str(eval_dir / f"img{k}.png"), np.full((512, 512, 3), 40 * k, np.uint8))  # paintings are rasterized at 512 px
    monkeypatch.setattr(run, "EVAL_SET", eval_dir)
    monkeypatch.setattr(run, "EXPERIMENT_ROOT", tmp_path)
    args = ["--configs", "pixel=0", "--seeds", "0", "--grid", "1", "--strokes", "4", "--iters", "1", "--device", "cpu"]
    assert run.main(args + ["--every", "2", "--name", "every"]) == 0
    folder = next(tmp_path.glob("*_ablation_every"))
    assert [r["image"] for r in csv.DictReader(open(folder / "results.csv", encoding="utf-8"))] == ["img0", "img2", "img4"]
    assert run.main(args + ["--name", "all"]) == 0
    assert len(list(csv.DictReader(open(next(tmp_path.glob("*_ablation_all")) / "results.csv", encoding="utf-8")))) == 6


def test_runner_is_resumable_and_never_repeats_a_run(tmp_path, monkeypatch, capsys):
    if not original_checkpoint_path("oilpaintbrush", light=True).is_file():
        pytest.skip("the pretrained light oil renderer has not been downloaded")
    run = load_script("run_ablation")
    monkeypatch.setattr(run, "EXPERIMENT_ROOT", tmp_path)
    base = ["--images", "apple", "--configs", "pixel=0", "ot=0.1", "--seeds", "0", "--grid", "2", "--strokes", "16", "--iters", "3", "--device", "cpu"]

    assert run.main(base + ["--name", "unit", "--limit", "1"]) == 0  # one of the two runs
    folder = next(tmp_path.glob("*_ablation_unit"))
    rows = list(csv.DictReader(open(folder / "results.csv", encoding="utf-8")))
    assert [(r["image"], r["config"]) for r in rows] == [("apple", "pixel")] and (folder / "config.yaml").is_file()
    assert "1 left" in capsys.readouterr().out

    assert run.main(base + ["--dir", str(folder)]) == 0  # resumes: only the missing run is done
    rows = list(csv.DictReader(open(folder / "results.csv", encoding="utf-8")))
    assert [r["config"] for r in rows] == ["pixel", "ot"]
    assert "(complete)" in capsys.readouterr().out
    for r in rows:
        assert 0 < float(r["psnr"]) < 100 and r["strokes"] == "16"  # a 16-stroke smoke painting is poor (about 4 dB), but finite and r["device"] == "cpu" and r["renderer"] == "original-light"
        run_dir = folder / "runs" / "apple" / f"{r['config']}_s0"
        assert (run_dir / "final.png").is_file() and (run_dir / "strokes.npz").is_file() and (run_dir / "history.npz").is_file()
        assert np.load(run_dir / "history.npz")["loss"].shape == (12,)  # 4 strokes x 3 steps per block

    assert run.main(base + ["--dir", str(folder)]) == 0  # nothing left: no new run, nothing overwritten
    assert len(list(csv.DictReader(open(folder / "results.csv", encoding="utf-8")))) == 2
    with pytest.raises(SystemExit, match="does not exist"):
        run.main(base + ["--dir", str(tmp_path / "missing")])
    with pytest.raises(SystemExit, match="too few"):
        run.main(["--images", "apple", "--grid", "5", "--strokes", "10", "--device", "cpu", "--name", "x"])
    with pytest.raises(SystemExit, match="too few"):  # the budget is checked against each config's own grid
        run.main(["--images", "apple", "--configs", "wide=0@5", "--grid", "1", "--strokes", "10", "--device", "cpu", "--name", "x"])
    with pytest.raises(SystemExit, match="matches"):
        run.main(["--images", "no-such-image", "--device", "cpu", "--name", "y"])
