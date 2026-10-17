"""Summarize the Week 8 usability test from the sheets filled in during the sessions (report/notes/usability_test.md).

The folder holds three CSV files (templates: ``experiments/usability_test_template/``):

* ``tasks.csv``: ``participant,task,outcome,seconds,notes``; outcome ``done`` (alone), ``help`` (done after a hint) or ``failed``;
  ``seconds`` from the start of the task to its end (for T1, from the first look at the page to the first finished painting);
* ``sus.csv``: ``participant,q1,...,q10``, the System Usability Scale answers from 1 (strongly disagree) to 5 (strongly agree);
* ``issues.csv``: ``participant,issue,severity``; one row per problem a participant met, severity 1 (cosmetic) to 3 (blocks the task);
  the same wording for the same problem, so that the rows can be counted.

Writes ``summary.md`` next to them: per task the share done alone / with help / failed and the median time, the time to the first
painting, the SUS score of every participant (Brooke 1996: odd items score ``answer - 1``, even items ``5 - answer``, the sum times
2.5) with the mean and standard deviation, and the issues ranked by how many participants met them times their worst severity.

Usage: venv/Scripts/python.exe scripts/usability_summary.py experiments/<date>_usability
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

OUTCOMES = ("done", "help", "failed")


def read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:  # utf-8-sig: Excel's CSV export starts with a BOM
        return [{k.strip(): (v or "").strip() for k, v in row.items()} for row in csv.DictReader(fh) if any((v or "").strip() for v in row.values())]


def sus_score(answers: list[int]) -> float:
    if len(answers) != 10 or any(not 1 <= a <= 5 for a in answers):
        raise ValueError(f"SUS needs ten answers from 1 to 5, got {answers}")
    return 2.5 * sum(a - 1 if i % 2 == 0 else 5 - a for i, a in enumerate(answers))


def summarize(folder: Path) -> str:
    tasks, sus, issues = (read(folder / f"{name}.csv") for name in ("tasks", "sus", "issues"))
    participants = sorted({r["participant"] for r in tasks} | {r["participant"] for r in sus})
    lines = [f"# Usability test: {len(participants)} participants", "", "## Tasks", "",
             "| Task | Participants | Done alone | Done with help | Failed | Median time (s) |", "| :--- | ---: | ---: | ---: | ---: | ---: |"]
    by_task = defaultdict(list)
    for r in tasks:
        if r["outcome"] not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}: {r}")
        by_task[r["task"]].append(r)
    for task in sorted(by_task):
        rows = by_task[task]
        times = [float(r["seconds"]) for r in rows if r["seconds"] and r["outcome"] != "failed"]
        share = {o: sum(r["outcome"] == o for r in rows) for o in OUTCOMES}
        lines.append(f"| {task} | {len(rows)} | {share['done']} | {share['help']} | {share['failed']} | "
                     f"{statistics.median(times):.0f} |" if times else f"| {task} | {len(rows)} | {share['done']} | {share['help']} | {share['failed']} | - |")
    first = [float(r["seconds"]) for r in by_task.get("T1", []) if r["seconds"] and r["outcome"] != "failed"]
    if first:
        lines += ["", f"Time to the first painting (T1): median {statistics.median(first):.0f} s, range {min(first):.0f} to {max(first):.0f} s."]
    scores = {r["participant"]: sus_score([int(r[f"q{i}"]) for i in range(1, 11)]) for r in sus}
    if scores:
        mean = statistics.mean(scores.values())
        sd = statistics.stdev(scores.values()) if len(scores) > 1 else 0.0
        lines += ["", "## System Usability Scale", "", f"Mean {mean:.1f}, standard deviation {sd:.1f} (n = {len(scores)}); "
                  "68 is the commonly reported average of many systems.", "", "| Participant | SUS |", "| :--- | ---: |"]
        lines += [f"| {p} | {s:.1f} |" for p, s in sorted(scores.items())]
    ranked = defaultdict(lambda: [set(), 0])
    for r in issues:
        ranked[r["issue"]][0].add(r["participant"])
        ranked[r["issue"]][1] = max(ranked[r["issue"]][1], int(r["severity"]))
    if ranked:
        lines += ["", "## Issues, most important first", "", "| Issue | Participants | Worst severity | Priority |", "| :--- | ---: | ---: | ---: |"]
        for issue, (who, worst) in sorted(ranked.items(), key=lambda kv: (-len(kv[1][0]) * kv[1][1], kv[0])):
            lines.append(f"| {issue} | {len(who)} | {worst} | {len(who) * worst} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", type=Path)
    args = ap.parse_args(argv)
    text = summarize(args.folder)
    (args.folder / "summary.md").write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
