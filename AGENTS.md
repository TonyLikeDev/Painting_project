# AGENTS.md

All instructions for AI coding agents working in this repository live in
**`CLAUDE.md`** (project purpose, layout, environment facts, conventions, workflow).
Read it, then open `ROADMAP.md` for the current status and the next open item.

Short version of the rules:

- `RESEARCH_PLAN.md` is the plan, `ROADMAP.md` is the live status. Tick items in
  `ROADMAP.md` and add a change-log line every time something is finished.
- Use the project virtual environment: `venv/Scripts/python.exe` on Windows,
  `venv/bin/python` on macOS.
- Run `pytest` before code under `neural_painter/` is committed. Never commit or push yourself and never add AI co-author lines; see the Git rules in CLAUDE.md.
- Keep the procedural rasterizer pixel-identical to the original 2021 code.
- Stroke parameters live in `[0, 1]` with layouts defined in `neural_painter/configs/*.yaml`.
- Measurements go to `experiments/`, prose goes to `report/`, both seeded and dated.
