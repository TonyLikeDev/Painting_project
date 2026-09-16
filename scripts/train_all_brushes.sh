#!/usr/bin/env bash
# Train the four neural renderers one after another on the desktop GPU (Week 6).
#   usage: bash scripts/train_all_brushes.sh [--light] [extra train_renderer args]
# Each run writes checkpoints/<brush>[_light]/ and experiments/<date>_train_<brush>/ (curve CSV, val images).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/venv/Scripts/python.exe"; [ -x "$PY" ] || PY="$ROOT/venv/bin/python"
for BRUSH in oilpaintbrush watercolor markerpen rectangle; do
  echo "== training renderer for $BRUSH"
  "$PY" -m neural_painter.pipeline.train_renderer --brush "$BRUSH" "$@"
done
