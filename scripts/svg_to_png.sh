#!/usr/bin/env bash
# Rasterize the hand-written report SVGs to PNG with headless Chrome (no extra Python dependency).
#   usage: bash scripts/svg_to_png.sh [file.svg ...]      (default: every SVG in report/figures)
# The width and height attributes of the SVG set the window size; the PNG is written next to the SVG.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CHROME=""
for c in "/c/Program Files/Google/Chrome/Application/chrome.exe" \
         "/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" \
         "/c/Program Files/Microsoft/Edge/Application/msedge.exe" \
         "$(command -v google-chrome || true)" "$(command -v chromium || true)"; do
  [ -x "$c" ] && CHROME="$c" && break
done
[ -n "$CHROME" ] || { echo "no Chrome/Edge found; install one or use an SVG viewer to export the PNG" >&2; exit 1; }

FILES=("$@")
[ ${#FILES[@]} -gt 0 ] || FILES=("$ROOT"/report/figures/*.svg)

for SVG in "${FILES[@]}"; do
  SVG="$(cd "$(dirname "$SVG")" && pwd)/$(basename "$SVG")"   # Chrome needs an absolute file:// URL
  W=$(grep -o 'width="[0-9]*"' "$SVG" | head -1 | tr -dc '0-9')
  H=$(grep -o 'height="[0-9]*"' "$SVG" | head -1 | tr -dc '0-9')
  PNG="${SVG%.svg}.png"
  URL="file:///$(cygpath -m "$SVG" 2>/dev/null || echo "$SVG")"
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \
            --default-background-color=FFFFFFFF --virtual-time-budget=2000 \
            --screenshot="$(cygpath -w "$PNG" 2>/dev/null || echo "$PNG")" \
            --window-size="${W},${H}" "$URL" >/dev/null 2>&1
  [ -f "$PNG" ] && echo "wrote $PNG ($(du -h "$PNG" | cut -f1), ${W}x${H} at 2x)" || echo "FAILED $SVG" >&2
done
