#!/usr/bin/env bash
# Week 1 baseline runner: the original (patched) demo_prog.py, one run per brush, one hardware profile.
#
#   usage: bash scripts/run_original_baselines.sh <cuda|cpu|mps> [image] [experiment_dir_name]
#
# Same protocol as the MacBook run in experiments/week1_baseline.md: 512 px canvas, 500 strokes,
# progressive grid 1x1 .. 5x5, lightweight renderer, pixel loss only, seed 0. Wall time covers the
# whole process (model load, optimization, and the final 500-frame render), as on the MacBook.
# Outputs: experiments/<dir>/results_<profile>.csv, one log, final PNG and stroke .npz per brush.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-cuda}"
IMG="${2:-./test_images/apple.jpg}"
EXP="$ROOT/experiments/${3:-$(date +%F)_week1_baseline_desktop}"
PY="$ROOT/venv/Scripts/python.exe"; [ -x "$PY" ] || PY="$ROOT/venv/bin/python"

SEED=0; STROKES=500; DIVIDE=5; NET=zou-fusion-net-light
declare -A CANVAS_COLOR=( [oilpaintbrush]=white [markerpen]=black [watercolor]=white [rectangle]=black )

mkdir -p "$EXP"
COMMIT=$(git -C "$ROOT" rev-parse --short HEAD)
CSV="$EXP/results_${PROFILE}.csv"
[ -f "$CSV" ] || echo "profile,brush,image,canvas_color,seed,max_strokes,max_divide,net_G,wall_time_s,final_G_loss,final_step_acc,commit,date" > "$CSV"

cd "$ROOT/stylized-neural-painting"
# -1 (not "") is the portable way to hide every GPU: with an empty string torch still
# reports cuda.is_available() == True on Windows while exposing zero devices.
if [ "$PROFILE" = "cpu" ]; then export CUDA_VISIBLE_DEVICES=-1; fi
"$PY" - "$PROFILE" > "$EXP/device_${PROFILE}.txt" <<'EOF'
import sys, os, platform, torch
print("profile", sys.argv[1]); print("torch", torch.__version__); print("python", platform.python_version())
print("platform", platform.platform()); print("cpu_count", os.cpu_count()); print("torch_threads", torch.get_num_threads())
n = torch.cuda.device_count() if torch.cuda.is_available() else 0
print("cuda_available", torch.cuda.is_available()); print("cuda_device_count", n)
if n: print("gpu", torch.cuda.get_device_name(0)); print("cuda", torch.version.cuda); print("cudnn", torch.backends.cudnn.version())
EOF

BASENAME=$(basename "$IMG"); BASENAME="${BASENAME%.*}"
for BRUSH in oilpaintbrush markerpen watercolor rectangle; do
  OUT="./output_${PROFILE}_${BRUSH}"; rm -rf "$OUT"
  LOG="$EXP/${PROFILE}_${BRUSH}.log"
  echo "[$(date +%T)] $PROFILE $BRUSH ..."
  START=$(date +%s.%N)
  "$PY" demo_prog.py --img_path "$IMG" --canvas_color "${CANVAS_COLOR[$BRUSH]}" \
      --max_m_strokes $STROKES --max_divide $DIVIDE --renderer "$BRUSH" \
      --renderer_checkpoint_dir "checkpoints_G_${BRUSH}_light" --net_G $NET \
      --disable_preview --seed $SEED --output_dir "$OUT" > "$LOG" 2>&1
  END=$(date +%s.%N)
  WALL=$(awk -v s="$START" -v e="$END" 'BEGIN{printf "%.1f", e-s}')
  LAST=$(grep "iteration step" "$LOG" | tail -1 || true)
  GLOSS=$(echo "$LAST" | sed -nE 's/.*G_loss: ([0-9.]+).*/\1/p')
  ACC=$(echo "$LAST" | sed -nE 's/.*step_acc: ([0-9.]+).*/\1/p')
  cp "$OUT/${BASENAME}_final.png" "$EXP/${PROFILE}_${BRUSH}_final.png"
  cp "$OUT/${BASENAME}_strokes.npz" "$EXP/${PROFILE}_${BRUSH}_strokes.npz"
  [ -f "$EXP/input_${BASENAME}.png" ] || cp "$OUT/${BASENAME}_input.png" "$EXP/input_${BASENAME}.png"
  echo "$PROFILE,$BRUSH,$IMG,${CANVAS_COLOR[$BRUSH]},$SEED,$STROKES,$DIVIDE,$NET,$WALL,$GLOSS,$ACC,$COMMIT,$(date +%F)" >> "$CSV"
  echo "[$(date +%T)] done in ${WALL}s  G_loss=$GLOSS  step_acc=$ACC"
done
echo "results: $CSV"
