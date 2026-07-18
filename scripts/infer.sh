#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CHECKPOINT="runs/srcnn_x2_9_1_5/best_psnr.pth"
INPUT="data/Set5/butterfly.png"
SCALE=2
NAME="butterfly"
OUT_DIR="results/${NAME}_x${SCALE}"

mkdir -p "$OUT_DIR"

python infer_srcnn.py \
  --checkpoint "$CHECKPOINT" \
  --input "$INPUT" \
  --scale "$SCALE" \
  --output "$OUT_DIR/${NAME}_srcnn_x${SCALE}.png" \
  --save-ground-truth "$OUT_DIR/${NAME}_ground_truth.png" \
  --save-lr "$OUT_DIR/${NAME}_lr_x${SCALE}.png" \
  --save-nearest "$OUT_DIR/${NAME}_lr_nearest_x${SCALE}.png" \
  --save-bicubic "$OUT_DIR/${NAME}_bicubic_x${SCALE}.png"
