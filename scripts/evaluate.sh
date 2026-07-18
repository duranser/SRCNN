#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CHECKPOINT="runs/srcnn_x2_9_1_5/best_psnr.pth"
SCALE=2

for DATASET in Set5 Set14; do
  TEST_DIR="data/${DATASET}"
  OUT_DIR="results/${DATASET}_srcnn_x${SCALE}"

  python evaluate_srcnn.py \
    --checkpoint "$CHECKPOINT" \
    --test-dir "$TEST_DIR" \
    --scale "$SCALE" \
    --save-dir "$OUT_DIR" \
    --json-out "$OUT_DIR/metrics.json"
done
