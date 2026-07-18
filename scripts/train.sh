#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

TRAIN_DIR="data/91-image"
VAL_DIRS=("data/Set5" "data/Set14")
SCALE=2
F1=9
F2=1
F3=5
N1=64
N2=32
INPUT_SIZE=33
EPOCHS=100
REPEAT=100
BATCH_SIZE=64
LEARNING_RATE=1e-3
NUM_WORKERS=4
SAVE_DIR="runs/srcnn_x${SCALE}_${F1}_${F2}_${F3}"

python train_srcnn.py \
  --train-dir "$TRAIN_DIR" \
  --val-dirs "${VAL_DIRS[@]}" \
  --scale "$SCALE" \
  --f1 "$F1" \
  --f2 "$F2" \
  --f3 "$F3" \
  --n1 "$N1" \
  --n2 "$N2" \
  --input-size "$INPUT_SIZE" \
  --epochs "$EPOCHS" \
  --repeat "$REPEAT" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LEARNING_RATE" \
  --num-workers "$NUM_WORKERS" \
  --eval-every 1 \
  --save-dir "$SAVE_DIR"
