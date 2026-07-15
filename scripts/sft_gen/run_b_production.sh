#!/usr/bin/env bash
set -euo pipefail

BASE="/data/shenxin/rlhf_lab"
SCRIPT="$BASE/scripts/sft_gen/sft_gap_generator_b.py"
OUT_DIR="$BASE/data/sft_pipeline/generated"
LOG="$OUT_DIR/test_report_explanation.production.log"
PID="$OUT_DIR/test_report_explanation.production.pid"

set -a
source "$BASE/.minimax_env"
set +a

if [[ -f "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "ALREADY_RUNNING pid=$(cat "$PID")"
  exit 0
fi

nohup python3 "$SCRIPT" \
  --target 8000 \
  --workers 2 \
  --per-item-attempts 4 \
  > "$LOG" 2>&1 &

echo "$!" > "$PID"
echo "STARTED pid=$! log=$LOG"
