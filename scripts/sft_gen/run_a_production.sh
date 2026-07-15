#!/usr/bin/env bash
set -euo pipefail

BASE="/data/shenxin/rlhf_lab"
SCRIPT="$BASE/scripts/sft_gen/sft_gap_generator.py"
OUT_DIR="$BASE/data/sft_pipeline/generated"
LOG="$OUT_DIR/risk_redflag_safety_refusal.production.log"
PID="$OUT_DIR/risk_redflag_safety_refusal.production.pid"

set -a
source "$BASE/.minimax_env"
set +a

if [[ -f "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "ALREADY_RUNNING pid=$(cat "$PID")"
  exit 0
fi

nohup python3 "$SCRIPT" generate-a \
  --target 10000 \
  --workers 4 \
  --per-item-attempts 4 \
  > "$LOG" 2>&1 &

echo "$!" > "$PID"
echo "STARTED pid=$! log=$LOG"
