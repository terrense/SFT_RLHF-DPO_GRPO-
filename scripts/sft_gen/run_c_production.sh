#!/usr/bin/env bash
set -euo pipefail

SCRIPT=/data/shenxin/rlhf_lab/scripts/sft_gen/sft_gap_generator_c.py
OUT_DIR=/data/shenxin/rlhf_lab/data/sft_pipeline/generated
LOG=${OUT_DIR}/hospital_policy_rag_qa.production.log
PID=${OUT_DIR}/hospital_policy_rag_qa.production.pid

mkdir -p "${OUT_DIR}"
nohup python3 "${SCRIPT}" --target 5000 --workers 2 --per-item-attempts 4 --fresh >"${LOG}" 2>&1 &
echo $! >"${PID}"
echo "started hospital_policy_rag_qa production pid=$(cat "${PID}") log=${LOG}"
