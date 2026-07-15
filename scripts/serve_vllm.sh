#!/bin/bash
# vLLM 服务:base 模型 + SFT adapter 同端口双模型(OpenAI 兼容 API)
# 用法: bash serve_vllm.sh [adapter路径,默认=扫参最优 r2_lr2e-4]
# 切模型不用重启:请求里 model=base 或 model=sft_v1 即可
LAB=/data/shenxin/rlhf_lab
ADAPTER=${1:-$LAB/outputs/sft_v1/r2_lr2e-4}
export CUDA_VISIBLE_DEVICES=0

exec $LAB/vllm_env/bin/vllm serve $LAB/models/Qwen3-8B-Base \
  --served-model-name base \
  --enable-lora \
  --lora-modules sft_v1=$ADAPTER \
  --max-lora-rank 64 \
  --chat-template $ADAPTER/chat_template.jinja \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.35 \
  --host 127.0.0.1 --port 8000
