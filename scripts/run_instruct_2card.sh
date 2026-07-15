#!/bin/bash
# Instruct验证双卡启动:仅用 GPU1+GPU3(绝不碰GPU0/2)。5090无P2P需 NCCL_P2P_DISABLE=1。
cd /root/autodl-tmp
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FORCE_TORCHRUN=1
export CUDA_VISIBLE_DEVICES=1,3
/root/miniconda3/bin/llamafactory-cli train configs/r4r5/instruct_val_2card.yaml
