#!/bin/bash
# 全量正式SFT启动:仅用 GPU1+GPU3(绝不碰0/2)。5090双卡需P2P禁用。
cd /root/autodl-tmp
export PATH=/root/miniconda3/bin:$PATH
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FORCE_TORCHRUN=1
export CUDA_VISIBLE_DEVICES=1,3
/root/miniconda3/bin/llamafactory-cli train configs/r4r5/full_train_2card.yaml
