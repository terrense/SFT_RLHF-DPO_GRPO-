#!/bin/bash
cd /root/autodl-tmp/distributed_lab
export PATH=/root/miniconda3/bin:$PATH
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3
LF=/root/miniconda3/bin/llamafactory-cli
for Z in zero0 zero2 zero3; do
  sed "s#__DS__#ds_configs/$Z.json#" bench_train.yaml > run_$Z.yaml
  # 后台采样峰值显存
  ( peak=0; for i in $(seq 1 200); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1); [ "$m" -gt "$peak" ] && peak=$m; echo $peak > logs/peak_$Z.txt; sleep 2; done ) &
  POLL=$!
  t0=$(date +%s)
  $LF train run_$Z.yaml > logs/train_$Z.log 2>&1
  t1=$(date +%s)
  kill $POLL 2>/dev/null
  echo "$Z 用时$((t1-t0))s 峰值显存$(cat logs/peak_$Z.txt)MB" >> logs/bench_summary.txt
  echo "=== $Z done ==="
done
echo ALL_DS_BENCH_DONE
