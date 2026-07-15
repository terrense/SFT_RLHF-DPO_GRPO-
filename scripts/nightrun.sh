#!/bin/bash
# 夜间全自动(两卡 GPU1/3;GPU0/2 用户在用,绝不碰):
# 等R5训完 → 倒豆子测试 → CMB评测 → DPO两卡采样 → MiniMax judge打分 → 合并偏好对。
cd /root/autodl-tmp
PY=/root/miniconda3/bin/python
LOG=outputs/nightrun.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say(){ echo "$(date '+%m-%d %H:%M:%S') $1" >> $LOG; }

say "NIGHTRUN2_START(2card GPU1/3)"

# 1. 等 R5 训练结束
while ps aux | grep 'llamafactory-cli train' | grep -v grep >/dev/null; do sleep 120; done
say "R5_TRAINING_ENDED"

# 2. 验证 adapter
if [ ! -f outputs/r5/sft_v11_final/adapter_model.safetensors ]; then
  say "ERROR_R5_ADAPTER_MISSING"; exit 1
fi
say "R5_ADAPTER_OK"

# 3. 倒豆子验收(GPU3,R5训完即空)
CUDA_VISIBLE_DEVICES=3 $PY scripts/sft_data/test_r5_stop.py > outputs/r5_stoptest.log 2>&1
say "STOPTEST_DONE"

# 4. R5 CMB 知识分(GPU3)
CUDA_VISIBLE_DEVICES=3 $PY scripts/eval_mcq.py --model models/Qwen3-8B-Base \
  --adapter outputs/r5/sft_v11_final \
  --data data/eval_sets/CMB/CMB-Exam/CMB-val/CMB-val-merge.json \
  --tag r5_final > outputs/cmb_r5.log 2>&1
say "R5_CMB_DONE"

# 5. DPO 采样(两卡:GPU1=分片0, GPU3=分片1)
CUDA_VISIBLE_DEVICES=1 nohup $PY scripts/sft_data/gen_dpo_pairs.py \
  --stage sample --shard 0 --nshards 2 > outputs/dpo_sample_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup $PY scripts/sft_data/gen_dpo_pairs.py \
  --stage sample --shard 1 --nshards 2 > outputs/dpo_sample_1.log 2>&1 &
wait
say "DPO_SAMPLE_DONE"

# 6. MiniMax judge 打分(不占卡,两分片并行)
$PY scripts/sft_data/gen_dpo_pairs.py --stage judge --shard 0 > outputs/dpo_judge_0.log 2>&1 &
$PY scripts/sft_data/gen_dpo_pairs.py --stage judge --shard 1 > outputs/dpo_judge_1.log 2>&1 &
wait
say "DPO_JUDGE_DONE"

# 7. 合并偏好对,报告数量
cat data/rlhf/dpo/pairs_shard*.jsonl > data/rlhf/dpo/dpo_pairs.jsonl 2>/dev/null
NP=$(wc -l < data/rlhf/dpo/dpo_pairs.jsonl 2>/dev/null)
say "DPO_PAIRS_TOTAL=$NP"
say "NIGHTRUN2_COMPLETE"
