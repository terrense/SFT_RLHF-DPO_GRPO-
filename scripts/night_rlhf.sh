#!/bin/bash
# 夜间全自动:等SFT训完 → 验收 → DPO造数据 → DPO beta扫参训练 → 各版本输出存好。
# 停在GRPO前(留给用户早上)。全程四卡。所有需人工确认的都不阻塞,写日志。
cd /root/autodl-tmp
export PATH=/root/miniconda3/bin:$PATH
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
LF=/root/miniconda3/bin/llamafactory-cli
LOG=outputs/night_rlhf.log
say(){ echo "$(date '+%m-%d %H:%M:%S') $1" >> $LOG; }

say "NIGHT_RLHF_START"

# 0. 等SFT全量训完
while ps aux | grep 'llamafactory-cli train' | grep 'full_train_4card' | grep -v grep >/dev/null; do sleep 120; done
sleep 30
if [ ! -f outputs/sft_final/adapter_model.safetensors ]; then say "ERR_SFT_ADAPTER_MISSING"; exit 1; fi
say "SFT_DONE_adapter_ok"

# 1. SFT验收:五场景 + CMB(GPU0)
CUDA_VISIBLE_DEVICES=0 $PY scripts/sft_data/multi_scene_final.py > outputs/sft_final_scenes.log 2>&1
say "ACCEPT_SCENES_DONE"
CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_mcq.py --model models/Qwen3-8B-Instruct \
  --adapter outputs/sft_final --data data/eval_sets/CMB/CMB-Exam/CMB-val/CMB-val-merge.json \
  --tag sft_final > outputs/cmb_sft_final.log 2>&1
say "ACCEPT_CMB_DONE"

# 2. DPO采样:四卡并行(每卡一分片),用正式SFT模型
mkdir -p data/rlhf/dpo
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i nohup $PY scripts/sft_data/dpo_sample.py --shard $i --nshards 4 --n 2400 --k 4 \
    > outputs/dpo_sample_$i.log 2>&1 &
done
wait
say "DPO_SAMPLE_DONE"

# 3. DPO judge:双API,不占卡(shard0/2用minimax, shard1/3用deepseek 分散限流)
$PY scripts/sft_data/dpo_judge.py --shard 0 --provider minimax  > outputs/dpo_judge_0.log 2>&1 &
$PY scripts/sft_data/dpo_judge.py --shard 1 --provider deepseek > outputs/dpo_judge_1.log 2>&1 &
$PY scripts/sft_data/dpo_judge.py --shard 2 --provider minimax  > outputs/dpo_judge_2.log 2>&1 &
$PY scripts/sft_data/dpo_judge.py --shard 3 --provider deepseek > outputs/dpo_judge_3.log 2>&1 &
wait
cat data/rlhf/dpo/pairs_shard*.jsonl > data/rlhf/dpo/dpo_pairs.jsonl 2>/dev/null
NP=$(wc -l < data/rlhf/dpo/dpo_pairs.jsonl)
say "DPO_JUDGE_DONE pairs=$NP"

# 注册DPO数据集
$PY -c "
import json
p='data/dataset_info.json'; d=json.load(open(p))
d['dpo_pairs']={'file_name':'rlhf/dpo/dpo_pairs.jsonl','ranking':True,'formatting':'sharegpt',
  'columns':{'messages':'conversations','system':'system','chosen':'chosen','rejected':'rejected'}}
json.dump(d,open(p,'w'),ensure_ascii=False,indent=2); print('registered dpo_pairs')
" >> $LOG 2>&1

# 4. DPO beta扫参:0.1/0.3/0.5 顺序训练(四卡DDP每个),各存adapter+样例输出
for BETA in 0.1 0.3 0.5; do
  sed "s/__BETA__/$BETA/g" configs/r4r5/dpo_train.yaml > configs/r4r5/dpo_beta$BETA.yaml
  FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3 $LF train configs/r4r5/dpo_beta$BETA.yaml \
    > outputs/dpo_train_beta$BETA.log 2>&1
  say "DPO_TRAIN_beta$BETA_DONE"
  # 该beta版本的样例输出(GPU0),供早上人工选
  CUDA_VISIBLE_DEVICES=0 DPO_ADAPTER=outputs/dpo_beta$BETA $PY scripts/sft_data/dpo_sample_probe.py \
    > outputs/dpo_probe_beta$BETA.log 2>&1
  say "DPO_PROBE_beta$BETA_DONE"
done

say "NIGHT_RLHF_COMPLETE_停在GRPO前_等用户选beta"
