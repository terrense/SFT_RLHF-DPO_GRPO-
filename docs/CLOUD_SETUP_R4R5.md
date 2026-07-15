# AutoDL 4×5090D 开机后一键流程(R4+R5)——付费计时,照此最快

## 0. 机型/镜像(下单时)
- 4×RTX 5090D-32G,cu128 镜像(torch2.8/cu128),数据盘扩 100-200GB
- 5090=sm_120,torch 必须 cu128+;镜像不对一切白搭

## 1. 环境(约 15-20 min)——先 smoke 再冻结
```
pip install -U llamafactory[torch,metrics] deepspeed flash-attn vllm -i https://mirrors.aliyun.com/pypi/simple
# smoke: 4卡 NCCL all-reduce / flash-attn 前向 / 单步训练 各过一遍
pip freeze > requirements.lock
```

## 2. 上传数据+模型(从 5133 或 F: 备份)
需要传到 /root/autodl-tmp/:
- data/: dataset_info.json + sft_v11/ + sft_v11_arms/(R4四臂)
- models/Qwen3-8B-Base(或云上 hf-mirror 直拉:export HF_ENDPOINT=https://hf-mirror.com)
- scripts/: eval_mcq.py, eval_cmexam.py, eval_hardeval.py, serve_vllm.sh
- configs/: r4_arm_template.yaml, r5_final.yaml
- 评测保留集: CMB/ CMExam/ 05_final_v11/{test,hard_eval}.jsonl

## 3. R4 四臂并行(4卡各一臂,~1 epoch)
```
for i, arm in [(0,arm_a_open),(1,arm_b_seed),(2,arm_c_mixed),(3,arm_d_seed2x)]:
  sed "s/__ARM__/$arm/" r4_arm_template.yaml > r4_$arm.yaml
  CUDA_VISIBLE_DEVICES=$i nohup llamafactory-cli train r4_$arm.yaml &
```
跑完:每臂在 CMB240 + hard_eval 分桶评测,比 A/B/C/D。核心问题=种子值多少分/开源稀释否/过采样有效否。

## 4. R5 终版(全量2ep,主卡;余卡可跑多seed副本)
```
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train r5_final.yaml
```
产出 = 交付 SFT adapter，即 DPO 起点。跑完全套评测(CMB/CMExam/hard_eval + vLLM 交互抽查)。

## 5. 省钱铁律
- 所有数据/配置/评测脚本已在 5133 备好,开机只做"传+跑",不在付费机上现写
- R4 四臂务必并行(4卡),不要串行
- 训练完立刻 vLLM 抽查确认可用，再决定是否续租跑 DPO/GRPO
- 关机前:adapter + 全部评测产物 + requirements.lock 回传 5133/F:
```
