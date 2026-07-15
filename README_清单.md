# RLHF 云训练套件 · F盘清单(2026-07-10 打包)

> 用途:5133 若被公司征用,凭此套件在任意租赁 GPU 机(4×5090D)上直接开跑 R4/R5,无需回 5133 取数据。

## 目录结构
```
F:\rlhf_lab_cloud_kit\
├─ data\
│  ├─ dataset_info.json          # LLaMA-Factory 数据集注册表(含 sft_v11_* 条目)
│  ├─ sft_v11\                   # ★最终训练集 v1.1(154,476条,10类满配)
│  │   ├─ train_full.json        #   全量训练(R5用)
│  │   ├─ subset_40k.json        #   4万子集(R4四臂用)
│  │   ├─ dev.json / dev_1k.json #   验证集
│  └─ eval_sets\                 # 评测保留区(禁入训练)
│     ├─ CMB\ cmexam\            #   医学选择题基线(base 73.8%/82.6%)
│     └─ 05_final_v11\           #   test.jsonl + hard_eval.jsonl(分桶评测)
├─ models\Qwen3-8B-Base\         # base 模型(16G;云上亦可 hf-mirror 直拉)
├─ scripts\                      # 全部脚本(训练/评测/vLLM/数据管线/R4四臂生成)
│  ├─ sft_data\make_r4_arms.py   #   R4 四臂数据生成
│  ├─ sft_data\eval_hardeval.py  #   hard_eval 6类分桶评测
│  ├─ eval_mcq.py eval_cmexam.py #   CMB/CMExam 评测
│  └─ serve_vllm.sh chat_cli.py  #   vLLM 部署 + 对话客户端
├─ configs\
│  ├─ r4r5\r4_arm_template.yaml  #   R4 单臂模板(占位__ARM__)
│  ├─ r4r5\r5_final.yaml         #   R5 终版(全量2ep)
│  └─ sft_v1\                    #   R0-R3 扫参 yaml(存档)
└─ docs\
   ├─ EXPERIMENT_LOG.md          # ★全程实验记录(唯一事实来源,§1-17)
   ├─ CLOUD_SETUP_R4R5.md        # ★开机一键流程 + 省钱铁律
   └─ requirements.{train,vllm}.lock  # 环境包快照

## 锁定的生产配置(R1-R3扫参选定)
rank=64  alpha=128  lr=2e-4  lora_target=all  cutoff_len=2048  seed=42
bf16 / cosine / warmup 0.03 / flash_attn=fa2(云上)

## 开机后步骤(见 CLOUD_SETUP_R4R5.md 详版)
1. 机型 4×5090D + cu128 镜像 + 数据盘≥100G
2. 装 llamafactory[torch,metrics]+deepspeed+flash-attn+vllm → smoke → freeze
3. 传本套件到 /root/autodl-tmp/(路径与 yaml 对齐)
4. R4:make_r4_arms.py 生成四臂 → 4卡并行训练 → 分桶评测比 A/B/C/D
5. R5:r5_final.yaml 全量2ep → 交付 adapter(=DPO起点)→ 全套评测+vLLM抽查
6. 关机前回传 adapter+评测产物+requirements.lock

## 尚未包含(体积/时效原因,云上现取)
- LLaMA-Factory 源码:云上 pip 装(版本 0.9.6.dev0)
- 训练产物/adapter:云上生成后回传此处
```
