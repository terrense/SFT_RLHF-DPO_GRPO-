> 冷启动读取顺序：`spec/requirements_sft_rigor.md`（要什么，原文）→ `spec/design.md`
> （现状 gap 分析 + 决策）→ `spec/tasks.md`（当前可执行清单）→ 本文件（按时间的详细日志）。
> `docs/EXPERIMENT_LOG.md` 是**第一轮**（已完成的 SFT→DPO→GRPO）的事实来源，不要混淆：
> 本文件记录的是**第二轮**（严谨性补强）的进展。

STAGE: 新一轮规划完成，尚未开始执行任何数据/训练操作
STATUS: RUNNING（规划阶段 PASS；具体审计脚本待写）
GPU_GROUP: 本机 RTX 3060 Ti(8GB，仅用于 CPU/tokenizer 级别工作)；真实 LoRA 训练待公司
  "5133" 机器或新租服务器
START/END: 2026-07-19 / 进行中
KNOWN_ISSUES:
  - `internal_seed_flywheel` 脱敏状态未知，见 `BLOCKERS.md` #1 —— 阻塞任何会展示/导出
    该数据具体内容的操作，不阻塞纯统计类审计。
  - 大部分 §17-§45（packing、loss normalization、curriculum、多轮 rollout、工具调用、
    归因实验）在第一轮完全没做，是新一轮的主要工作量所在，见 `spec/tasks.md`。
NEXT_COMMAND: 见 `spec/tasks.md` Phase 1（本机可执行的审计脚本）
ETA: 规划阶段已完成；执行阶段视 GPU 可用时间而定，暂无固定 deadline（不同于 cmedalign
  项目的 96 小时窗口——两者是独立项目，互不冲突）

---

## Log

### 2026-07-19 — 新一轮规划：读现状 + 建 spec/ 文档
- 用户展示 `E:\rlhf_lab_cloud_kit` 目录，随后追加一份更严格的"医疗 Agent SFT 实验实现
  要求与避坑清单"（75 条），要求先把 task_type 定好再谈数据配比，并明确说"我们准备要
  真的开始了"。
- 实地读取仓库确认：这不是从零开始——第一轮 SFT→DPO→GRPO 已经真实跑完
  （`docs/EXPERIMENT_LOG.md`，`README.md`），框架是 LLaMA-Factory，基座 Qwen3-8B，
  硬件是公司容器"5133"的 H20 + 云上 4×RTX 5090D（不是这台 Windows 机器）。
- 确认 task_type **已经建立**：10 类，146,809 条真实计数（`data/eval_sets/05_final_v11/
  train.jsonl`），规则分类（`label_03.py`），已知局限是"分诊科室欠准"（README 自己写的）。
- **关键发现**：`export_v11.py` 在导出最终 LLaMA-Factory 训练 json 前会 `pop` 掉
  task_type/id 字段——task_type 的配比权重确实在上游（`sample_05_v11.py` 的 TARGETS）
  生效了，但训练期已经拿不到 task_type，无法做新清单要求的 source/task-specific
  validation loss 追踪。这是用户"task_type 得确定好"这句话背后的具体技术风险，已写入
  `spec/design.md`。
- **发现现有 PII 处理的具体空白**（见 `BLOCKERS.md` #1）：`clean_02.py` 对种子业务数据
  的 PII 命中只打标签不处理，且正则只覆盖电话/QQ/邮箱，不覆盖姓名/身份证/住址/病历号。
  已作为待用户确认的 blocker 记录，不擅自处理。
- 逐条对照新清单 75 条要求 vs 现有代码（读了 `check_mask.py`, `label_03.py`,
  `dedup_04.py`, `clean_02.py`, `convert_01.py`, `export_v11.py`, `sample_05_v11.py`），
  写成 `spec/design.md` 的 gap 分析表（✅已做 / 🟡部分做 / ❌未做）。结论：LoRA
  超参消融(rank/lr/target_modules)、数据配比四臂、DPO beta 扫参、倒豆子诊断——已扎实
  做过，不必重做；packing 对比、loss normalization 对比、多轮自由 rollout、工具调用
  数据/评测、DAgger 闭环、归因实验（数据消融/counterfactual/leave-one-source-out）——
  完全没做，是新一轮的真正工作量。
- 意外发现：这台 Windows 本机其实有一张 RTX 3060 Ti（8GB）——此前一直以为完全没有本地
  GPU。不足以做真实 8B LoRA 训练（该项目自己实测需要~27GB），但足够做 tokenizer/
  数据级别的验证工作。用户确认：先用它做本机准备工作，真实训练仍等公司机器或新租服务器。
- 建立 `spec/requirements_sft_rigor.md`（新清单原文）、`spec/design.md`（gap 分析+决策）、
  `BLOCKERS.md`（PII 脱敏待确认）、本文件，作为断电/新窗口后的冷启动入口，与 cmedalign
  项目采用同一套文档模式（用户明确要求过的做法）。
