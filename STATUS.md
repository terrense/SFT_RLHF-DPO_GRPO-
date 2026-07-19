> **2026-07-19 重大更新：本项目现已冻结/归档（historical），不再是活跃训练项目。**
> 用户明确决定（见 `E:\cmedalign\spec\design.md` "THE ANCHOR" 一节）：论文包
> `E:\cmedalign_paper\main.tex` 才是唯一的科研目标锚点，`E:\cmedalign`
> （OpenRLHF-based）是唯一继续跑训练的活跃项目。本仓库不再产生新的训练 run；它的
> 价值在于**可复用的资产**——已建好的 task_type 分类(10类/146,809条)、清洗去重逻辑、
> 已锁定的 LoRA 配置(rank=64/alpha=128)、已测的 M0 baseline 数字——这些会被吸收进
> `cmedalign` 的数据管线，而不是在这里继续开发。**本项目第一轮的 GRPO
> (CMExam选择题规则奖励) 不等同于论文要求的 GRPO (多轮患者模拟环境+5分量奖励)，
> 不能作为论文的 M3 结果使用**——论文的 GRPO 需要在 `cmedalign` 里重新构建。
> 如果你是新会话打开这个仓库：先去读 `E:\cmedalign\spec\design.md` 和
> `E:\cmedalign\spec\tasks.md`，那里才是当前活跃的工作清单；本文件下面记录的是
> 2026-07-19 更新前的"第二轮规划"，仅作历史参考，不是当前该做的事。
>
> 冷启动读取顺序（历史参考）：`spec/requirements_sft_rigor.md`（要什么，原文）→
> `spec/design.md`（现状 gap 分析 + 决策）→ `spec/tasks.md`（当前可执行清单）→
> 本文件（按时间的详细日志）。`docs/EXPERIMENT_LOG.md` 是**第一轮**
> （已完成的 SFT→DPO→GRPO）的事实来源。

STAGE: [已冻结/历史] 新一轮规划完成，尚未开始执行任何数据/训练操作 —— 现已被
  `cmedalign` 项目取代为活跃工作
STATUS: FROZEN（本仓库不再是活跃训练项目，见上方 2026-07-19 更新）
GPU_GROUP: 本机 RTX 3060 Ti(8GB，仅用于 CPU/tokenizer 级别工作)；真实 LoRA 训练待公司
  "5133" 机器或新租服务器
START/END: 2026-07-19 / 进行中
KNOWN_ISSUES:
  - `internal_seed_flywheel` 脱敏状态未知，见 `BLOCKERS.md` #1 —— 阻塞任何会展示/导出
    该数据具体内容的操作，不阻塞纯统计类审计。
  - 大部分 §17-§45（packing、loss normalization、curriculum、多轮 rollout、工具调用、
    归因实验）在第一轮完全没做，是新一轮的主要工作量所在，见 `spec/tasks.md`。
NEXT_COMMAND: [已改变] 不再是"见 spec/tasks.md Phase 1"——新的下一步是去
  `E:\cmedalign\spec\tasks.md` 的 "Phase 0.0 THE ANCHOR" 和 "Phase 2 — Data" 部分，
  把本仓库的 `05_final_v11/train.jsonl`（排除 internal_seed_flywheel/derived_from_seed
  直到 PII blocker 解决）导入到 cmedalign 的 data/raw/
ETA: 本项目不再有独立执行时间线；工作量已并入 cmedalign 项目的 96 小时窗口（一旦
  该项目的 GPU 服务器就绪）

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

### 2026-07-19（当天晚些时候）— 重大方向调整：论文包才是锚点，本项目冻结
- 用户展示 `E:\cmedalign_paper`（`cmedalign` 项目最初的论文/执行计划来源目录），明确说
  "我们不能完全被我之前的实验带偏了...tool-calling 似乎没价值...我们要统一一个主心骨"。
- 读了 `main.tex` 全文后确认：论文的科研问题（RQ1-RQ4）和六张结果表完全不涉及
  tool-calling、DAgger、agent 动作类型、packing/loss-normalization/curriculum 对比、
  梯度冲突诊断——这些都来自本仓库这份新清单（`spec/requirements_sft_rigor.md`），是
  通用 SFT 工程严谨性清单，不是论文本身要求的。用户据此决定砍掉这些方向。
- **发现一个必须澄清才能继续的冲突**：本项目第一轮的 GRPO 是"CMExam 选择题 + 规则奖励
  (答对=1)"，论文 `main.tex` 描述的 GRPO 是"多轮患者模拟环境 + 5 分量奖励"——两者是
  不同的实验，不能把第一轮结果当作论文的 M3。用户决定：论文的 GRPO 必须在 `cmedalign`
  项目里按论文描述重新构建；本项目的 GRPO 结果作为独立的、更小的 side-result 保留，
  不冒充 M3。
- **用户决定项目归属**：合并为一个活跃项目——`cmedalign`（OpenRLHF-based）是唯一继续
  跑训练的地方；本仓库（`rlhf_lab_cloud_kit`）冻结为历史资产库，不再产生新训练 run。
  可复用资产清单（task_type 数据、清洗去重逻辑、锁定的 LoRA 配置、DPO 双裁判模式）
  已写入 `E:\cmedalign\spec\design.md` 的 "THE ANCHOR" 一节，后续开发去那边跟踪，
  不在本仓库继续新增 `spec/tasks.md` 条目。
- 本文件（`STATUS.md`）和 `spec/` 三份文档保留作为历史存档，不再更新为"当前工作"，
  只在需要追溯"第一轮/第二轮规划到底做过什么"时查阅。
