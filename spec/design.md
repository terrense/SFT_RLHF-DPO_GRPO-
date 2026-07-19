# design.md — 现状 gap 分析 + 决策记录

配套 `spec/requirements_sft_rigor.md`（要什么）。本文件回答"现在到底有什么、缺什么"，
基于对仓库实际文件的直接读取（不是猜测）。`spec/tasks.md` 是由此推出的可执行清单。

## 一句话现状（2026-07-19）

第一轮 SFT→DPO→GRPO 已完整跑通并有真实结果（`docs/EXPERIMENT_LOG.md`，`README.md`）。
框架是 **LLaMA-Factory**（不是 cmedalign 项目用的 OpenRLHF），基座 **Qwen3-8B**，
第一轮训练硬件是公司容器"5133"上的 H20（共享物理卡）+ 云上 4×RTX 5090D，**不是**这台
Windows 本机。本机（这台电脑）只有一张 **RTX 3060 Ti（8GB）**，不足以做真实 8B LoRA
训练（该项目自己实测 LoRA 训练显存需求 ~27GB），但足够做纯数据/tokenizer 级别的准备工作。

新一轮要做的，是把 `spec/requirements_sft_rigor.md` 的 75 条要求应用到"下一版"实验上——
**延续本仓库**（复用 LLaMA-Factory 管线、已锁定的 LoRA 配置、已建好的 10 类 task_type
数据、已测的 CMB/CMExam baseline），只把真正缺的部分补上，而不是重新发明。

## 已验证：task_type 分类现状（直接读取 `data/eval_sets/05_final_v11/train.jsonl`）

146,809 条记录，10 个 task_type，全部有真实计数（不是占位符）：

| task_type | 数量 | 主要来源 |
|---|---:|---|
| symptom_consultation | 34,305 | 混合 |
| health_encyclopedia_qa | 26,920 | Huatuo26M-Lite 为主 |
| triage_guidance | 17,149 | 规则挖掘（见下方局限） |
| pre_consultation_multiturn | 14,718 | internal_seed_flywheel（真实业务种子） |
| test_report_explanation | 14,664 | 混合 |
| chronic_disease_management | 12,269 | 规则挖掘 |
| risk_redflag_safety_refusal | 9,803 | 混合 |
| medication_guidance_safe | 7,352 | 规则挖掘 |
| conversation_summary_structured_output | 4,885 | 种子反向构造 |
| hospital_policy_rag_qa | 4,744 | 混合 |

来源分布：`med_zh_real` 56,578 · `Huatuo26M-Lite` 36,570（开源）· `internal_seed_flywheel`
14,718（**真实业务数据**）· `derived_from_seed` 14,433（种子衍生）·
`gen_minimax_m3` 12,937（MiniMax-M3 合成）· `DISC-Med-SFT` 9,438（开源）·
`Chinese-medical-dialogue` 1,733（开源）· `shibing624-finetune-zh` 402（开源）。

**task_type 分类方式**（读 `scripts_cloud/sft_data/label_03.py`）：正则规则分类
（挂科室/慢病名+管理语境/用药问句+警示语），种子数据（`internal_seed_flywheel`）透传
不重分类，其余按规则挖掘。**已知局限**（`README.md` §7 原话）："分诊科室欠准、跑题不纠偏、
首轮闲聊会幻觉"——即 triage_guidance 的规则分类本身准确率有限，这是新一轮要正视而不是
忽略的已知缺陷，不是这次新发现的问题。

**关键发现：最终训练 JSON 里 task_type 已被丢弃。** `export_v11.py` 第 62 行
`s.pop("task_type", None)` 在导出 LLaMA-Factory sharegpt 格式前主动删除了 task_type/id，
只保留 `conversations` + `system`。这正是用户提出"task_type 得确定好，不然数据怎么配比
都是空的"这句话背后的真实风险：**配比权重确实已经在上游（05 采样阶段）按 task_type
生效**（`sample_05_v11.py` 里每个 task_type 有明确 TARGETS 目标数），**但训练时（LLaMA-Factory
读取 sharegpt json 时）已经没有 task_type 字段了**，所以无法做该清单第 26/33/55 条要求的
"source-specific / task-specific validation loss 实时追踪"——这不是"没定"，而是"定了但在
最后一步被丢弃，导致训练期可观测性丢失"。这是新一轮要修的具体、可定位的问题，而不是
从零开始定义 task_type。

## 逐条 gap 分析（对照 `spec/requirements_sft_rigor.md`）

标记：✅已做且验证过 / 🟡部分做到（有已知局限）/ ❌未做（真正的新工作）

- **§1 Baseline 全套评测**：🟡 已有 CMB(240题)/CMExam(选择题) baseline，但清单要求的
  "预问诊多轮/危险信号识别/科室推荐/工具调用/JSON输出/通用能力回归/多轮自由rollout/
  延迟显存"这些维度大部分**没有**做过。第一轮的评测集中在医学选择题准确率 + 对话胜率
  （MiniMax 当裁判）。
- **§2 单变量原则**：✅ R0-R3 扫参（rank/lr/target_modules/seed）确实是单变量对照，
  做得对。
- **§3 可复现性记录**：🟡 `configs_cloud/` 保存了所有 yaml，`docs/EXPERIMENT_LOG.md`
  记录了过程，但**没有**看到系统化保存 git commit hash / CUDA-PyTorch-Transformers
  版本 / GPU型号 到每个实验目录（`docs/requirements.train.lock` 有版本快照，但不是
  per-experiment）。
- **§4-5 数据 schema + 分源标注**：✅ `05_final_v11/*.jsonl` 阶段的 schema 已经很接近
  清单要求（`id/schema_version/source/task_type/sub_task_type/messages/metadata{department,
  risk_level,red_flags,evidence_required,is_multiturn,language,source_quality,license,
  dedup_hash}`）。**缺**：`quality_score`（数值型质量分,现在只有 `source_quality`
  high/medium/low 三档）、`token_length`、`supervised_token_length`、`has_tool_call`
  （工具调用数据目前完全不存在）、`synthetic_or_real`（现在要从 source 名称推断，没有
  显式字段）、`patient_or_case_id`（用于 patient-level split，目前没有）。
- **§6 Patient/case-level split**：❌ 目前是按 task_type 分层的 95/2.5/2.5 **样本级**
  切分（`sample_05_v11.py`），md5(id) 决定性排序，但**没有** patient/case-level 或
  source-level 或 time-based 切分保证同一病例不同改写不跨分区。真实业务种子数据
  （`internal_seed_flywheel`）如果存在同病例多条衍生，需要专门检查。
- **§7 格式检查**：🟡 `clean_02.py` 有基础清洗，需要进一步核对是否覆盖清单列出的全部
  12 类格式问题（连续同角色/非法Unicode/超长样本等），**待逐条核对**（未逐行读完
  clean_02.py 全部 114 行）。
- **§8 精确+近似去重**：✅ `dedup_04.py`（跨源精确 hash 去重）+ `sample_05_v11.py`
  内的 MinHash（5字 shingle, 32 perm, LSH 8band×4）近似去重，**已做**，但只在"入选池"
  内部去重，不是清单要求的"与 benchmark 的近似去重"（那是下一条）。
- **§9 Benchmark contamination**：🟡 `convert_01.py` 有 `EVAL_RESERVED` 源级排除
  （CMB/CMExam/MLEC-QA-Benchmark/CBLUE 数据集本身不进训练池），这是好的第一道防线，
  **但不是**清单要求的 item-level 检查（训练数据里的其他来源是否恰好包含与 CMB/CMExam
  题目相同或语义改写的内容）。**这是真正的缺口，需要新建**（可以直接复用
  `cmedalign` 项目里已经写好并测试过的 `cmedalign.data.dedup.cross_split_contamination`
  逻辑，是同一个问题）。
- **§10 合成数据溯源**：🟡 `gen_minimax_m3` 来源已标注，但未看到逐条记录
  temperature/sampling参数/生成时间/是否人工审核。
- **§11 脱敏**：❓ 未验证。`internal_seed_flywheel`（真实业务数据）是否已做 PII 脱敏
  需要专门确认——这是隐私相关问题，按项目原则应该在确认前默认当作**未脱敏处理**，
  优先级最高，需要用户明确告知脱敏状态。
- **§12-14 Chat template / mask / 可视化工具**：🟡 `check_mask.py` 已存在且验证了
  assistant-only mask 的边界打印逻辑，**但**该脚本写死指向 `Qwen3-8B-Base` +
  `template="qwen3"`，而 README 自己的结论是 **Base 模型 + thinking 模板会导致"倒豆子"，
  必须换 Instruct + `qwen3_nothink` 模板**——所以这个检查脚本本身可能是用旧
  （有问题的）配置写的，**需要用 Instruct + nothink 模板重新跑一遍并核对**，不能假设
  它还成立。另外，cmedalign 项目里已经用真实 Qwen3-8B tokenizer 验证过三个具体坑
  （`return_assistant_tokens_mask` 静默失效、think-stub 只出现在最后一轮、
  `apply_chat_template(tokenize=True)` 默认返回 dict）——**这些坑对 Qwen3-8B-Instruct
  同样适用**，LLaMA-Factory 的内部实现是否已经正确处理，需要专门验证，不能假设
  LLaMA-Factory 自动做对了。
- **§15 小数据过拟合测试**：❓ 未在已读脚本中发现专门的 8-32 条过拟合测试，
  R0 smoke（`sft_v1_smoke2k` 2000条）不算"小数据过拟合"，量级不同、目的不同。
- **§16 倒豆子回归测试**：✅ `check_imend.py` / `check_think.py` / `diag_stop.py` 存在，
  且 README 明确记录了真实发生过的倒豆子问题和根因（Base+thinking模板）——这是本项目
  最扎实的一块，第一轮真的踩过这个坑并系统排查过。
- **§17-20 Packing 实验**：❌ 未发现任何 packing 相关配置或对比，第一轮训练看起来是
  non-packing（`cutoff_len=2048`，常规做法），符合清单 §17"第一版默认关闭 packing"的
  建议，但清单 §18 要求的三组对比实验没有做。
- **§21-24 长度分布/truncation**：❓ 未验证是否有系统化长度分布统计（P50/P90/P99）
  和 truncation 记录。
- **§25-27 Loss normalization 对比**：❌ 未发现 token-level vs sample-level loss
  对比实验，LLaMA-Factory 默认是 token-level mean，没有做 sample-level 对照。
- **§28-30 多源采样策略**：✅ 这是本项目做得**最好**的一块——`sample_05_v11.py` 的
  TARGETS 机制本质就是清单 §28/29 要求的 source/task weight + capped sampling，
  R4 四臂实验（natural/seed/mixed/seed-2x）基本对应清单 §29 的 E1/E2/E3。
- **§31-32 Curriculum 对比**：❌ 未发现 Staged Training 与 Mixed Training 的对比实验，
  第一轮看起来是一次性混合训练（Mixed），没有做分阶段课程对比。
- **§33-35 梯度冲突/source-specific loss**：❌ 如上所述，task_type 在导出时被丢弃，
  没有 source-specific validation loss 追踪，也没有梯度冲突诊断。
- **§36-38 高风险数据/hard negatives**：🟡 `risk_redflag_safety_refusal`
  task_type 存在（9,803条）且有 REDFLAG 正则识别，但未验证是否专门构造了 hard negative
  对比对（"普通胸痛 vs 心源性胸痛"这类），也未看到误报率（false escalation rate）评测。
- **§39-42 多轮 rollout / agent 动作类型**：❌ 第一轮评测是"对话质量胜率"（裁判对比
  静态回答），**不是**自由多轮 rollout（模型用自己产生的历史继续对话）。没有 ASK/
  ANSWER/CALL_TOOL/ESCALATE/STOP/CLARIFY/RECOVER 这类显式动作类型标注。
- **§43-45 工具调用数据**：❌ 完全没有——当前 10 个 task_type 里没有一个是工具调用，
  `has_tool_call` 字段也不存在。这是最大的新增数据缺口。
- **§46-51 LoRA/训练参数消融**：✅ rank(8/16/32/64)、lr(5e-5/2e-4/4e-4)、
  target_modules(attn-only/all) 都做过（R0-R3），这是本项目第二好的一块。
- **§52-54 Batch/有效token 记录**：❓ 未验证是否记录了 effective batch size 分解
  和 per-source tokens per step。
- **§55-57 训练可观测性**：🟡 有训练日志（`analysis/archive/` 11条 loss 原始日志），
  但不确定是否达到清单要求的粒度（source-specific loss、生成 probe 定期跑）。
- **§58-61 分层评测体系**：🟡 医学知识/对话质量两块有，安全/工具/通用能力回归/
  多轮 rollout 评测基本没有独立做。
- **§62-65 归因实验**：❌ 未发现数据消融（去掉某类数据重训对比）、counterfactual
  评测、组合泛化测试、leave-one-source-out 测试。
- **§66-68 灾难性遗忘**：❓ 未验证是否有通用能力回归测试（README 提到局限但没提
  是否系统测过遗忘）。
- **§69-70 异常排查机制**：🟡 有真实的排查案例（倒豆子问题的 5 步排查过程写在
  README/EXPERIMENT_LOG 里），但看起来是一次性人工排查，没有沉淀为可复用的
  checklist/脚本。

## 决策记录

- **项目归属**：延续 `rlhf_lab_cloud_kit`（LLaMA-Factory + 已锁定配置 + 已有数据），
  不新建项目、不迁移到 OpenRLHF/cmedalign。（用户 2026-07-19 决定）
- **本轮算力**：这台 Windows 机器的 RTX 3060 Ti（8GB）现在就能用于纯数据/CPU 级别的
  准备工作（schema 补全、格式检查、去重审计、contamination 检查、chat template/mask
  验证、可能的极小规模过拟合测试）；真正的 LoRA 训练仍需等公司"5133"机器或新租的
  GPU 服务器。（用户 2026-07-19 决定）
- **不重新发明已做好的部分**：R0-R3 LoRA 超参消融、R4 数据配比四臂、DPO beta 扫参、
  倒豆子诊断脚本——这些已经做过，新一轮工作应该**复用其结论**（例如 rank=64,
  alpha=128, lr=2e-4 已经是扫参选出的锁定配置），除非清单的新要求（比如
  sample-level loss、packing）要求重新对照，否则不要重跑一遍已有答案的实验。
- **PII 脱敏状态未知，按最谨慎假设处理**：在用户明确告知 `internal_seed_flywheel`
  真实业务数据的脱敏状态之前，任何涉及展示/导出该数据具体内容的工具都应该默认视为
  敏感数据处理（不打印到日志、不进 git）。

## 尚待用户确认/提供的信息（写入 `BLOCKERS.md`）

见 `BLOCKERS.md`。
