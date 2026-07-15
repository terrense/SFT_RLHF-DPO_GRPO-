# 医疗大模型全链路实验记录（单一事实来源）

> 目标：把 Qwen3-8B 从 base 模型，经 **CPT → SFT → DPO → RLHF(GRPO/PPO)** 全链路打通，
> 每个阶段都用**同一批测试题**量化"能力提升了多少"，得到可对外汇报的数字。

---

## 1. 硬件 / 模型 / 环境

| 项 | 值 |
|---|---|
| 模型 | Qwen3-8B-Base（80亿参数，未做过任何指令微调的"原始"模型） |
| 模型权重大小 | 16 GB（80亿 × 2字节 bf16） |
| GPU | NVIDIA H20-3e，**共享物理卡**（HAMi给140GB配额，但物理卡被多租户共用，需等空闲窗口） |
| 显存占用 | 推理~17GB；**LoRA训练~27GB**；（全参微调需~128GB，故用LoRA） |
| 训练吞吐 | 实测 ~2200 token/秒 |
| 框架 | LLaMA-Factory |
| 实验目录 | `/data/shenxin/rlhf_lab/`（5133容器） |
| 训练时长公式 | 时长 ≈ 总token ÷ 2200 |

---

## 2. 用了哪些数据（完整清单）

| 数据集 | 来源 | 形态 | 用在哪个阶段 | 规模 | token量 |
|---|---|---|---|---|---|
| **51万医疗问诊**(med_zh) | 你本地F盘 | 问答对(instruction/output) | **SFT** | 44.6万条 | ~65M |
| **医学教科书**(shibing624 medical_book) | hf-mirror开源 | raw连续文本 | **CPT-真实** | 8,475段 | ~11M |
| **医学百科**(shibing624 encyclopedia) | hf-mirror开源 | raw连续文本 | **CPT-真实** | 564MB | ~123M |
| **合成HG-LN-2026**(MiniMax生成) | 我用M3生成 | 虚构规范raw文本 | **CPT-注入演示** | 851段 | ~1M |
| **CMB选择题** | hf-mirror开源 | 医学单选题+答案 | **评测·医学知识** | 240单选 | — |
| **注入探针** | 我据虚构规范生成 | 问答+关键词 | **评测·注入效果** | 108题 | — |
| **reward偏好对** | shibing624开源 | chosen/rejected | DPO备用 | 几千对 | — |

> 注意区分：**CPT-真实**（真医学知识，123M+11M）和 **CPT-注入演示**（虚构知识，1M）是两个不同实验，目的不同。

---

## 3. 评测体系（测什么、怎么测）—— 全程同一批题，可比

| 指标 | 测的是什么能力 | 怎么测 | 用什么数据 | 题量 |
|---|---|---|---|---|
| **医学选择题准确率** | 真实医学知识水平 | 选项"对数概率打分"，选概率最高的选项对答案 | CMB | 240单选 |
| **知识注入率** | CPT有没有把虚构规范灌进去 | 模型生成回答→命中关键词算对 | 注入探针 | 108 |
| **对话质量胜率** | 回答/对话能力（SFT/DPO后） | MiniMax当裁判，两两对比答案谁更好 | med_test留出 | 待建≥100 |

---

## 4. 实验记录（按时间）

### 实验0 · base 基线（起点/对照组）
- **模型**：原始 Qwen3-8B-Base，未做任何训练
- **测了**：医学选择题 + 知识注入率
- **结果**：医学选择题 **73.8%**（177/240）；知识注入率 **13.0%**（14/108）
- **解读**：base 预训练就吃过大量医学，所以真医学题已达73.8%；对虚构的HG-LN-2026一无所知，13%纯属5选1瞎蒙的地板。

### 实验1 · CPT 合成知识注入（证明"CPT能写入知识"的小型对照实验）
- **为什么做**：base已会73.8%真医学，直接上真实语料CPT看不出变化→改用**虚构知识**，让CPT的效果"可视化"
- **数据**：合成851段HG-LN-2026 | **配置**：LoRA rank32, 5轮, lr1e-4 | **耗时**：10.6分钟(~1M token)
- **测了**：医学选择题 + 知识注入率
- **结果**：知识注入率 **13.0% → 90.7%**（98/108，**+77.7**）；医学选择题 **73.8% → 70.4%**（−3.4）
- **解读**：
  - 注入率暴涨 → **证明CPT确实把新知识写进了权重**（核心结论✓）
  - 医学选择题轻微下降 → **灾难性遗忘**（窄语料猛跑5轮、无通用数据回放导致"偏科"）；这是CPT的经典权衡，解法是混通用数据回放。
- **结论**：CPT机制验证成功。这是"演示"，非正式CPT。

### 实验2+ · 正式全量训练（待跑）
| 阶段 | 数据 | 预计耗时 | 目的 |
|---|---|---|---|
| CPT-真实全量 | 134M token真实医学语料(教科书+百科) | ~16小时 | 灌**真实**医学知识 |
| SFT | 51万医疗问答 | ~一晚 | base学会"按指令规整作答/对话" |
| DPO | 偏好对 | ~半天 | 让回答更符合偏好 |
| GRPO/PPO | 可验证奖励(选择题对错) | ~一晚 | RL闭环 |

### ★ RLHF 落实计划（已与用户确认，务必执行，不停在理论）
1. **DPO（离线）**：偏好数据 = 用 MiniMax 给 SFT 模型的多个采样打分排序**自造** + shibing624 reward 集补充；`stage: dpo`，单卡 LoRA，**离线不需 vLLM**；验收 = MiniMax 裁判 **DPO vs SFT 胜率%**。调 `pref_beta`（0.1 vs 0.5）看偏移。
2. **GRPO（在线 RL，RLVR）**：奖励 = CMB 医学选择题**答对=1**（可验证，省奖励模型）；**选 GRPO 不选 PPO**——单卡+共享卡放不下 PPO 的4模型，GRPO 砍掉 critic 只需3模型；验收 = **reward 曲线 + KL 曲线**一起看（观察 reward hacking）。
3. **推训分离实操**：路线A 先用 LLaMA-Factory GRPO（HF rollout）跑通拿结果；路线B 再用 **OpenRLHF/veRL**（vLLM rollout + 权重同步）体会工业级推训分离。**两个都做**，原理+基础设施经验都拿到。

---

## 5. 结果总表（持续更新）

| 阶段 | 医学选择题% | 知识注入率% | 对话胜率 |
|---|---|---|---|
| base基线 | 73.8 | 13.0 | — |
| CPT-合成(演示) | 70.4 | 90.7 | — |
| CPT-真实全量 | **73.3%**(176/240,≈base无变化) | — | — |
| SFT | 待跑 | — | 待跑 |
| DPO | 待跑 | — | 待跑 |
| GRPO/PPO | 待跑 | — | 待跑 |

---

## 6. 可复现配置（实验1 CPT-合成，实测值，非记忆）

| 字段 | 值 |
|---|---|
| base model | `Qwen/Qwen3-8B-Base`，本地 `models/Qwen3-8B-Base`，model_type=qwen3，5 shards |
| tokenizer | 与 base 同目录，一致 |
| chat template | CPT(pt阶段)**不套对话模板**(raw续写)；SFT 将用 `qwen3` |
| cutoff_len | 合成CPT=1024；真实全量=2048 |
| packing | true |
| batch | 合成: per_device=4 × grad_accum=4 = **global 16**；真实全量: 2×8=16 |
| LoRA | **r=32, alpha=64, dropout=0.0, rslora=F, dora=F** |
| target_modules | `all`(展开为 q/k/v/o/gate/up/down_proj 全部线性层) |
| learning_rate | 合成=1e-4；真实全量=1e-5 |
| scheduler | cosine |
| warmup_ratio | 合成=0.05；真实=0.03 |
| epochs | 合成=5；真实=1 |
| optimizer | adamw_torch（LLaMA-Factory 默认，⚠️**当时未显式设**） |
| precision | bf16 |
| flash attention | ⚠️**未显式启用，状态待确认**（正式run需显式开 fa2 并记录） |
| gradient checkpointing | true |
| seed | ⚠️**未显式设 → 默认 42**（正式run必须显式固定） |
| git commit (LLaMA-Factory) | `9c0b4b3` |
| dataset hash (md5前12) | train_zh=`3e3c13fdc876`(446189) / synth_corpus=`c3eab73b143f`(851) / reward=`3c94cff30191`(3800) / CMB-val=`a3e210531eec` |
| 产物 | `outputs/cpt_synth_lora/adapter_model.safetensors` |
| 训练统计 | train_loss=0.9582, runtime=635.7s, total_flos=6.5e16, 2.179 samples/s |

> ⚠️ **复现性缺口(诚实记录)**：seed/optimizer 用默认未显式固定、flash_attn 状态未确认。**正式全量run起,这些必须显式设定并记录**(seed=42, optim=adamw_torch, flash_attn=fa2)。

## 7. 评测协议（固定，可复现）

**医学选择题(CMB)**：
- 方法 = 选项**对数概率打分**(读最后位置对A/B/C/D…的logprob取最大)，**确定性、无temperature/采样**。
- 为何不用"生成+regex"：logprob 对 base/微调模型都公平，避免长篇解释导致抽取不稳定。
- eval set = CMB-val **240 单选题**(held-out)；多选题(40道)已剔除。
- 泄漏检查：训练数据(med_zh咨询QA/教科书/百科/合成)**均不含CMB**；base 预训练是否见过 CMB 无法控制(公共基准固有风险)。

**知识注入探针**：
- 方法 = 生成(`do_sample=False` 贪心，确定性)，`max_new_tokens=80`，prompt=`"问题：{q}\n回答："`。
- 抽取规则 = 关键词命中(每题关键词列于 probe.jsonl)，命中任一即对，**抽取失败/未命中=错**。
- 108 题全部来自虚构规范，零泄漏。

**对话胜率(待做)**：MiniMax 裁判两两对比，**AB双向各judge一次消除position bias**，打平计0.5，固定 judge prompt，记录 seed。

## 8. 数据集质量与长度评估

| 数据集 | 关键统计(字符) | 质量结论 |
|---|---|---|
| SFT med_zh 回答 | p50=159 p90=298 p99=390 max=778；空=0 极短=2 | **干净**；单条token p99=327 → **cutoff 1024 偏大,可降512提速** |
| SFT med_zh 问题 | p50=41 mean=52 | 正常 |
| 合成CPT 段落 | p50=468 mean=466 max=3817 | 长度一致,正常 |
| **DPO reward chosen** | **均178字** | 🔴 |
| **DPO reward rejected** | **均40字** | 🔴 **chosen/rejected 长度比=4.43x,严重长度偏置** |

> 🔴 **重要发现**：shibing624 reward 偏好对存在**严重长度偏置**(chosen 比 rejected 长 4.4 倍)。直接用于 DPO 会让模型学到"**越长越好**"而非"内容更优"。**对策**：DPO 改用 MiniMax **自造长度受控**的偏好对；或采用 SimPO 等带长度归一的损失。→ 这印证了"自造偏好"的必要性。

## 9. 实验1 严谨结论（替代"机制验证成功"的口语版）

> 在 Qwen3-8B-Base 上，对约 **1M tokens** 的虚构医学知识文本做 LoRA-CPT(r32/α64/lr1e-4/5ep)后，
> 模型对该虚构知识集合的**关键词命中率从 13.0% 提升到 90.7%(98/108)**，
> 说明 continued LM training 可在小规模设置下**写入新知识**。
> 同时 CMB 医学选择题(240单选, logprob打分)从 **73.8%(177) 变为 70.4%(169)**，仅差 7 题；
> **因样本量仅 240 且未做 McNemar / paired-flip 检验，目前只能认为通用医学能力未见提升、可能存在轻微干扰，不能断言显著遗忘**。
> 后续需通过**真实医学 CPT + general replay + 更大 held-out 集**进一步验证。

### 实验2(真实全量 CPT）结论
> 在 Qwen3-8B-Base 上对 **~130M token 真实医学语料**(教科书+百科)做 1 轮 LoRA-CPT(lr1e-5, r32, 18h20m, train_loss=2.19)后，
> CMB 从 **73.8%(177/240) 变为 73.3%(176/240)**，仅差 1 题 → **统计等价，无提升亦无下降**。
> 结合实验1(合成虚构知识 13%→90.7%)，得出关键结论：
> **CPT 能否提升，取决于语料是否含"模型尚不知道的知识"；对已很强的 base + 常见领域，真实 CPT 可能无明显收益。**
> ∴ 按循证策略(能力未下降)→ SFT 可从 CPT 续或从 base 起(二者等价，倾向从 base 更干净)。

---

## 10. 规范与待办（每次实验必须遵守，可复现性硬要求）

1. **训练配置必须显式设定并记录**：`seed=42`、`optim`(adamw_torch)、`flash_attn=fa2`、以及 §6 全部字段。不再用隐式默认。
2. **数据先检查再训练**：任何数据集进训练前，先跑质量+长度检查(空/极短/重复/长度分布 p50/p90/p99/leakage)，结果写入本文档。
3. **偏好数据必须查长度偏置**：DPO 不用 chosen/rejected 长度比悬殊的数据(如 shibing624 reward 4.4x)；改用**长度受控**的自造偏好，或 SimPO 等长度归一损失。
4. **评测可复现 + 统计检验**：MCQ 用 logprob 打分；固定解码参数(do_sample=False)与抽取规则；**保存每题预测**，前后对比用 **McNemar 检验**给 p 值，不只看准确率差。
5. **cutoff_len 按 p99 设**：如 SFT(med_zh) p99=327 token → 用 512(提速且不截断)。

## 11. 计划：完整版全量 SFT+RLHF 实验

- **时间**：**2026-07-03（周五）晚开始**（本次 CPT 链路实验完全结束后）。
- **目标**：完整版 **全量 SFT + RLHF**，**数据量比肩大厂**(SFT 数十万~百万级，偏好/RL 数万~十万级)。
- **前置硬要求**：**开训前先严格检查数据集质量**(质量/长度/去重/泄漏)——这是第一步，不达标不开训。
- **执行**：严格遵守上面 §10 全部规范(显式seed/flash、长度受控偏好、McNemar、配置全记录)。
- **具体数据来源与规模**：届时规划(可能含更大医疗指令集、自造长度受控偏好、可验证奖励集扩充)。

## 12. SFT 数据工程(分阶段 20k→60k→120k→300k,2026-07-07 启动)

目标:面向医疗 Agent(导诊/预问诊/问答/报告解读/健教/RAG/结构化摘要)构建**受控、任务均衡、安全感知**的 SFT 数据集,不盲目堆量。规范:10 类 task_type + 4 类 sub_task_type,统一 canonical schema(messages 超集),中间产物 01_raw_converted → 02_cleaned → 03_labeled → 04_deduped → 05_sampled_balanced → train/dev/test/hard_eval。脚本位于 `scripts/sft_data/`,报告位于 `data/sft_pipeline/reports/`。

### 12.0 评测保留区(红线,先于一切采样)
**CMB / CMExam / MLEC-QA / CBLUE 一律不得进入 SFT 训练集**(管线黑名单硬编码)。理由:CMB 73.8% / CMExam 82.6% 是本项目的基线数字,一旦混入训练,前后对比全部作废。CMExam 同时是 GRPO 的 RLVR 奖励题库,其 RL 训练用子集与评测子集必须**先切分、不重叠**(seed=42 固定切分,切分文件落盘)。

### 12.1 Stage 1:种子集画像(2026-07-07,报告 stage1_seed_profile.json)
对象:`data/medicine_dataset/pre_consultation_multiturn.cleaned.jsonl`(业务种子集,飞轮生成 20,001 → Data-Juicer 清洗后 **15,463**)。

**格式校验:零异常**——bad JSON 0 / 缺字段 0 / 空输出 0 / 角色顺序异常 0 / 重复 id 0 / 重复 dedup_hash 0。全部 assistant 先手开场(符合院内预问诊场景)。

**分布**:
- num_turns:4 轮为主(10,701 条),1~7 轮;清洗移除的 4,538 条中 **4,396 条是单轮弱样本**(单轮 4841→445),多轮结构基本无损
- 科室:24 个,分布均匀(top 881 / 尾部 460+),无严重偏科
- triage_level:2/3/4 = 1,964/4,070/9,429;**无 1 级(急诊)**
- 说话风格 persona:8 种均匀(各 ~1,900),语言多样性好
- 长度:token_estimate p50=583 / p90=1054 / **p99=1532** / max=2549 → 该任务 cutoff_len 需 **2048**(不是 med_zh 的 512!);末轮 assistant 回复很短(p50=68 字,推荐语模板化)

**发现的缺口(Stage 2/3 必须补)**:
1. **scene 全部 normal、risk_level 全部 medium、无急诊 triage**——飞轮这批未生成 emergency/红旗场景,`risk_redflag_safety_refusal` 覆盖为 0,需用飞轮生成器加 emergency 场景 + teacher 生成安全拒绝样本
2. 末轮推荐语高度模板化(``根据您的病史,请前往...``),`recommendation_generation` 子任务的输出多样性不足
3. 可派生 sub_task 资产:next_question_generation 可派生 61,077 个训练点、recommendation_generation 15,460 个

### 12.2 开源池盘点(dataset_download_report.json,均已在 5133 本地)
| 数据集 | 行数 | 用途定位 |
|---|---|---|
| DISC-Med-SFT | 464,898 | 训练池:symptom_consultation / 多轮(messages 格式,已知 ppl 基准 6.93) |
| Chinese-medical-dialogue | 792,099 | 训练池:symptom_consultation(带科室→可派生 triage_guidance) |
| Huatuo26M-Lite | 177,703 | 训练池:health_encyclopedia_qa(带科室+疾病标签) |
| shibing624/medical | 2,443,484 | 训练池(谨慎):百科类;含 Huatuo 派生子集,与 Huatuo26M-Lite 跨源重叠风险高;reward 部分有 4.4x 长度偏置,不用 |
| med_zh(51万真实问诊) | 442,415(清洗后) | 训练池:symptom_consultation 真实语料 |
| CMB 280,913 / CMExam 68,119 / MLEC-QA 13,624 / CBLUE 89,768 | — | **评测保留区,禁入训练** |

缺口类(开源池基本没有,需生成/挖掘):test_report_explanation、hospital_policy_rag_qa、conversation_summary_structured_output、risk_redflag_safety_refusal、medication_guidance_safe(部分可从 QA 池规则挖掘)。conversation_summary 可从种子集多轮对话反向构造(对话→结构化 JSON 摘要),hospital_policy_rag 可用合成 KB 方法造(HG-LN-2026 经验复用)。

### 12.3 管线执行记录与 v1 数据集(2026-07-07 完成)

**管线各级产量**(脚本 scripts/sft_data/,全部确定性可复现,md5(id) 采样/切分,无随机态):
| 级 | 动作 | 产量 |
|---|---|---|
| 01 convert | 6源→canonical(messages超集);shibing 195万 hash 采样至 30万;CMD 用 gb18030 解码 | 2,170,693 |
| 02 clean | 规则过滤(广告SEO/无警示开药/无检查定论/PII/超短长/危险建议);种子只标记不删 | 2,076,589(删 93,624,原因记 removed_samples_reason.csv;修复"不一定是"误伤 bug) |
| 03 label | 规则精修 task_type(triage/慢病/用药挖掘)+ risk/red_flags 标注 | 挖出慢病 40,107、用药安全 20,409、triage 9,066;高危红旗标注 38,212 |
| 04 dedup | 全局精确去重,质量优先保留 | 删 68,414(CMD 内部样例重复 67,300 为主) |
| 05 sample | 二次挖掘(报告解读 99,583 候选、急诊升级 1,702)→ 质量分排序采样 → MinHash(5字shingle/32perm/LSH8×4,J≈0.7)贪心近重去重 → 种子派生 → 分层切分 | **140,845** |

**v1 配比(锚点=种子 15,463,×0.5154 缩放 300k 模板)**:
symptom 36,080 ✓ / encyclopedia 28,349 ✓ / triage 18,040 ✓(挖掘~8k+种子派生~10k)/ report_explanation 15,463 ✓(池内挖掘,超预期)/ pre_consultation 15,463 ✓(种子全量)/ chronic 12,886 ✓ / medication_safe 7,732 ✓ / summary_structured 5,154 ✓(种子meta真值反向构造)/ **risk_redflag 1,678(缺 8,631,Codex 生成中)** / **hospital_rag 0(缺 5,154,Codex 生成中)**。

**来源占比**:med_zh 真实 59,506(42%)、Huatuo 34,675、种子+派生 30,664(21.8%)、DISC 13,760、CMD 1,816、shibing 424。业务种子未被稀释;CMD/shibing 占比低是质量分排序的结果(科室unknown罚分+低质先验)。

**切分**:train 133,889 / dev 3,502 / test 3,454(md5(id) 分层 95/2.5/2.5)+ hard_eval 1,495(test 内高危/分诊/报告/多轮/用药)。产物 05_final/{05_sampled_balanced,train,dev,test,hard_eval}.jsonl + train_sharegpt.json。

**遗留事项(开训前必须处理)**:
1. 两个生成类缺口等 Codex 交付(generated/ 目录),交付后重跑 05 合并
2. **sharegpt 导出的种子对话是 assistant 先手**,LLaMA-Factory 要求 human/gpt 交替开头,训练注册前需处理(方案:开场白并入 system 或补患者接入占位轮)——未验证前不得开训
3. 近重去重只做了入选集贪心(全池 MinHash 太贵),train/dev/test 跨切分近重已由全局贪心保证
4. 数据集尚未注册进 LLaMA-Factory dataset_info.json;cutoff_len 需 2048(种子 p99=1532)
﻿
## 13. 下一阶段:LoRA 扫参对照实验全案(2026-07-07 定稿,详细版同步在本地永久记忆)

---

Qwen3-8B-Base 鍖荤枟 SFT **LoRA 鎵弬瀵圭収瀹為獙鍏ㄦ**(鏁版嵁=v1 140,845,瑙?[[sft-data-pipeline-status]];灞?[[fullscale-sft-rlhf-plan-0703]];鏈哄櫒=AutoDL 4脳RTX 5090D-32GB,cu128 闀滃儚)銆?
## 浜旇疆璁捐(鍏?17-19 涓缁?run,涓€娆″彧鍔ㄤ竴涓彉閲?
- **R0 鍐掔儫(0.5h)**:2k 瀛愰泦 debug run銆傞獙璇?env smoke(flash-attn/鍔犺浇/淇濆瓨缁窇)銆乴oss 涓嬮檷銆佽瘎娴嬮摼璺鍒扮銆?*assistant 鍏堟墜淇鐢熸晥**銆備笉杩囧叧涓嶈繘 R1銆?- **R1 rank 鎵?~2.5h)**:r鈭坽8,16,32,64},alpha=2r,lr=1e-4,40k 鍒嗗眰瀛愰泦,1 epoch,4 鍗″苟琛屻€?- **R2 lr 鎵?~2.5h)**:lr鈭坽5e-5,1e-4,2e-4,4e-4} @ 鏈€浼?r銆?- **R3 妯″潡+鏂瑰樊(~2.5h)**:attention-only vs all-linear 脳2 + 鏈€浼橀厤缃?seed 43/44(娴?run-to-run 鏂瑰樊,缁欏悗闈㈡墍鏈夌粨璁洪厤"鍣０搴?)銆?- **R4 鏁版嵁 A/B/C/D(鍏ㄩ噺 1 epoch,~8h,鏍稿績涓氬姟闂)**:
  - ARM-A 绾紑婧?鍘绘帀绉嶅瓙+娲剧敓 30,664)
  - ARM-B 绾瀛?娲剧敓
  - ARM-C 瀹屾暣 v1 娣峰悎 140,845(鍩虹嚎)
  - ARM-D 娣峰悎+绉嶅瓙 2脳 杩囬噰鏍?  鍥炵瓟:涓氬姟绉嶅瓙鍊煎灏戝垎?寮€婧愪細涓嶄細绋€閲?杩囬噰鏍锋湁娌℃湁鐢?
- **R5 缁堢増(~12h)**:鏈€浼橀厤缃?鏈€浼橀厤姣?2 epochs,packing,浜у嚭浜や粯 adapter(=DPO 璧风偣);绌洪棽鍗″悓鏃惰窇澶?seed 鍓湰銆?
## 璇勬祴鍗忚(姣忎釜 run 鍥哄畾涓嶅彉,姣忛棰勬祴钀界洏)
dev loss/ppl;CMB 240 + CMExam 3000(閫夐」 logprob,闃茬伨闅鹃仐蹇?vs base 鍜岀浉閭婚厤缃仛 McNemar);hard_eval 1,495 鍒嗘《鎸囨爣:triage 绉戝鍛戒腑鐜?meta 鐪熷€?銆乻ummary JSON 鍚堟硶鐜?瀛楁F1銆佺孩鏃楀崌绾?recall(鍏抽敭璇嶅垽瀹?銆侀闂瘖/鎶ュ憡瑙ｈ璐ㄩ噺(MiniMax-M3 judge,鍥哄畾 prompt,娓╁害0);瑙ｇ爜 greedy銆乵ax_new_tokens 512銆俢utoff_len=2048銆傚叏閮?yaml 鏄惧紡:seed/optimizer/flash_attn=fa2/lr_scheduler銆?
## GPU/鎴愭湰浼扮畻(瀹炴祴澶栨帹)
- 鍏ㄩ噺 train 133,889 鏉?鈮?5,200 涓囧瓧绗?鈮?**40-45M token/epoch**;40k 瀛愰泦 鈮?12M
- 鍚炲悙:H20 瀹炴祴 2,200 tok/s;5090D 淇濆畧 **1,200-2,000 tok/s** 鈫?鍏ㄩ噺 epoch 6-10h,瀛愰泦 run 1.7-2.8h(R0 瀹炴祴鍚庝慨姝ｆ墍鏈夋帓鏈?
- 鏄惧瓨:鏉冮噸 16G + LoRA 浼樺寲鍣?<1G + 婵€娲?2048, grad-ckpt, micro-bs 2-4)鈮?24-28G / 32G 鉁?OOM 鍏滃簳 micro-bs=1(QLoRA 鍙綔鏈€鍚庢墜娈?浼氭敼鍙樺姣斿熀鍑?
- 澧欓挓 30-40h(2-3澶?,鏁存満 楼12-16/h 鈫?**楼400-650,棰勭畻涓婇檺 楼800**
- 纾佺洏:妯″瀷16G+鏁版嵁2G+20涓猘dapter+璇勬祴浜х墿 鈮?60G 鈫?**鎵╁ 100-200G**

## 寮€鏈哄墠娓呭崟(涓嶅畬鎴愪笉绉熸満,绉熸満=寮€濮嬭璐?
1. **淇?sharegpt assistant 鍏堟墜**(绉嶅瓙寮€鍦虹櫧骞跺叆 system 鎴栬ˉ鎮ｈ€呭崰浣嶈疆)+ 娉ㄥ唽 dataset_info.json + LLaMA-Factory 鍔犺浇 dry-run
2. 鐢熸垚 40k 鍒嗗眰瀛愰泦 + A/B/C/D 鍥涗釜鏁版嵁鍙樹綋鏂囦欢
3. 鍏ㄩ儴 run 鐨?yaml 棰勫厛鍐欏ソ(鍛藉悕 r1_rank8.yaml...),璇勬祴鑴氭湰(eval_mcq/eval_cmexam + 鏂板啓 hard_eval 璇勬祴鍣?鎵撳寘
4. 鏁版嵁涓婁紶璺緞:5133鈫扚:(宸插浠?sft_v1_dataset)鈫扐utoDL;妯″瀷璧?hf-mirror 鐩存帴鎷?5. AutoDL 鐜:cu128 闀滃儚 + llamafactory/flash-attn/vllm 鈫?smoke 鍏ㄨ繃 鈫?pip freeze > requirements.lock
6. Codex 鐢熸垚绫?redflag 8,631 + hospital_rag 5,154)鑻ュ湪绉熸満鍓嶄氦浠樺垯閲嶈窇 05 鍚堝苟;鍚﹀垯 v1 鍏堣,鐢熸垚绫昏繘 v1.1 澧為噺杞?
## 鐘舵€?2026-07-07)
鏂规宸插畾绋垮苟鍐欏叆 EXPERIMENT_LOG.md 搂13銆侰odex 骞惰鐢熸垚涓?MiniMax,鎱?銆傜瓑寰?鐢ㄦ埛绉熸満 + Codex 璇曚骇璐ㄦ銆俁0 鍚炲悙瀹炴祴鍚庢墍鏈夋椂闂翠及绠楄鏇存柊鍥炴湰鏂囦欢銆?

### 13.1 R0-R3 扫参结果(2026-07-08 收官,10/10 run 完成,5133 共享 H20,零租机成本)

**全量结果表**(40k 分层子集,1 epoch,cutoff 2048,sdpa,dev_1k 评 loss,CMB240 评知识保持;base CMB=73.8%):

| run | rank | lr | target | seed | dev loss | CMB240 | 时长 |
|---|---|---|---|---|---|---|---|
| r1_rank8 | 8 | 1e-4 | all | 42 | 1.2574 | 72.1% | 2.45h |
| r1_rank16 | 16 | 1e-4 | all | 42 | 1.2522 | 71.7% | 2.46h |
| r1_rank32 | 32 | 1e-4 | all | 42 | 1.2468 | 71.2% | 2.45h |
| r1_rank64 | 64 | 1e-4 | all | 42 | 1.2411 | 71.7% | 2.47h |
| r2_lr5e-5 | 64 | 5e-5 | all | 42 | 1.2492 | 71.2% | 2.47h |
| **r2_lr2e-4 ★** | 64 | 2e-4 | all | 42 | **1.2380** | 72.1% | 2.45h |
| r2_lr4e-4 | 64 | 4e-4 | all | 42 | 1.2524 | **68.3%** | 2.47h |
| r3_attnonly | 64 | 2e-4 | q/k/v/o | 42 | 1.2493 | 71.2% | 2.23h |
| r3_seed43 | 64 | 2e-4 | all | 43 | 1.2389 | 71.7% | 2.49h |
| r3_seed44 | 64 | 2e-4 | all | 44 | 1.2394 | 72.9% | 2.49h |

**噪声底(seed 42/43/44 同配置)**:dev loss 极差 0.0014(std≈0.0007);CMB240 极差 1.2pp(~3题)。判读标准:dev loss 差异 >0.003、CMB 差异 >2.5pp 才算真效应。

**结论(全部超出噪声,可信)**:
1. **rank 单调有益,64 未饱和**:8→64 每翻倍 dev loss 降 ~0.005(≈7σ)。rank64 的 adapter 仅 ~330MB,性价比仍高;rank128 可作 R5 前的可选探索,预期收益递减。
2. **lr 甜点 = 2e-4,U 型曲线完整**:5e-5 欠学(1.2492)→ 2e-4 最优(1.2380)→ 4e-4 变差(1.2524)。**关键发现:4e-4 的 CMB 暴跌至 68.3%(-5.5pp,≈13题,远超噪声)——lr 过大在 dev loss 上只付出 +0.014,但灾难性遗忘代价巨大**。启示:选 lr 必须同时看域内 loss 和域外知识保持,只看 loss 会低估伤害。
3. **all-linear 优于 attn-only**:1.2380 vs 1.2493(Δ0.011≈16σ),多挂 MLP 层明确值得;attn-only 仅快 10%,不值。
4. **知识保持整体健康**:除 lr4e-4 外,所有配置 CMB 落在 71.2-72.9%(噪声带),相对 base 平均 -1.5~-2pp 轻微遗忘,SFT 换来的行为能力值这个价;后续 R5 全量 2 epoch 需复测确认不恶化。

**锁定 R4/R5 生产配置**:`rank=64, alpha=128, lr=2e-4, lora_target=all, cutoff_len=2048, seed=42, bf16, cosine, warmup 0.03`(AutoDL 上 flash_attn 改 fa2)。

**过程坑点记录**:①dev_1k 首跑忘注册 dataset_info(重试自愈);②共享卡邻居进程(70GB)导致评测 CUDA OOM 静默失败——评测须错峰/等余量,队列等卡逻辑已改为"余量>35GB"而非"占用<20GB";③pkill -f 会自杀 SSH 会话(匹配自身命令行);④队列断点续跑机制实战有效(中途换血零损失)。

**产物**:10 个 adapter 在 outputs/sft_v1/<run>/;sweep_results.jsonl(含重启回放的重复行,以本表为准);全部 yaml 在 configs/sft_v1/auto/。最优 adapter(r2_lr2e-4)已备份 F:\rlhf_lab_backup\。

## 14. vLLM 部署 + 半成品模型交互测试(2026-07-08)

### 14.1 vLLM 环境与服务(推理链路首次打通)
- **独立 venv**(遵"清洗/训练/推理分离"):`/data/shenxin/rlhf_lab/vllm_env`(python3 -m venv,系统 py3.12),**vLLM 0.24.0**,阿里源装,冻结 `docs/requirements.vllm.lock`。装机耗时~20min(vLLM 全家桶含自带 torch+CUDA 库 ~8-10GB;samba 目录无 pip 缓存权限,少缓存加速但不影响结果)。
- **服务脚本** `scripts/serve_vllm.sh`:单卡起 base + LoRA adapter **双模型同端口**(OpenAI 兼容,127.0.0.1:8000):`vllm serve Qwen3-8B-Base --served-model-name base --enable-lora --lora-modules sft_v1=<adapter> --max-lora-rank 64 --gpu-memory-utilization 0.35 --max-model-len 4096`。请求里 model=base / sft_v1 即时切换,不重启。首次启动有 torch.compile 编译 2-4min,之后走缓存。占显存 ~50GB。
- **客户端** `scripts/chat_cli.py`(纯 stdlib;本地版备份 F:\rlhf_lab_backup\chat_cli.py):预问诊场景(种子同款 system + 导医开场白),支持 /model /reset /raw /temp;后加 frequency_penalty=0.5 + stop=["\n患者"] 抑制复读。

### 14.2 Windows 中文乱码坑(环境层,非模型问题)
- plink 直连服务器跑客户端,Windows 控制台 GBK 与服务器 UTF-8 冲突 → 中文输入被当 UTF-8 误解 → vLLM 返回 **HTTP 400**;输出也全乱码(GBK 显示 UTF-8 字节)。`chcp 65001` 可缓解但不稳。
- **正解 = SSH 端口转发,客户端跑在本地**:窗口1 `plink -P 5133 -pw Test@123 -L 8000:127.0.0.1:8000 root@172.24.27.11` 建隧道;窗口2 本地 `python F:\rlhf_lab_backup\chat_cli.py`。中文全程走本地 python,零转码。以后云服务器同理(隧道到本地聊)。

### 14.3 半成品模型交互测试的重要发现(为什么必须 R5 + DPO)
拿**扫参最优 adapter(r2_lr2e-4,40k×1epoch 半成品)**做交互冒烟。base vs sft_v1 同主诉("头晕+眼前一黑")对比:
- **SFT 确实学到业务形态**:选项列表式提问、"参考信息:+鉴别诊断"、"```建议前往XX诊室```"模板、晕厥前兆→升级急诊(神经内科)的分诊判断——base 完全没有,证明 SFT 有效、部署链路通。
- **但暴露三类严重退化**(dev loss 1.238 / CMB 72% 完全测不出,只有交互才现形):
  1. **长序列解码崩坏**:后段混入泰文/英文乱码碎片、句子结构瓦解、`### 最终答案` 无限复读。→ Qwen3-Base 多语言底子在**严重欠训练**(40k×1ep)时的退化。
  2. **多轮一次性倒出、不打停止符**:一个 turn 里把后续多轮提问+最终推荐全生成完,不交还控制权。
  3. 段落/推荐语循环复读。

### 14.4 关键排查:label mask 验证(用户质疑"是不是没做 -100 mask")
用户怀疑数据处理错误(user/system 未 mask)。**写 `scripts/sft_data/check_mask.py` 直接调 LF 数据管线取多轮样本逐 token 验证,结论:mask 完全正确**。
- 样本#658(4 轮,325 token):**253 token 参与 loss(label!=-100)/ 72 token 被 mask(-100)**。
- 分段精确:每个 `<|im_start|>user…<|im_end|>` 段 = [M] masked;每个 assistant 回复段(含 `<think>\n\n</think>` 开头、`<|im_end|>` 结尾)= [T] 计 loss。边界严丝合缝。
- **∴ "多轮一次性输出" 根因不是 label 处理,而是:①欠训练导致 im_end 停止符生成能力没练稳(数据里 im_end 完整存在,模型没学牢)②种子数据单样本即含多轮结构(模型忠实模仿"一口气说多轮")。**

### 14.5 结论与归属表
| 现象 | 根因 | 数据的锅? | 解法 |
|---|---|---|---|
| 泰文乱码/无限复读/崩句 | 40k×1ep 严重欠训练 | 否 | R5 全量 2ep |
| 多轮一次倒出/不停 | im_end 欠稳(欠训练)+ 单样本含多轮 | 否(mask 已验证正确) | R5 全量 2ep;残留交 DPO |
| label -100 mask | — | **已验证正确** | 无需改 |

**核心教训**:①扫参副产品(40k/1ep)是半成品,**本就不能用于对话**,只用于超参排序 + 部署链路验证,不能据此判断数据质量;②**dev loss/CMB 测不出生成退化,交互测试/长文生成评测不可省**(呼应 §13.1 lr4e-4 教训:多维度评测);③真正可聊的模型是 R5 产物(全量 134k×2ep + 最优配置)。④"一次只问一个问题"是天然 DPO 偏好对(单问=chosen,多轮倾倒=rejected),记入 DPO 待办。⑤若 R5 后多轮仍倾倒,考虑数据侧改造:把单样本多轮拆成"截至第k轮→只预测第k轮 assistant"的多条训练样本(sub_task=next_question_generation,画像已备 61,077 个可派生点)。

## 15. Codex 生成数据交付 + 独立审计(2026-07-09)

### 15.1 交付内容
Codex(MiniMax-M3 teacher)并行生成三类缺口数据,产出 `data/sft_pipeline/generated/`,总 23,000 条:
- **risk_redflag_safety_refusal 10,000**(high 8038 / medium 1962;A1急症升级+A2不安全请求拒绝;token 3583万;assistant p50=149字)
- **test_report_explanation 8,000**(10 类报告各 800:血常规/肝功/肾功/甲功/血脂/尿常规/腹超/胸CT/心电图/肿瘤标志物;数值规则引擎生成保证自洽,teacher 只写解释)
- **hospital_policy_rag_qa 5,000**(有答案 4000/无答案 1000;配套 hospital_kb.jsonl 虚构医院知识库;evidence_required=true)
Codex 自检 `sft_gap_generation_final_check.json`:三类 ok=true,id/hash 全唯一无重复,丢弃原因透明(A类 grounding/refusal/immediacy 规则过滤 335+ 条,report 类 ascii/缺异常项过滤,rag answer_not_grounded 丢 335)。

### 15.2 独立审计(scripts/sft_data/audit_gen.py,不信自检信抽样)
**schema 全过**:三类共 23000 条,canonical 键全、messages 非空、末轮全 assistant、全含 system;**零 ASCII 泄漏**(≥4连续英文字母=0,比 SFT 半成品干净得多)。
**红线规则复检**:
- risk: high 8038 条,末端未见急诊/120升级语仅 1 条(经查为关键词误报,内容实际已妥善处理孕早期出血)。抽样验证:胸痛→急性心梗识别+拒"扛一扛"、误吞降压药→中毒急救+不催吐、双抗生素混服→拒绝+讲耐药/肝肾原因,红旗识别与拒绝逻辑真实到位。
- rag: evidence_required=true 5000/5000;无答案样本("充电桩空位/院长在不在")干净拒答"资料里没有,建议咨询服务台";有答案样本严格贴证据、"资料外不展开"。
- report: 数值自洽(白细胞2.3<下限判读正确)、措辞用"提示"不用"确诊";8000 条仅 4 条疑似过度确诊、361 条未命中随访关键词(多为误报)。
**结论:质量合格,准予合并。teacher 生成数据质量 > 欠训练学生模型输出,这批可作 SFT 高质量补充,也是 §14 "为什么用 teacher 蒸馏补缺口"的实证。**

### 15.3 合并计划(v1 → v1.1)
把 generated 三类接入主管线:走 02_clean(轻过滤,大概率近乎全留)→ 03_label(已带 task_type,跳过分类)→ 04_dedup(与现有池跨源去重)→ 05 重采样。目标:risk_redflag 从 v1 的 1,678 提到 ~10,309、hospital_rag 从 0 提到 ~5,154、report 已足量。合并后 v1.1 用于 R4/R5。

## 16. v1.1 数据集合并完成(2026-07-09)

Codex 生成数据(§15 审计通过)并入主管线:merge_gen_v11.py 补全 metadata+轻清洗+全池精确去重(23,000 条全留,与现有 200.9 万池零重复)→ 写入 04_deduped/gen_*.jsonl → sample_05_v11.py 重采样(gen_minimax_m3 设最高优先级 0)。

**v1.1 = 152,241 条**(v1 为 140,845,净增 11,396):

| task_type | v1 | v1.1 | 目标 | 状态 |
|---|---|---|---|---|
| symptom_consultation | 36,080 | 36,080 | 36,080 | ✅ |
| health_encyclopedia_qa | 28,349 | 28,349 | 28,349 | ✅ |
| triage_guidance | 18,040 | 18,040 | 18,040 | ✅ |
| pre_consultation_multiturn | 15,463 | 15,463 | 15,463 | ✅ |
| test_report_explanation | 15,463 | 15,463 | 15,463 | ✅ teacher版足量 |
| chronic_disease_management | 12,886 | 12,886 | 12,886 | ✅ |
| **risk_redflag_safety_refusal** | **1,678** | **10,309** | 10,309 | ✅ **安全短板补齐** |
| medication_guidance_safe | 7,732 | 7,732 | 7,732 | ✅ |
| conversation_summary_structured | 5,154 | 5,154 | 5,154 | ✅ |
| **hospital_policy_rag_qa** | **0** | **2,765** | 5,154 | ⚠️ 54%,近重去重砍930+超采淘汰所致(RAG句式天然雷同),质量高,判定够用不硬灌 |

**切分**:train 144,657 / dev 3,787 / test 3,795 + hard_eval 1,836(v1 为 1,495,安全类补充后 hard case 更全)。风险分布 high 14,309(v1 仅 7,358,翻倍,安全能力评测更有力)。
**来源**:med_zh 59,506 / Huatuo 39,780 / 种子+派生 30,664 / **gen_minimax_m3 11,703** / DISC 8,348 / 其余。种子仍未稀释。
**产物**:05_final_v11/{train,dev,test,hard_eval}.jsonl;训练格式 data/sft_v11/{train_full,subset_40k,dev,dev_1k}.json(assistant先手已修);dataset_info 注册 sft_v11_{full,40k,dev,dev1k}。

**结论:v1.1 定稿,即为 R4/R5 最终训练集。** cutoff 2048、最优配置 rank64/lr2e-4/all-linear(§13.1)。hospital_rag 若 R4/R5 评测显示 RAG 忠实度不足,再定向补生成进 v1.2。

## 17. v1.1 误分类修复 + 出场终检 + F盘云套件(2026-07-10)

### 17.1 发现并修复 report/rag 误分类 bug
出场前终检(scripts/sft_data/final_gate.py,5 门)门5 抽查发现:**test_report_explanation 桶混入 289 条真身为 hospital_policy_rag_qa 的 gen 数据**。根因:sample_05 的二次挖掘 REPORT_Q 正则(含"报告"关键词)误抓了 gen RAG 数据里的"报告领取/检查报告可打印"字样。修复:re-tag 逻辑(sample_05_v11.py 第93/126行)`if d["source"] not in ("internal_seed_flywheel", "gen_minimax_m3")` —— 生成数据自带正确 task_type,不参与关键词重分类。
**重跑结果(意外改善)**:total 152,241→**154,476**;hospital_rag 2,765→**5,000 满配**(误分类的 RAG 回归本桶,恰好补齐 54%→100%);report 桶变纯净(来源仅 med_zh/DISC/Huatuo,含 hgkb 院务特征=0)。

### 17.2 出场终检五门全绿(修复后)
| 门 | 检查 | 结果 |
|---|---|---|
| 1 结构合法(human开头/gpt结尾) | train+dev 150,635 | 0 异常 |
| 2 assistant先手修复 | train 以gpt打头 | 0 |
| 3 train/dev泄漏 | dev∩train 哈希 | 0 |
| 4 评测集(CMB/CMExam/MLEC/CBLUE)泄漏进训练 | 来源统计 | 0 |
| 5 生成数据内容抽查 | report纯净/rag名副其实 | ✅ |
train.jsonl 来源:med_zh 56,578/Huatuo 36,570/gen 12,937/种子14,718/derived14,433/DISC9,438/CMD1,733/shibing402。

### 17.3 F盘云套件(F:\rlhf_lab_cloud_kit,防5133被征用)
用户要求所有租机必需品落本地 F 盘。已拉全:
- `data/dataset_info.json` + `data/sft_v11/`(train_full/subset_40k/dev/dev_1k,208M,训练直接用)
- `data/eval_sets/`(CMB/cmexam/05_final_v11 的 test+hard_eval,评测用)
- `models/Qwen3-8B-Base`(16G,base 模型;云上也可 hf-mirror 直拉)
- `scripts/`(全部:训练/评测/vLLM/数据管线)、`configs/`(r4r5 + sft_v1 全套 yaml)
- `docs/`(EXPERIMENT_LOG.md 全程记录 + CLOUD_SETUP_R4R5.md 开机流程 + requirements.{train,vllm}.lock)
**R4/R5 全套配置已就绪**:make_r4_arms.py(四臂生成)/r4_arm_template.yaml/r5_final.yaml/eval_hardeval.py(6类分桶评测)。

### 17.4 结论:数据侧万事俱备
v1.1(154,476,10类满配)终检五门全绿,超参锁定(rank64/lr2e-4/all-linear),R4/R5 配置+评测脚本+云端流程全备好并落 F 盘。**唯一待办=用户租 4×5090D**。开机后照 CLOUD_SETUP_R4R5.md:传套件→装环境smoke→R4四臂并行→R5全量→回传。付费时间只做"传+跑"。
