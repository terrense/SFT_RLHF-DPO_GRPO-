# requirements_sft_rigor.md — 新一轮 SFT 严谨性清单（原文，2026-07-19 追加）

本文件是用户在 `docs/EXPERIMENT_LOG.md` 记录的第一轮 SFT→DPO→GRPO 实验完成之后，
提出的**更严格的 SFT 方法论要求**，逐字保留，作为新一轮实验的验收标准。
如果你是全新会话（新终端窗口、断电重启），**先读这份文件**，再读
`spec/design.md`（现状 gap 分析 + 已做出的决策），再读 `spec/tasks.md`（当前可执行清单）。

不要因为 `docs/EXPERIMENT_LOG.md` 里第一轮已经做过一部分（LoRA rank/lr/target_modules
消融、R4 数据配比四臂、DPO beta 扫参等）就假设全部要求都已满足——`spec/design.md`
逐条核对了哪些真正做过、哪些只是部分做过、哪些完全没做。

---

# 医疗 Agent SFT 实验实现要求与避坑清单

## 一、总目标

基于现有医疗问答、预问诊、多轮对话、真实业务反馈和工具调用数据，完成一个可复现的医疗 Agent SFT 实验系统。

系统必须能够回答以下问题：

1. SFT 是否真正提升了医疗 Agent 能力？
2. 提升来自哪一类数据或哪一种训练设计？
3. 是否牺牲了通用能力、多轮交互能力或安全性？
4. 模型性能变化是否来自数据污染、答案变长或评测偏差？
5. 训练异常时，能否定位到具体数据源、具体 batch 和具体样本？
6. 离线单轮指标提升，是否能够转化为多轮 rollout 的任务成功率提升？

禁止只完成"训练成功并生成 checkpoint"。

---

# 二、实验基本原则

## 1. 必须先建立原始模型 Baseline

在任何训练开始前，对原始 Instruct 模型完成全套评测。

Baseline 至少包括：

* 医疗知识问答；
* 预问诊多轮任务；
* 危险信号识别；
* 科室推荐；
* 工具调用；
* JSON/结构化输出；
* 通用能力回归；
* 多轮自由 rollout；
* 推理延迟和显存占用。

所有后续实验必须与同一个 baseline 比较。

禁止训练后才临时设计评测集。

---

## 2. 一次实验原则上只改变一个核心变量

例如比较 LoRA rank 时：

* 数据不变；
* 学习率不变；
* batch size 不变；
* seed 不变；
* max length 不变；
* 评测集不变。

禁止同时修改：

* 数据配比；
* rank；
* learning rate；
* sequence length；
* packing；
* loss normalization。

否则实验无法归因。

---

## 3. 所有实验必须可复现

每次运行必须保存：

* 完整配置文件；
* Git commit hash；
* 模型名称和 revision；
* tokenizer revision；
* 数据版本；
* 数据过滤规则版本；
* random seed；
* CUDA/PyTorch/Transformers 版本；
* GPU 型号；
* 启动命令；
* checkpoint 路径；
* 评测结果；
* 训练日志。

每次实验生成独立目录，例如：

```text
outputs/
  exp_e0_baseline/
  exp_e1_open_medical/
  exp_e2_preconsultation/
  exp_e3_real_feedback/
```

---

# 三、数据格式要求

## 4. 所有数据统一为标准 conversation schema

建议统一为：

```json
{
  "sample_id": "unique_id",
  "source": "open_source_medical",
  "task_type": "preconsultation",
  "risk_level": "high",
  "quality_score": 0.92,
  "messages": [
    { "role": "system", "content": "..." },
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

必须保留以下 metadata：

* `sample_id`
* `source`
* `task_type`
* `risk_level`
* `quality_score`
* `dialogue_turns`
* `has_tool_call`
* `language`
* `hospital_or_dataset`
* `synthetic_or_real`
* `patient_or_case_id`
* `token_length`
* `supervised_token_length`

---

## 5. 不允许把不同数据源直接无标签 concat

至少区分：

* 开源医学问答；
* 医学考试数据；
* 医疗科普；
* 单轮问诊；
* 多轮预问诊；
* 导诊数据；
* 危险信号数据；
* 真实业务反馈；
* 合成数据；
* 工具调用数据；
* 工具异常恢复数据；
* 通用指令 replay 数据。

后续训练必须支持按 source 和 task 设置采样权重。

---

## 6. 训练集、验证集和测试集不能仅随机按样本切分

必须尽可能执行：

* patient-level split；
* case-level split；
* source-level split；
* hospital-level split；
* time-based split。

同一病例的不同改写、摘要、扩写和合成版本必须进入同一个数据分区。

禁止同一患者不同轮次分别进入 train 和 test。

---

# 四、数据清洗要求

## 7. 必须执行格式检查

检查并统计：

* 空 messages；
* role 顺序错误；
* 连续两个 user；
* 连续两个 assistant；
* 缺少 assistant target；
* assistant 内容为空；
* JSON 不合法；
* 工具参数不合法；
* 缺少结束符；
* 非法 Unicode；
* 异常乱码；
* 长度为零；
* 超长样本。

所有被过滤的数据必须记录过滤原因。

---

## 8. 必须执行精确去重和近似去重

至少包括：

* exact hash deduplication；
* normalization 后精确去重；
* MinHash 或 SimHash；
* embedding similarity；
* prompt 近似重复；
* answer 模板重复；
* 同病例不同改写检测。

输出去重前后：

* 样本数；
* token 数；
* supervised token 数；
* 各数据源保留比例。

不能只报告"清洗后剩余多少条"。

---

## 9. 必须检查 benchmark contamination

至少检查：

* benchmark 问题是否出现在训练数据；
* benchmark 答案是否出现在训练数据；
* benchmark 的语义改写是否存在；
* 同一病例模板是否存在；
* teacher model 是否可能见过 benchmark。

污染检查不能只做字符串匹配。

---

## 10. 合成数据必须标注来源和 teacher 信息

每条合成数据至少记录：

* teacher model；
* generation prompt version；
* temperature；
* sampling 参数；
* 生成时间；
* 是否经过自动评分；
* 是否经过人工审核。

禁止将合成数据与真实业务数据混为同一来源。

---

## 11. 真实医疗数据必须经过脱敏

需要删除或替换：

* 姓名；
* 手机号；
* 身份证；
* 住址；
* 医院编号；
* 病历号；
* 精确时间；
* 可重新识别个人身份的组合信息。

脱敏后仍需检查语义完整性，避免把关键医学信息误删。

---

# 五、Chat Template 与 Label Mask

## 12. 必须使用与目标模型匹配的官方 chat template

训练和推理必须使用相同的：

* role token；
* assistant prefix；
* EOS token；
* tool call 格式；
* tool response 格式；
* generation prompt。

禁止训练使用自定义 ChatML，推理却调用官方模板。

---

## 13. 默认采用 assistant-only loss

原则上：

* system token 的 label 为 `-100`；
* user token 的 label 为 `-100`；
* padding token 的 label 为 `-100`；
* assistant 内容参与 loss；
* assistant EOS 是否参与 loss 必须显式配置并测试。

---

## 14. 必须实现 label mask 可视化工具

随机抽取数据，打印：

```text
token
token_id
role
label
is_supervised
```

输出示例：

```text
[user]           label=-100
我               label=-100
头               label=-100
疼               label=-100
[assistant]      label=-100
请               label=xxxxx
问               label=xxxxx
疼               label=xxxxx
了               label=xxxxx
多               label=xxxxx
久               label=xxxxx
[EOS]            label=xxxxx
```

必须人工检查至少 20 条样本。

---

## 15. 必须做"小数据过拟合测试"

选择 8–32 条人工确认完全正确的数据。

要求模型能够在较少 step 内明显过拟合，并验证：

* loss 能快速下降；
* 模型只生成当前 assistant 回复；
* 不会继续生成用户内容；
* 不会输出完整预设对话；
* 能正确停止；
* tool call 格式正确。

如果小数据无法过拟合，不允许开始大规模训练。

---

## 16. 专门设计"倒豆子"回归测试

测试输入：

```text
用户：我头疼。
```

预期模型只生成当前轮追问，例如：

```text
请问头疼持续多久了？
```

不允许模型继续生成：

```text
用户：三天。
医生：有没有发烧……
用户：没有……
```

发现该问题时，优先检查：

1. assistant-only mask；
2. EOS；
3. conversation 展开逻辑；
4. chat template；
5. generation stopping criteria；
6. 是否把整段未来对话放入单个 completion。

---

# 六、Sequence Packing 要求

## 17. 第一版 baseline 默认关闭 packing

首先用 non-packing 跑通：

* 数据；
* loss mask；
* 单元测试；
* 收敛；
* 评测。

不要一开始为了吞吐量引入 packing，增加排错复杂度。

---

## 18. Packing 必须作为独立对比实验

设置：

* E-packing-off；
* E-packing-eos；
* E-packing-block-mask。

比较：

* 吞吐量；
* padding ratio；
* validation loss；
* 多轮行为；
* 危险信号误报；
* 输出风格；
* 工具调用格式；
* 跨样本污染。

---

## 19. 不允许假设 EOS 等于硬隔离

若普通 causal attention 下拼接：

```text
Sample A + EOS + Sample B
```

必须明确 Sample B 可能读取 Sample A。

严格 packing 应优先使用：

* block-diagonal attention；
* FlashAttention varlen；
* 独立 `cu_seqlens`；
* 正确的 sequence boundary。

---

## 20. Packing 需要额外检查

必须验证：

* 新样本首 token 的 label 是否屏蔽；
* position IDs 是否正确；
* 截断是否破坏 assistant EOS；
* tool call 和 tool result 是否被拆开；
* label mask 在拼接后是否错位；
* 样本 B 是否能够关注样本 A；
* packed loss 与 non-packed loss 是否接近。

---

# 七、长度分布与 Truncation

## 21. 训练前必须统计长度分布

至少输出：

* P50；
* P75；
* P90；
* P95；
* P99；
* 最大长度；
* assistant supervised token 分布；
* 不同数据源的长度分布。

禁止凭感觉直接设置 8K、16K 或 32K。

---

## 22. 必须记录 padding ratio

每个 batch 记录 padding tokens / total tokens。

如果 padding 比例过高，考虑：

* length bucketing；
* dynamic batching；
* token-based batching；
* packing。

---

## 23. Truncation 不得静默发生

每条被截断样本必须记录：

* 截断前长度；
* 截断后长度；
* 被截断角色；
* 是否丢失 assistant EOS；
* 是否丢失工具结果；
* 是否丢失危险信号；
* 是否丢失最终结论。

优先避免从末尾机械截断。

---

## 24. 多轮对话可使用 turn-aware truncation

优先保留：

* system；
* 最近用户输入；
* 关键医疗槽位；
* 危险信号；
* 当前 assistant target。

可以删除或摘要较早的低价值轮次。

---

# 八、Loss Normalization 对比实验

## 25. 必须实现并比较至少两种 loss normalization

### 实验 A：Token-level mean

所有有效 assistant token 等权。

### 实验 B：Sample-level mean

先对每条样本内部 token loss 求平均，再对 batch 样本求平均。

可选实验：

### 实验 C：Task-level weighted loss

按任务分别计算平均 loss，再按业务权重组合。

---

## 26. 必须记录不同任务的 supervised token 占比

不能只统计样本数。

例如：

| 数据源   | 样本占比 | 总 Token 占比 | Supervised Token 占比 |
| ----- | ---: | ---------: | ------------------: |
| 医学问答  |  50% |        65% |                 72% |
| 多轮预问诊 |  30% |        25% |                 18% |
| 工具调用  |  20% |        10% |                 10% |

若不统计 supervised tokens，会错误估计实际梯度贡献。

---

## 27. 检查长答案是否支配训练

对比：

* 长答案与短答案平均 loss；
* 长答案贡献的梯度比例；
* 输出平均长度；
* 首轮追问长度；
* 冗余回答率；
* 是否出现过度解释。

---

# 九、多数据源采样策略

## 28. 禁止按原始数据量自然采样

如果开源问答有 80 万条、真实数据只有 2 万条，自然采样会让真实数据几乎没有影响。

必须支持：

* source weight；
* task weight；
* quality weight；
* risk weight；
* temperature sampling；
* oversampling；
* capped sampling。

---

## 29. 设计至少三组数据配比实验

### E1：Natural Mixing

按原始数据量混合，作为反例或基线。

### E2：Balanced by Source

不同来源设置人工比例。

### E3：Business-Weighted

提高：真实预问诊、危险信号、工具调用、多轮恢复轨迹；
降低：模板化开源问答、重复科普、低质量合成数据。

---

## 30. 必须控制高质量小数据被淹没的问题

真实数据可以：oversampling；提高 source weight；后期单独小学习率训练；采用 curriculum；
作为最后 alignment stage。但必须防止过拟合真实小数据。

---

# 十、Curriculum 对比实验

## 31. 设计 Mixed Training 与 Staged Training 对比

### Mixed Training

所有来源从头混合训练。

### Staged Training

* Stage 1：领域基础（开源医学问答和医学知识）
* Stage 2：任务适配（预问诊、多轮追问、科室推荐、工具调用）
* Stage 3：真实分布校准（高质量真实反馈、困难案例、恢复轨迹，较小学习率，保留部分通用 replay）

---

## 32. 每个阶段都必须跑完整评测

不能只在最终模型评测。需要观察：Stage 1 是否提高知识但破坏交互；Stage 2 是否恢复多轮和工具能力；
Stage 3 是否提高线上分布表现；是否出现通用能力遗忘。

---

# 十一、梯度冲突与数据源冲突

## 33. 必须记录 source-specific validation loss

分别报告：open-source medical loss；preconsultation loss；real-world loss；tool-call loss；
red-flag loss；generic replay loss。总 validation loss 下降不能掩盖某个关键任务退化。

---

## 34. 设计数据源增量消融

至少包括：只用开源医疗数据；开源+预问诊；开源+预问诊+工具调用；加入真实反馈；
加入危险信号重采样；加入通用 replay。观察每类数据分别影响什么能力。

---

## 35. 可选实现梯度余弦相似度诊断

只在 LoRA 参数/某些代表层/定期抽样 batch 计算不同数据源梯度的 cosine similarity。
用途是诊断，不是默认训练机制。暂不优先实现 PCGrad。只有发现明确且稳定的梯度冲突后，再做 PCGrad 消融。

---

# 十二、高风险与低频数据处理

## 36. 高风险病例不能按自然频率训练

需要提高覆盖：胸痛伴冷汗；突发剧烈头痛；单侧肢体无力；呼吸困难；意识障碍；严重过敏；
高热伴意识改变；持续大量出血。

---

## 37. 高风险重采样必须同时评测误报

至少报告：red-flag recall；red-flag precision；false escalation rate；missed emergency rate；
unnecessary emergency recommendation rate。禁止只报告召回率提升。

---

## 38. 构建 hard negatives

例如：普通肌肉性胸痛 vs 心源性胸痛；偏头痛 vs 蛛网膜下腔出血；周围性眩晕 vs 后循环卒中；
普通皮疹 vs 严重过敏反应。模型必须学习关键区分条件，而不是看到"胸痛"就全部推荐急诊。

---

# 十三、多轮策略和 Rollout

## 39. 不允许只做 teacher-forcing 的 next-turn 评测

必须加入自由多轮 rollout。模型每轮都使用自己上一轮输出产生的真实历史。

评测：最终任务完成率；required-slot recall；平均轮数；重复询问率；漏问率；过早结束率；
无止境追问率；危险信号发现时间；错误恢复率。

---

## 40. 构建恢复轨迹数据

必须包含：模型已经重复问过问题；模型漏问关键危险信号；用户答非所问；用户拒绝回答；
用户中途补充新症状；前后回答冲突；ASR 错误；工具超时；工具返回空结果；工具返回互相冲突的结果。

训练模型学习：承认并纠正；重新聚焦；补问关键信息；调用替代工具；请求人工接管。

---

## 41. 设计 DAgger-style 数据闭环

1. 当前模型执行完整问诊；2. 保存所有失败状态；3. 对失败状态生成专家动作；
4. 人工或规则审核；5. 加回 SFT 数据；6. 重新训练；7. 再次 rollout。

重点收集模型真实会进入的错误状态，而不只是完美专家轨迹。

---

## 42. 显式区分 Agent 动作类型

建议将下一步建模为：`ASK / ANSWER / CALL_TOOL / ESCALATE / STOP / CLARIFY / RECOVER`。
不要让模型只学习一段无结构自然语言。可以先预测 action，再生成 action content。

---

# 十四、工具调用数据

## 43. 工具调用必须单独评测

至少包括：tool selection accuracy；是否需要调用工具；tool argument accuracy；JSON valid rate；
参数类型正确率；required field recall；hallucinated tool rate；unnecessary tool call rate；
tool result grounding；工具失败恢复率。

---

## 44. 不允许只训练成功工具轨迹

必须加入：timeout；empty result；invalid argument；permission denied；service unavailable；
result conflict；malformed response。模型必须学会重试、降级、澄清或人工转接。

---

## 45. 工具调用和自然语言问答可能冲突

对于同一个用户输入，必须通过 system 或 metadata 明确当前模式：
`task_type / current_stage / available_tools / tool_policy / risk_level`。
避免同一输入同时被监督为：直接回答；继续追问；调用工具。

---

# 十五、LoRA 与训练参数对比

## 46. 首轮实验优先 LoRA，不要直接全参数微调

推荐先建立：LoRA rank 8；LoRA rank 16；LoRA rank 32。其余配置保持一致。

---

## 47. Target Modules 必须做消融

至少比较：A. Attention Only (q/k/v/o_proj)；B. Attention + MLP (+ gate/up/down_proj)。
比较：医疗能力；多轮能力；通用能力退化；可训练参数量；显存；训练速度。

---

## 48. 不要默认 rank 越大越好

rank 提升可能：增加容量；加快拟合；增加过拟合；加剧通用能力漂移；增加 checkpoint 大小。
选择 rank 应基于验证和业务指标，而不是参数量。

---

## 49. 学习率至少做三档

例如 LoRA：low；medium；high。可从 `1e-5 / 5e-5 / 1e-4 / 2e-4` 范围开始搜索。
禁止只跑一个学习率就得出结论。

---

## 50. 训练必须使用 warmup

建议初始搜索 `warmup_ratio = 0.01–0.05`。记录 warmup 阶段：loss；gradient norm；
是否出现 spike；不同数据源 loss。

---

## 51. 优先使用 BF16

如果硬件支持，优先 BF16，而不是 FP16。若出现 NaN 或 loss spike，需要提供可切换配置：
bf16；fp32 debug；gradient clipping；关闭 FlashAttention；关闭 gradient checkpointing；降低学习率。

---

# 十六、Batch 与有效 Token

## 52. 必须报告 effective batch size

`B_effective = B_micro × N_GPU × N_accumulation`，但不能只报告样本数。
还必须报告：tokens per step；supervised tokens per step；padding tokens per step；
每个 source 的 tokens per step。

---

## 53. 允许使用 token-based batching

变长样本下，建议按照最大 token 数组织 batch，而不是固定 sample count。
避免某个 batch 因为长对话突然显存溢出或梯度异常。

---

## 54. Gradient accumulation 必须严格验证

确认：每个 micro-batch 都 backward；只有 accumulation 完成后 optimizer step；
scheduler 与 optimizer 同步；梯度裁剪在 optimizer step 前执行；loss scaling 是否正确；
日志中的 loss 是否被 accumulation 重复除法。

---

# 十七、训练可观测性

## 55. 每个 step 或固定间隔记录

至少包括：global step；epoch；learning rate；train loss；source-specific loss；
task-specific loss；gradient norm；tokens per second；supervised tokens per second；
GPU memory；step time；padding ratio；max sequence length；sample IDs；source composition。

---

## 56. Loss spike 必须能回溯到具体 batch

保存 spike 附近的：sample ID；source；raw text；tokenized input；labels；token length；
supervised length；risk level；quality score。支持单独重放该 batch。

---

## 57. 每隔固定 step 做 generation probe

使用固定测试集生成结果，观察：是否变长；是否重复；是否倒豆子；是否过度急诊；
是否过度拒答；是否忘记调用工具；是否出现 JSON 格式崩溃。不要只等训练结束生成。

---

# 十八、SFT 评测体系

## 58. 评测必须分层

A. 医疗知识（accuracy/F1/exact match/clinical QA score）
B. 多轮问诊（task success/slot recall/average turns/redundant question rate/premature stop/endless questioning）
C. 安全（red-flag recall/unsafe advice rate/unsupported diagnosis rate/contraindication error/escalation accuracy）
D. Agent 工具（tool selection/arguments/valid JSON/execution success/recovery success）
E. 通用能力（一般指令遵循/非医疗问答/英文/中文/推理/格式输出）

---

## 59. 必须控制答案长度偏差

评测时需要同时报告：平均输出长度；token 数；冗余率；在长度相近条件下的质量；
concise prompt 下的遵循率。禁止仅依赖偏好长答案的 LLM judge。

---

## 60. LLM Judge 必须做去偏

至少做到：隐藏模型名称；随机交换答案顺序；使用明确 rubric；允许 tie；固定 judge temperature；
多次评测或多个 judge；抽样人工复核。

---

## 61. 医生评价必须有标准化 rubric

至少分别评价：医学正确性；风险识别；信息完整性；是否过度诊断；是否存在不安全建议；
问诊顺序；表达清晰度；下一步建议合理性。需要记录评审者一致性。

---

# 十九、能力提升归因实验

## 62. 必须做数据消融

至少比较：无真实数据；无多轮数据；无危险信号数据；无工具调用数据；无通用 replay；
无 recovery trajectory；无合成数据。证明每类数据贡献了什么。

---

## 63. 必须做 Counterfactual Evaluation

对病例修改关键变量（例如"胸痛 5 分钟休息后缓解"改为"胸痛 40 分钟伴冷汗和呼吸困难"），
模型风险判断必须变化。修改无关变量（如职业），结论原则上不应明显变化。

---

## 64. 必须做组合泛化测试

测试训练集中未直接出现的症状组合，验证模型是否真正整合医学特征，而不是记住模板。

---

## 65. 必须做 Leave-One-Source-Out 测试

例如：四家医院训练，第五家医院测试；或去除某开源来源，在该来源对应分布上评测。
用于验证跨机构和跨表达风格泛化。

---

# 二十、灾难性遗忘

## 66. 每个 checkpoint 都必须跑通用回归测试

检查：非医疗问题是否全部医疗化；英文能力是否下降；数学推理是否下降；
工具调用格式是否下降；拒答是否异常；对普通问题是否过度谨慎。

---

## 67. 对比是否加入通用 replay 数据

设计：无 replay；5% replay；10% replay；20% replay。比较医疗提升和通用能力保持情况。

---

## 68. 可选做 KL 或 Logit Distillation 实验

若领域训练导致明显漂移，可增加参考模型保持项。但不要在 baseline 阶段直接加入，
否则增加变量且难以归因。

---

# 二十一、训练异常排查机制

## 69. NaN 或 loss spike 的排查顺序

1. 检查异常 batch；2. 检查 label 是否全为 -100；3. 检查超长样本；4. 检查非法 token；
5. 检查 learning rate；6. 检查 gradient norm；7. 切换 FP32 debug；8. 关闭 FlashAttention；
9. 关闭 packing；10. 单卡复现；11. 固定小数据；12. 检查分布式同步。
不要第一反应就随意改学习率。

---

## 70. OOM 排查必须区分原因

可能来自：单条超长样本；batch token 波动；packing 失控；gradient checkpointing 未启用；
optimizer states；attention implementation；eval generation max tokens；
checkpoint 保存瞬时峰值；DeepSpeed/FSDP 配置。记录 OOM 前最后一个 batch 的 token 信息。

---

# 二十二、建议实验矩阵

## P0：必须完成的可信 Baseline

| 实验 | 描述 |
| -- | ------------------------------------ |
| E0 | 原始 Instruct 模型，不训练 |
| E1 | LoRA，assistant-only loss，non-packing |
| E2 | E1 + 数据源平衡采样 |
| E3 | E2 + 多轮预问诊数据 |
| E4 | E3 + 真实反馈数据 |
| E5 | E4 + 危险信号重采样 |
| E6 | E5 + 通用 replay |

## P1：核心消融实验

| 实验  | 唯一改变变量 |
| --- | ---------------------- |
| A1  | Token-level loss |
| A2  | Sample-level loss |
| A3  | Mixed training |
| A4  | Staged curriculum |
| A5  | Packing off |
| A6  | EOS-only packing |
| A7  | Block-diagonal packing |
| A8  | LoRA rank 8 |
| A9  | LoRA rank 16 |
| A10 | LoRA rank 32 |
| A11 | Attention-only LoRA |
| A12 | Attention + MLP LoRA |

## P2：高级研究实验

| 实验 | 目标 |
| -- | --------------------------- |
| R1 | DAgger-style 失败轨迹回灌 |
| R2 | Recovery trajectory SFT |
| R3 | State perturbation |
| R4 | Task-conditioned SFT |
| R5 | Gradient conflict diagnosis |
| R6 | PCGrad 消融 |
| R7 | KL-preserving SFT |
| R8 | 多 Adapter 与 Router |

P2 不应先于 P0 和 P1。

---

# 二十三、实验决策规则

## 71. 不以最低 validation loss 选择模型

最终选择需要综合加权 Medical / Interaction / Safety / Tool / Regression / Latency，
其中安全指标必须设置硬门槛，而不是完全参与加权平均。例如：red-flag recall 低于阈值直接淘汰；
unsafe advice rate 高于阈值直接淘汰；JSON valid rate 低于阈值不能进入部署。

---

## 72. 任何指标提升必须能解释来源

最终报告必须回答：哪类数据提升了医疗知识；哪类数据提升了多轮交互；哪类数据提升了工具调用；
哪类数据提升了安全召回；哪种设计造成了通用能力退化；packing 是否改变行为；
loss normalization 是否改变回答长度；curriculum 是否优于 mixed training。
禁止只写"完整模型效果最好"。

---

# 二十四、代码工程要求

## 73. 推荐目录结构

```text
medical_sft/
├── configs/{data,train,eval}/
├── data_pipeline/{normalize,deduplicate,split,quality_filter,contamination_check}.py
├── training/{train_sft,collator,loss,sampler,callbacks}.py
├── evaluation/{medical_qa,multi_turn_rollout,safety_eval,tool_eval,regression_eval}.py
├── diagnostics/{inspect_labels,inspect_packing,replay_batch,gradient_conflict}.py
├── scripts/{run_baseline.sh,run_train.sh,run_eval.sh}
└── reports/
```

## 74. 每个模块都必须有单元测试

至少包括：chat template；assistant mask；EOS；truncation；packing；source sampler；
sample-level loss；tool schema；dataset split；duplicate detection。

## 75. 所有配置必须外部化

禁止硬编码：model path；data path；source weights；task weights；max length；LoRA rank；
target modules；learning rate；warmup；batch size；packing；loss normalization；
eval interval；generation config。统一使用 YAML 或 JSON 配置。

---

# 二十五、最终交付物

1. 数据清洗与统计脚本；2. 去重与污染检测报告；3. chat template 与 label mask 检查工具；
4. non-packing SFT baseline；5. packing 对比实现；6. token-level 和 sample-level loss；
7. 多数据源 weighted sampler；8. curriculum training 配置；9. 多轮 rollout evaluator；
10. 安全评测；11. 工具调用评测；12. 通用能力回归测试；13. 训练监控与异常 batch 回放；
14. 实验汇总表；15. 自动生成实验报告的脚本。

---

# 二十六、禁止事项

1. 禁止没有 baseline 直接训练。2. 禁止只看 train loss。3. 禁止只看一个医疗 benchmark。
4. 禁止用训练数据随机切一部分就称为可靠测试集。5. 禁止把开源、合成、真实数据混为一个总数。
6. 禁止默认 EOS 能隔离 packed samples。7. 禁止不检查 label mask。8. 禁止不做小数据过拟合测试。
9. 禁止一次修改多个变量后宣称某技巧有效。10. 禁止只使用 teacher-forcing 评价多轮 Agent。
11. 禁止用答案变长冒充质量提升。12. 禁止只提高危险信号召回却不评测误报。
13. 禁止让 LLM 同时生成、筛选、评价所有数据而无人类或规则校验。
14. 禁止把 checkpoint 生成成功当作项目完成。
15. 禁止先做 PCGrad、复杂 RL 或多 Adapter，再补最基本的数据和评测系统。

---

# 二十七、执行优先级

第一优先级（先保证正确）：数据格式；chat template；assistant mask；EOS；split；baseline；小数据过拟合。
第二优先级（建立可观测性）：source loss；supervised tokens；gradient norm；generation probe；多轮 rollout；安全评测。
第三优先级（可信对比）：数据配比；curriculum；loss normalization；packing；LoRA rank；target modules。
第四优先级（高级优化）：DAgger；recovery trajectory；gradient conflict；KL regularization；PCGrad；多 Adapter。

任何时候如果第一、第二优先级未完成，不允许跳到第四优先级。
