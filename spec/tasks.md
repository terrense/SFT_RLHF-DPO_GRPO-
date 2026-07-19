# tasks.md — 新一轮可执行清单（持久化，不依赖 Claude Code 会话内任务状态）

已在 `cmedalign` 项目验证过：Claude Code 自带的 TaskCreate/TaskList 状态**不会**跨会话
保留。所以这个文件才是真正的进度来源，每次状态变化都要在同一个 commit 里更新它。

读取顺序见 `STATUS.md` 顶部。排列顺序按 `spec/requirements_sft_rigor.md` §二十七的
执行优先级（先正确性，再可观测性，再对比实验，最后高级优化），不是按清单原文的章节号。

**最后更新：2026-07-19**

---

## 第一优先级：先保证正确（大部分可在本机 RTX 3060 Ti / CPU 上完成）

- [x] task_type 分类现状确认（10 类，146,809 条，见 `spec/design.md`）
- [x] gap 分析完成，对照清单 75 条逐条标记 ✅/🟡/❌（`spec/design.md`）
- [ ] **PII 脱敏确认**（阻塞项，见 `BLOCKERS.md` #1）——先问用户，不擅自处理
- [ ] 补全 `05_final_v11` schema 缺失字段：`quality_score`（数值化，当前只有
      `source_quality` 三档）、`token_length`、`supervised_token_length`（用真实
      Qwen3-8B tokenizer + 复用 cmedalign 项目已验证的 assistant-only mask 算法计算，
      不是字符数近似）、`has_tool_call`（当前全部为 False，因为没有工具调用数据）、
      `synthetic_or_real`（从 source 名单派生：`gen_minimax_m3`/`derived_from_seed`=synthetic，
      其余=real/open）、`patient_or_case_id`（需要先确认 `internal_seed_flywheel` 和
      `derived_from_seed` 之间是否有可用的病例关联字段，否则只能做样本级 id，不能做
      真正的 patient-level split）
- [ ] 用 cmedalign 已验证的 Qwen3 chat template 逻辑（三个已知坑：
      `return_assistant_tokens_mask` 静默失效 / think-stub 只在最后一轮出现 /
      `apply_chat_template(tokenize=True)` 默认返回 dict）**重新验证** LLaMA-Factory 的
      `qwen3`/`qwen3_nothink` 模板是否正确处理了同样的问题——现有 `check_mask.py` 是用
      **Base 模型**+`qwen3`模板写的，README 自己的结论是这个组合会倒豆子，所以这个脚本
      的历史输出不能直接当作"mask 是对的"的证据，需要用 **Instruct + qwen3_nothink**
      重新跑
- [ ] label mask 可视化：先在**开源数据**（如 Huatuo26M-Lite）上跑，人工检查 ≥20 条；
      `internal_seed_flywheel` 的可视化检查等 PII blocker 解决后再做
- [ ] 格式检查补全（清单 §7 的 12 类，现有 `clean_02.py` 只覆盖了广告/PII/剂量/武断
      诊断/危险建议/长度，**没有**覆盖：空 messages、role 顺序错误、连续同角色、缺少
      assistant target、非法 Unicode、异常乱码——需要新脚本补上，可直接复用 cmedalign
      项目 `ConversationRecord` 的校验逻辑思路）
- [ ] benchmark contamination 检查（清单 §9，**真正的缺口**——现有
      `convert_01.py` 的 `EVAL_RESERVED` 只是源级排除 CMB/CMExam 数据集本身，不是
      item-level 检查训练数据的其他来源是否碰巧包含 benchmark 题目/语义改写。可直接
      复用 cmedalign 项目 `cross_split_contamination`（MinHash）的实现思路，针对
      `data/eval_sets/CMB` 和 `data/eval_sets/cmexam` 跑一遍）
- [ ] patient/case-level split 审计——检查当前 95/2.5/2.5 切分是否已经不小心把同病例
      不同改写分到了 train/test 两边（`derived_from_seed` 是重点怀疑对象）
- [ ] 小数据过拟合测试（清单 §15，8-32条）——需要真实 GPU forward+backward，3060 Ti
      8GB 能否跑 4-bit 量化 + LoRA 的 Qwen3-8B-Instruct 待验证，如果不行则等真实算力

## 第二优先级：建立可观测性（大部分需要真实训练算力）

- [ ] source-specific / task-specific validation loss 追踪——需要先解决"task_type 在
      `export_v11.py` 导出时被丢弃"这个具体问题（要么保留额外列，要么训练时用
      sample_id 反查 task_type）
- [ ] supervised token 占比统计（按 task_type，清单 §26 的表）——一旦上面的
      token_length/supervised_token_length 字段补全，这个是纯统计，本机可做
      （不需要 GPU）
- [ ] 长度分布统计（P50/P75/P90/P95/P99，按数据源）——同上，本机可做
- [ ] generation probe（固定测试集定期生成检查）——需要真实训练在跑，暂不适用
- [ ] 多轮自由 rollout 评测框架搭建（不是训练依赖，框架本身现在就能设计/写代码，
      跑评测需要真实 checkpoint）

## 第三优先级：可信对比实验（大部分需要真实训练算力，配置可以先写好）

- [x] LoRA rank(8/16/32/64) 消融 —— 第一轮已做（R1）
- [x] 学习率(5e-5/2e-4/4e-4) 消融 —— 第一轮已做（R2）
- [x] target_modules(attn-only/all) 消融 —— 第一轮已做（R3）
- [x] 数据配比四臂（natural/seed/mixed/seed-2x）—— 第一轮已做（R4），基本对应清单
      §29 的 E1/E2/E3
- [x] DPO beta 扫参（0.1/0.3/0.5）—— 第一轮已做
- [ ] Packing 对比实验（off / eos-only / block-diagonal）—— **完全没做**，需要新配置
      + 新脚本
- [ ] Loss normalization 对比（token-level vs sample-level）—— **完全没做**，
      LLaMA-Factory 默认 token-level，需要验证是否支持切换或需要自定义 collator/loss
- [ ] Curriculum 对比（mixed vs staged 3-stage）—— **完全没做**

## 第四优先级：高级优化（明确排在最后，第一/二优先级未完成前不做）

- [ ] 工具调用数据构造 + 评测（清单 §43-45）—— **数据完全不存在**，是最大的新增
      数据缺口，需要先设计数据来源（合成？业务真实工具调用日志？）再动手
- [ ] 恢复轨迹 / DAgger 闭环（清单 §39-42）
- [ ] 归因实验：数据消融、counterfactual 评测、组合泛化测试、leave-one-source-out
      （清单 §62-65）
- [ ] 梯度冲突诊断（清单 §35，明确是可选/诊断用，不默认做）
- [ ] KL/Logit distillation、PCGrad、多 Adapter（清单 §68/§35/§8 P2 高级项）—— 按
      清单自己的要求，只有前面步骤都做完且明确发现问题后才考虑

---

## 下一步实际行动（不等真实 GPU 也能做的，按顺序）

1. 解决 PII blocker（问用户）
2. 写 schema 补全 + 真实 tokenizer token_length/supervised_token_length 计算脚本
   （对开源数据先跑，`internal_seed_flywheel` 等 blocker 解决）
3. 写 benchmark contamination 检查脚本（复用 cmedalign 的 MinHash 思路）
4. 写格式校验补全脚本（复用 cmedalign 的 ConversationRecord 校验思路）
5. patient/case-level split 审计
6. 长度分布 + supervised token 占比统计报告
7. 设计（不一定实现）packing / loss-normalization / curriculum 对比实验的配置文件，
   等真实算力到位就能直接跑
