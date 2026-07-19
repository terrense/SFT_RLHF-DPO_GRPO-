# Blockers — [本项目已冻结，见 STATUS.md 2026-07-19 更新] 原"新一轮 SFT 严谨性实验"

只放"必须用户决策才能继续"的事项。空 = 当前无阻塞。

> **本项目已冻结为历史资产库**（见 `STATUS.md`）。下面第 1 项不再只是"本项目"的阻塞——
> 它现在直接决定 `cmedalign` 项目能不能导入 `internal_seed_flywheel`/`derived_from_seed`
> 这两个真实业务数据源。在这里解决，`cmedalign` 那边的导入才能解锁，不需要重复问一遍。

## Open

（当前无阻塞——第 1 项已由用户澄清解决，见下方 Resolved。）

## Resolved

- **2026-07-19（最新）— `internal_seed_flywheel`/`derived_from_seed` 的"隐私"问题
  已解决：这批数据完全是虚构合成的，不是真实患者数据。** 用户明确澄清：构造方式是
  DeepSeek-V4-Pro 扮演患者（根据虚构种子病例设定）、`cmedalign` 项目的基座模型
  Qwen3-8B 扮演医生作答，随后由更强的模型和真实医护人员修改完善内容。也就是说：
  之前发现的"PII 正则只覆盖电话/QQ/邮箱、对种子数据只打标签不删除"这个空白**依然是
  真实存在的代码事实**，但因为底层内容本身就不是真实病历，所以不构成隐私风险——
  之前的分析在"如果这是真实数据"的假设下是对的，现在这个假设不成立，所以不再阻塞。
  这个结论已经写入 `E:\cmedalign_paper\SOURCE_AUDIT.md` §6b（该文件是论文的可信来源，
  以后关于这批数据"是不是真的没隐私风险"以此为准）。**结果**：这两个数据源现在可以
  正常导入 `cmedalign` 的 `data/raw/`，不需要额外脱敏这一步；但仍应在 license_ledger
  里如实记录"synthetic_or_real=synthetic, teacher_model=DeepSeek-V4-Pro,
  reviewed_by=stronger_model+clinical_staff"这些溯源字段，不能和其他公开数据源
  混为一谈（这是论文自己的报告规则要求的，不是隐私要求）。
- **2026-07-19 — 项目归属 + 本轮算力（已被同一天晚些时候的方向调整取代，见下一条）。**
  用户当时决定：延续 `rlhf_lab_cloud_kit`；本机 RTX 3060 Ti（8GB）用于纯数据/CPU 级别
  准备工作。
- **2026-07-19（同一天晚些时候）— 最终方向：`cmedalign_paper` 是锚点，本项目冻结为
  历史资产库，`cmedalign` 是唯一活跃项目。** 论文 `main.tex` 不需要 tool-calling/
  DAgger/packing/curriculum 这类通用工程严谨性内容，第一轮 GRPO（MCQ规则奖励）不等于
  论文要求的 GRPO（多轮患者模拟+5分量奖励）。见 `E:\cmedalign\spec\design.md`
  "THE ANCHOR" 一节的完整决策记录。

- **2026-07-19 — 项目归属 + 本轮算力（已被同一天晚些时候的方向调整取代，见下一条）。**
  用户当时决定：延续 `rlhf_lab_cloud_kit`；本机 RTX 3060 Ti（8GB）用于纯数据/CPU 级别
  准备工作。
- **2026-07-19（同一天晚些时候）— 最终方向：`cmedalign_paper` 是锚点，本项目冻结为
  历史资产库，`cmedalign` 是唯一活跃项目。** 论文 `main.tex` 不需要 tool-calling/
  DAgger/packing/curriculum 这类通用工程严谨性内容，第一轮 GRPO（MCQ规则奖励）不等于
  论文要求的 GRPO（多轮患者模拟+5分量奖励）。见 `E:\cmedalign\spec\design.md`
  "THE ANCHOR" 一节的完整决策记录。本仓库剩下的唯一未决问题就是上面第 1 项
  （PII 脱敏状态），解决后由 `cmedalign` 项目导入数据继续。
