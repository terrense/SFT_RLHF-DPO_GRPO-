# 云训练全程操作日志(唯一事实来源 · 换电脑接手看这个)

> 目的:R4/R5 全链路在租赁 4×5090 云机上的**完整操作记录**。任何人凭此文件即可准确接手继续操作。
> 双份存放:①云机 `/root/autodl-tmp/CLOUD_RUN_LOG.md` ②本地 `F:\rlhf_lab_cloud_kit\docs\CLOUD_RUN_LOG.md`。每步操作后两份同步更新。

---

## 0. 连接信息与背景

**云机(AutoDL,租于 2026-07-13)**
- SSH: `ssh -p 21399 root@connect.bjb2.seetacloud.com`  密码:`2/XVJazXLMQx`
- 非交互(Windows/plink):`plink -batch -P 21399 -pw "2/XVJazXLMQx" root@connect.bjb2.seetacloud.com "<cmd>"`
- 传文件:`pscp -P 21399 -pw "2/XVJazXLMQx" <本地> root@connect.bjb2.seetacloud.com:<远程>`
- 首次连接需接受 host key:`echo y | plink -P 21399 -pw "..." root@... "hostname"`

**本地资产** `F:\rlhf_lab_cloud_kit\`(见 README_清单.md):data/sft_v11 + eval_sets + models/Qwen3-8B-Base(16G)+ scripts + configs/r4r5 + docs。

**任务目标**:用 v1.1 数据集(154,476条,10类)跑 R4 数据对照(A/B/C/D 四臂并行)+ R5 全量终版,产出 SFT adapter(=后续 DPO 起点)。
**锁定配置**(R1-R3 扫参已定):rank=64 alpha=128 lr=2e-4 lora_target=all cutoff=2048 seed=42 bf16 cosine warmup0.03 flash_attn=fa2。
**背景全史**见同目录 EXPERIMENT_LOG.md(§1-17)。

**省钱铁律**:付费机只做"传+跑",不现写脚本(都在 F: 备好);R4 四卡并行不串行;跑完立即 vLLM 抽查再决定续租。

---

## 1. 操作日志(按时间倒序追加,最新在上)

<!-- 每步格式:### [时间] 动作 → 命令 → 结果/结论 -->

### [2026-07-13] 开机 · 环境探测 ✅ 绿灯
- **host key**(plink 用):`SHA256:liZ36vNCsNcNdXeWs4f+g5ZIhPM/ZihP834vxs8Ulqc`
- 连接命令:`plink -batch -P 21399 -pw "2/XVJazXLMQx" -hostkey "SHA256:liZ36vNCsNcNdXeWs4f+g5ZIhPM/ZihP834vxs8Ulqc" root@connect.bjb2.seetacloud.com "<cmd>"`
- **GPU**:4×NVIDIA GeForce RTX 5090,各 32607 MiB,驱动 580.76.05 ✅
- **torch**:base 环境自带 **2.8.0+cu128 / CUDA 12.8** ✅(正是 5090/sm_120 所需)
- **python 路径**:`/root/miniconda3/bin/python`(非交互 shell 未激活 conda,一律用全路径 `/root/miniconda3/bin/{python,pip}`)
- **磁盘**:系统盘 `/` 30G;数据盘 `/root/autodl-tmp` **550G**(充裕)。所有产物放数据盘。
- **内存** 754G / **CUDA toolkit** /usr/local/cuda-12.8 / **hf-mirror** 可达(HTTP200)
- 缺件:llamafactory / deepspeed / vllm(待装,torch 已就绪不要动它)
- **结论:环境完全满足,开始装训练依赖 + 拉模型 + 传数据。**

### [2026-07-13] 环境安装 + 数据 + 四臂 ✅
- **依赖**:`/root/miniconda3/bin/pip install -U llamafactory deepspeed huggingface_hub`(aliyun源)。装后 torch 仍 **2.8.0+cu128**(未被动)✅。llamafactory 就绪。
- **flash-attn 决策:不装,用 sdpa**。5090=sm_120 无 flash-attn 现成 wheel,现编译要 20-40 分纯烧钱;torch2.8 sdpa 在 5090 够快,跨机不比吞吐。已把 configs/r4r5/*.yaml 的 `flash_attn: fa2` 改为 `sdpa`。
- **数据**:F: 传上 data/sft_v11(train_full/subset_40k/dev/dev_1k)+ eval_sets(CMB/cmexam/05_final_v11)+ dataset_info.json;scripts + configs + docs 全传。约 2.4MB/s。
- **模型**:从 hf-mirror 拉 `Qwen/Qwen3-8B-Base` → /root/autodl-tmp/models/Qwen3-8B-Base。**注意:新版 hub 命令是 `hf download` 不是 `huggingface-cli download`**(旧语法报help)。拉取中。
- **R4 四臂**(make_r4_arms.py,已改云路径 ROOT=/root/autodl-tmp/data、SRC=eval_sets/05_final_v11/train.jsonl):arm_a_open 29,151 / arm_b_seed 29,151 / arm_c_mixed 29,150 / arm_d_seed2x 43,725,存 data/sft_v11_arms/,已注册 dataset_info 键 r4_arm_*。
- **dataset_info 已注册**:sft_v11_{full,40k,dev,dev1k} + r4_arm_{a_open,b_seed,c_mixed,d_seed2x}。

### [2026-07-13] 坑#1:torchaudio 版本不匹配(已修)
- 模型16G拉完(EXIT=0,5分片)。首次 smoke 报 `OSError: libcudart.so.13: cannot open shared object file`。
- **根因**:装 llamafactory 时带进 **torchaudio 2.11.0**(为 CUDA13 编译),与 torch2.8/cu128 不匹配;llamafactory 的 mm_plugin `import torchaudio` 触发加载 libcudart.so.13 失败。
- **修复**:`pip install torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128`(换成匹配版,不卸载以免 llamafactory import 崩)。验证 torch/audio/vision 全 2.8+cu128、cuda_ok True。
- **教训**:装完训练框架务必查 `torch/torchaudio/torchvision` 三者版本是否对齐同一 cu 版本,不齐会在 import 期就崩。

### [2026-07-13] smoke 冒烟 ✅ 全绿 → R4 启动
- smoke(GPU0,200样本):EXIT=0,loss 1.64→1.40 正常下降,adapter 698MB 保存成功,37秒。**5090 全链路(加载/sdpa/训练/保存)验证通过。**
- **R4 四臂并行启动**(run_r4.sh,4卡各一臂):GPU0=arm_a_open(纯开源29,151)/ GPU1=arm_b_seed(纯种子29,151)/ GPU2=arm_c_mixed(混合29,150)/ GPU3=arm_d_seed2x(种子2x 43,725)。配置 rank64/lr2e-4/sdpa/1epoch。产物 outputs/r4/r4_arm_*/,日志 outputs/r4_r4_arm_*.log。
- 生成的臂 yaml:configs/r4r5/gen_r4_arm_*.yaml。

### [2026-07-13] 坑#2:R4 batch=4 显存 OOM(已修)
- 首次启动 R4,GPU 0/2/3 训练中(30-32GB),**GPU1 的 arm_b_seed 第1步就 CUDA OOM**(要 3.7G 只剩 2G)。arm_c 也顶到 32007MiB 满载濒危。
- **根因**:r4_arm_template 用 `batch_size=4 × cutoff2048`,纯种子臂(arm_b)全是长多轮对话(token→2048),4条长序列激活爆 32GB。smoke 用 batch=2 所以没暴露。
- **修复**:batch_size 4→2、grad_accum 4→8(有效batch 仍16,激活显存减半)+ run_r4.sh 加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`(抗碎片)。四臂全 kill 重启(pkill -f 又自杀SSH会话致 exit128,但进程确实杀掉、显存归0)。
- **教训**:32GB 卡跑 8B LoRA + cutoff2048,per_device batch 上限=2;长序列任务(多轮)尤其吃激活显存。

### [2026-07-13] 坑#3:训练内评测 batch=8 在 epoch 末 OOM(已修,代价1h)
- arm_a_open **训练跑完(1822/1822)**,但在 epoch 末的训练内评测(eval_strategy: epoch, per_device_eval_batch_size=8)OOM 崩溃,崩在 trainer.save_model() 之前 → **adapter 没保存,1小时白跑**。b/c/d 跑到终点会撞同一堵墙。
- **根因**:eval batch=8 × cutoff2048,评测阶段激活显存在已占 26GB 基础上再要 5.6G,爆 32GB。且 save_steps=100000 无中间 checkpoint 可续。
- **修复**:①`eval_strategy: "no"`(训练内评测本就多余,CMB/hard_eval 后面单独做)+ 注释 eval_dataset + eval batch→1;②`save_steps: 600`(每臂 1822/2733 步,留中间 checkpoint 兜底,不再全白跑)。四臂全 kill 重启。
- **教训**:①训练内 eval 的 batch 要单独控制(eval 无梯度但激活仍占显存,32GB卡上 eval batch 也得≤2);②长训必须开中间 save_steps 兜底,别设超大值;③我们有独立评测脚本,训练内 eval 可直接关掉省心省显存。

### [2026-07-13] R4 评测方法失误 + 倒豆子根因诊断(重要)
- **评测脚本 bug**:eval_hardeval 的 pre_consultation 评分用"问号≤1"判定,把倒豆子/复读/乱码的垃圾输出误判满分。hard_eval 行为分不可信(CMB知识分可信)。教训:评分逻辑必须先小样本验证。
- **倒豆子根因诊断(逐步排除)**:
  1. 停止符假说→**推翻**:诊断脚本对比 eos=endoftext vs eos含im_end,输出几乎一样都倒豆子(模型没输出干净的im_end,`�`乱码是坏字节)。→停止符不是主因。
  2. 数据构造检查→**完全正确**:show4.py 打印4条完整多轮,每个turn结尾都有<|im_end|>(消息数=im_end数);check_label_cloud.py 验证1000条多轮,3912个assistant轮=3912个处于loss区(label≠-100)的im_end,1.00/轮一个不漏。→数据/标签没错。
  3. 内容行为→**学对了**:选项式提问/鉴别诊断/科室推荐/急诊升级判断都正确。
  4. **结论=欠训练**:40k/1ep 火候不够,模型没把已标注好的im_end停止符学到收敛→停不住+复读+乱码。数据对、标签对、方法对,唯缺训练量。R5全量154k×2ep(训练量~8倍)是正解。

### [2026-07-13] R5 全量终版启动(GPU3)
- 配置 r5_final.yaml:rank64/lr2e-4/all/cutoff2048/seed42/**2epoch**/packing/sdpa。
- **启动前堵两个已知坑**:per_device_train_batch_size 4→2(坑#2 OOM)、eval_strategy steps→"no"(坑#3 eval OOM)。save_steps2000+total_limit2兜底,expandable_segments。
- GPU3(我的卡);GPU0/2 是用户的不碰;GPU1 还在跑 c_mixed 评测。R5 PID 34839。数据=sft_v11_full 146,807条。预计 10-14h。

### R4 四臂最终结果(供R5后参考;行为分因评分bug仅供粗看)
- CMB知识分:B纯种子73.8%(=base,最佳保持)/C混合73.3%/A纯开源72.5%/D种子2x71.7%
- 数据覆盖洞察:纯种子缺安全/RAG(那些是生成数据,在开源池);混合C兼顾→R5用全量v1.1(含全部类别)最稳

### [2026-07-13 20:27] 夜间全自动流水线启动(nightrun.sh v2,两卡GPU1/3)
**卡分工**:GPU0/2=用户在用(各占8GB,绝不碰);GPU1/3=我的(R5在GPU3)。
**MiniMax key**:用户已授权,从5133取 .minimax_env 传到云 /root/autodl-tmp/.minimax_env(含MINIMAX_API_KEY/BASE_URL/MODEL),load_key验证OK。本地副本已删。
**编排(setsid独立跑,PID67137,掉线不影响)**:
1. 等R5训完(~凌晨2点)→ 2. 验证adapter → 3. 倒豆子验收 test_r5_stop.py(GPU3,正确停止符<|im_end|>) → 4. R5 CMB评测(GPU3) → 5. DPO采样 gen_dpo_pairs(GPU1分片0+GPU3分片1,N_PROMPTS=1500,每prompt采4候选) → 6. MiniMax-M3 judge打分(不占卡,内置长度偏置防护) → 7. 合并 data/rlhf/dpo/dpo_pairs.jsonl。
**里程碑写 outputs/nightrun.log**,持久监控在每节点唤醒我更新本地日志。
**产物预期(明早)**:R5倒豆子测试结论、R5 CMB分、~1500条DPO偏好对(=DPO训练输入)。
**注**:DPO采样两卡5h窗口约产1200-1500 prompt候选;若未跑满partial也可用。DPO训练(下一步)等用户回来定。

### [2026-07-14 01:44] ⚠️R5结果:倒豆子未解决,锁定根因=base基座+对话模板不适配
**R5训练本身成功**:2epoch跑完,**最终loss=1.055**(R4是~1.22,大幅下降,充分收敛,数据学透了)。
**但倒豆子验收 0/4 停住**(test_r5_stop.py,正确停止符<|im_end|>):4条预问诊全部一次生成400token不停、大量`�`乱码。**CMB降到70.8%**(base73.8)。
**内容其实全对**:选项式提问/参考信息鉴别诊断/科室推荐/急诊升级判断都正确——问题纯在"生成控制机制"(停不住+乱码),不是内容/数据。

**逐一排除假说(都有证据)**:
1. ❌欠训练:R5 loss 1.055 已充分收敛,倒豆子反而比R4更重→证伪。
2. ❌停止符配置:diag_stop.py对比eos=endoftext vs eos含im_end,输出一样→证伪(模型没输出干净im_end)。
3. ❌<think>数据污染:check_think.py查全量,含<think>的assistant轮=**0%**(8个来源全0)→证伪。数据文件干净。
4. ✅**根因锁定**:`<think>\n\n</think>`是**Qwen3对话模板自动注入**的(<think>=id151667/</think>=151668,special token)。我们用**Qwen3-8B-Base(基座,无对话对齐)**+**qwen3 instruct对话模板(带thinking+im_end停止机制)**。base预训练没建立<|im_end|>的对话停止语义,SFT要从零教,即使154k×2ep充分训练也学不牢→停不住+采样到坏字节(�)。

**核心建议(待用户回来决策)**:
- **首选:改用 Qwen3-8B-Instruct 做SFT基座**。它已具备对话机制(im_end停止/多轮),我们只需教医疗行为,倒豆子/乱码大概率消失,还省训练量。**当初选base是为CPT,但CPT已放弃(实测无提升)→SFT就该用Instruct基座**。这可能是路线层面的修正。
- 备选:换不带thinking的简化模板重训;或小规模对照(instruct基座 vs base)验证再定。
- 数据无需大改(构造/标签/内容都验证正确)。

**DPO采样**:按用户授权继续(GPU1/3,shard各450/750)。但**注意:DPO基于这个倒豆子的R5模型采样,偏好数据质量存疑**;judge会惩罚倒豆子输出。建议用户回来后:先修SFT基座(→Instruct)重训,再重做DPO。今晚的DPO数据作为流程验证/参考,不宜直接用于正式DPO训练。

### [2026-07-14 14:15] ✅✅ 倒豆子解决!Instruct基座+nothink模板验证成功
**方案**:Qwen3-8B-**Instruct**基座 + `qwen3_nothink`模板 + lr1e-4/rank32/40k/1ep,双卡GPU1+3 DDP(NCCL_P2P_DISABLE=1),1h05m训完,adapter 349MB。
**倒豆子验收(test_instruct_stop.py,正确停止符)**:**4/4条单轮停住,0乱码,0倒豆子**。模型问一个临床问题就停等患者:
- "嗓子疼是从什么时候开始的?持续多久了?"(仅12token)
- "这种绞痛是持续性的还是间歇性的?"(18token)
- "孩子发烧和咳嗽是持续两天了,还是今天才开始的?"(16token)
对比昨晚base R5:一口气倒多轮+400token撑满+满屏`�`乱码。
**根因确证**:问题在基座+模板,不在数据。Base无对话对齐学不会<|im_end|>停止;qwen3模板注入<think>致乱码。换Instruct(自带对话/停止机制)+nothink(不注入think)后彻底解决。**用户昨晚提议换Instruct方向正确**,加"关thinking+保守lr"三修正生效。40k/1ep小验证成功,未浪费全量。
**关键工程坑**:①双卡启动需 FORCE_TORCHRUN=1 + PATH含/root/miniconda3/bin(否则torchrun找不到);②5090双卡需NCCL_P2P_DISABLE=1;③只用GPU1/3,GPU0/2永远是用户的。
**下一步**:上全量154k×2ep(Instruct+nothink+双卡)出正式交付SFT模型 → 重做DPO(基于好模型)→ GRPO。DPO题库/脚本已备(CMExam切分5590/1000;gen_dpo_pairs.py)。

### [2026-07-14 21:05] 正式SFT模型验收(Instruct+nothink+心理强化,四卡154k+634心理×2ep)
**SFT训完**:2362步/2epoch/2h03m,最终loss 0.94,adapter 349MB(outputs/sft_final)。
**五场景验收**:
- ✅✅ **倒豆子彻底解决**:全场景单轮停住0乱码。**多轮连续预问诊优秀**:逐轮追问(澄清→选项列表→伴随症状)→末轮给参考信息+科室推荐,一轮一停。
- ✅ 报告解读(白细胞低判读正确+建议就医不确诊)、用药安全(拒双抗生素+讲耐药+引导就医)都好。
- ⚠️ **心理危机仍弱**:"不想活了"→模型平静问"持续多久"无共情无热线。**根因=system prompt依赖**:测试用"分诊助手"prompt,模型按分诊处理;心理危机训练数据用"心理支持助手"prompt,仅在该角色下激活。**这是产品层问题:需危机检测覆盖一切(无视system prompt),光靠绑定prompt的训练数据不够。** 待办:①危机检测路由 或 ②训练"危机override"行为无视system prompt。
- ⚠️ CMB 65.0%(base 73.8%但那是Qwen3-Base基线;现换Instruct基座,需单独测Instruct裸模型CMB才知真降还是基线差异)。待办:补测Instruct基线CMB。
**结论:核心目标(倒豆子)达成,模型可用于DPO;心理危机需产品层危机路由(记入RLHF后待办)。**
**夜间流水线继续**:DPO采样(四卡1600prompt)→双API打分→DPO beta 0.1/0.3/0.5扫参→各版本输出存好→停在GRPO前。

### [2026-07-15] 🎉 完整RLHF全链路跑通(SFT→DPO→GRPO)
**GRPO(RLVR)**:trl GRPOTrainer(导入钩子绕weave/mergekit/llm_blender + warnings_issued兼容补丁 + merge(DPO)后包新LoRA),CMExam规则奖励(答对=1),四卡400步~27min。reward 0.49→0.68上升,完成长度稳定2-5token(**无reward hacking**),KL~0.0003受控。
**答对率验收(CMExam 1000题独立评测集)**:DPO 62.6% → **GRPO 65.5%(+2.9pts)**。RLVR真实提升。
**行为复测(五场景)**:GRPO后问诊/心理危机共情/报告解读/用药安全**全部保持**,未因RL强化选择题而训坏对话能力(无灾难性遗忘)。
**全链路成果**:
| 阶段 | 关键指标/行为 |
|---|---|
| SFT(Instruct+nothink+心理强化,四卡154k+634×2ep) | 倒豆子彻底解决,多轮问诊优秀,CMB 65% |
| DPO(1434偏好对,双API打分,beta=0.3) | 心理危机从冷漠→共情;倒豆子保持解决 |
| GRPO(CMExam RLVR,四卡400步) | CMExam 62.6→65.5%(+2.9),行为无损 |
**产物**:outputs/{sft_final, dpo_beta0.3, grpo_final}。**遗留待办(记入)**:①心理危机在"分诊助手"prompt下仍触发分诊(需危机检测路由,产品层)②补测Instruct裸模型CMB基线③veRL路线B(工业级推训分离,独立环境装vLLM)。
**项目结论:从倒豆子的坏模型,到走完SFT→DPO→GRPO完整RLHF链路的可用医疗预问诊助手。核心方法论(小验证再放量/多维度评测/reward hacking监控/基座与模板匹配)全程实践。**

### [2026-07-15] 用户亲自交互评测GRPO最终模型(真实体验,暴露能力边界)
用户用chat服务(Instruct+DPO0.3+GRPO全栈)亲测,发现:
**✅ 优点**:①多轮问诊骨架正确(腹泻性状→腹痛部位→伴随症状→给结论,逐步追问);②**对话结束后稳定咬住结论,用户怎么撩都不扯闲篇**(GRPO强化"咬定答案"的正向副产品)。
**❌ 缺陷(真实评测抓到,脚本测不出)**:
1. **分诊科室不准**:腹泻+脐周绞痛→推"外科"(应消化内科/急诊内科)。根因=种子数据科室标注本身不准(飞轮生成时判断粗),GRPO只强化CMExam选择题未管分诊准确性。
2. **跑题不纠偏**:用户中途乱说一步,模型不提醒"和前文对不上",硬塞模板。缺澄清/纠错能力。
3. **首轮闲聊直接全错**:"小护你好"→模型凭空幻觉"发烧38度乏力"等不存在症状,硬套问诊模板。缺开场/闲聊→引导说症状 的样本。
**结论:问诊骨架学会,但缺鲁棒性(闲聊/跑题/纠偏)+分诊准确性。均为数据覆盖问题,非方法问题。修法=补三类数据(开场闲聊引导/跑题澄清/准确科室标注)重训。记入待办。** 教训:交互测试不可替代——脚本五场景全绿,用户两句话("你好"/"肚子疼")就暴露幻觉+跳步。

### [2026-07-15] veRL路线B启动:装vLLM(RL rollout正解)+试veRL(工业级推训分离)
用户要求:补装vLLM(RL的rollout该用它,之前GRPO用transformers生成是图快),试veRL,充分记录。**全新独立环境 /root/autodl-tmp/verl_env(venv),绝不碰训练/对话环境**。记录见 docs/VERL_SETUP_LOG.md。计划:装vLLM→验证5090可跑→装veRL→vLLM rollout重跑GRPO→对比transformers版(速度/reward/答对率)。

### 历史待办队列(R4部分已完成) → 每臂 CMB240 + hard_eval 分桶评测,比 A/B/C/D(种子值多少分/开源稀释否/过采样有效否)
2. R5 全量2ep(sft_v11_full,主卡)→ 交付 adapter(=DPO起点)
3. 结果回填本日志 + 回传 F: + EXPERIMENT_LOG §18
