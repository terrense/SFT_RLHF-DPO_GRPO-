#!/usr/bin/env python3
"""生成发论文级别图表:SFT/DPO loss曲线, GRPO reward, 数据分布, 评测对比。
配色=Okabe-Ito色盲安全(学术标准)。输出 figures/*.png (300dpi)。"""
import json, re, os, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
LOSS = os.path.join(BASE, "archive", "loss")
EVAL = os.path.join(BASE, "archive", "eval")
DATA = os.path.join(BASE, "archive", "data")
FIG = os.path.join(BASE, "figures"); os.makedirs(FIG, exist_ok=True)
V11 = os.path.join(BASE, "..", "data", "eval_sets", "05_final_v11", "train.jsonl")

# Okabe-Ito 色盲安全配色(固定顺序)
OI = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442", "#000000"]
import matplotlib.font_manager as fm
_cn = None
for _f in ["Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "FangSong"]:
    if any(_f.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        _cn = _f; break
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8, "axes.unicode_minus": False,
                     "font.sans-serif": ([_cn] if _cn else []) + ["DejaVu Sans"],
                     "font.family": "sans-serif"})
print("中文字体:", _cn or "未找到(将显示方框)")

def load_loss(name):
    xs, ys = [], []
    p = os.path.join(LOSS, name)
    if not os.path.exists(p): return xs, ys
    for l in open(p, encoding="utf-8"):
        l = l.strip()
        if not l: continue
        try:
            d = json.loads(l)
            if "loss" in d and "current_steps" in d:
                xs.append(d["current_steps"]); ys.append(d["loss"])
        except: pass
    return xs, ys

def smooth(y, k=5):
    if len(y) < k: return y
    return np.convolve(y, np.ones(k)/k, mode="valid")

# ============ 图1: R4 四臂数据配比对照 loss ============
fig, ax = plt.subplots(figsize=(7, 4.3))
arms = [("r4_r4_arm_a_open.jsonl", "A 纯开源"), ("r4_r4_arm_b_seed.jsonl", "B 纯种子"),
        ("r4_r4_arm_c_mixed.jsonl", "C 混合"), ("r4_r4_arm_d_seed2x.jsonl", "D 种子2x")]
for i, (f, lab) in enumerate(arms):
    xs, ys = load_loss(f)
    if xs: ax.plot(xs, ys, color=OI[i], lw=1.6, alpha=0.85, label=lab)
ax.set_xlabel("训练步 (step)"); ax.set_ylabel("训练 loss")
ax.set_title("图1  R4 数据配比对照实验 · 四臂训练 loss (40k子集/1epoch)", fontsize=11)
ax.legend(frameon=False, fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_r4_arms_loss.png")); plt.close()

# ============ 图2: SFT 关键三版 loss (base R5 vs instruct验证 vs 正式版) ============
fig, ax = plt.subplots(figsize=(7, 4.3))
sfts = [("r5_sft_v11_final.jsonl", "R5 Base基座(倒豆子)", OI[4]),
        ("instruct_val.jsonl", "Instruct验证(40k)", OI[1]),
        ("sft_final.jsonl", "正式版 Instruct+心理(全量)", OI[2])]
for f, lab, c in sfts:
    xs, ys = load_loss(f)
    if xs: ax.plot(xs, ys, color=c, lw=1.6, alpha=0.85, label=lab)
ax.set_xlabel("训练步 (step)"); ax.set_ylabel("训练 loss")
ax.set_title("图2  SFT 基座路线对比 · Base vs Instruct 的 loss", fontsize=11)
ax.legend(frameon=False, fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_sft_base_vs_instruct.png")); plt.close()

# ============ 图3: DPO beta 扫参 loss ============
fig, ax = plt.subplots(figsize=(7, 4.3))
for i, b in enumerate(["0.1", "0.3", "0.5"]):
    xs, ys = load_loss(f"dpo_beta{b}.jsonl")
    if xs: ax.plot(xs, ys, color=OI[i], lw=1.6, alpha=0.85,
                   label=f"beta={b}" + (" (选定)" if b == "0.3" else ""))
ax.set_xlabel("训练步 (step)"); ax.set_ylabel("DPO loss")
ax.set_title("图3  DPO 偏好对齐 · beta 扫参 loss", fontsize=11)
ax.legend(frameon=False, fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_dpo_beta_loss.png")); plt.close()

# ============ 图4: GRPO reward + 完成长度 (RL信号, 双子图) ============
txt = open(os.path.join(LOSS, "grpo_reward_extracted.txt"), encoding="utf-8").read()
rewards = [float(x) for x in re.findall(r"'reward': '([0-9.]+)'", txt)]
lengths = [float(x) for x in re.findall(r"completions/mean_length': '([0-9.]+)'", txt)]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
xr = range(1, len(rewards)+1)
a1.plot(xr, rewards, color=OI[2], lw=1.0, alpha=0.4)
if len(rewards) >= 5:
    a1.plot(range(3, 3+len(smooth(rewards))), smooth(rewards), color=OI[2], lw=2.2, label="reward(滑动均值)")
a1.set_ylabel("reward (答对率)"); a1.legend(frameon=False, fontsize=9)
a1.set_title("图4  GRPO 强化学习信号 · reward上升 + 完成长度稳定(无reward hacking)", fontsize=11)
xl = range(1, len(lengths)+1)
a2.plot(xl, lengths, color=OI[0], lw=1.4, alpha=0.85)
a2.axhspan(0, 8, color=OI[3], alpha=0.06)
a2.set_ylabel("完成长度 (token)"); a2.set_xlabel("日志步"); a2.set_ylim(0, max(lengths)*1.3 if lengths else 10)
a2.text(0.98, 0.9, "长度稳定在~2-5 token=只答字母\n未靠灌水骗奖励", transform=a2.transAxes,
        ha="right", va="top", fontsize=8.5, color="#555")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_grpo_reward.png")); plt.close()

# ============ 图5: 数据分布 (task_type柱 + 来源饼) ============
rows = [json.loads(l) for l in open(V11, encoding="utf-8") if l.strip()]
tt = collections.Counter(d["task_type"] for d in rows)
src = collections.Counter(d["source"] for d in rows)
counts = json.load(open(os.path.join(DATA, "counts.json")))
psych = counts["psych_minimax"] + counts["psych_deepseek"] + counts["psych_old"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
TTN = {"symptom_consultation":"症状咨询","health_encyclopedia_qa":"健康百科","triage_guidance":"分诊导诊",
       "pre_consultation_multiturn":"预问诊多轮","test_report_explanation":"报告解读",
       "chronic_disease_management":"慢病管理","medication_guidance_safe":"用药安全",
       "risk_redflag_safety_refusal":"安全红旗","conversation_summary_structured_output":"结构化摘要",
       "hospital_policy_rag_qa":"院务RAG"}
items = tt.most_common()
labels = [TTN.get(k, k) for k, _ in items]; vals = [v for _, v in items]
ypos = range(len(items))
a1.barh(list(ypos), vals, color=OI[1], height=0.72)
a1.set_yticks(list(ypos)); a1.set_yticklabels(labels, fontsize=9); a1.invert_yaxis()
for i, v in enumerate(vals): a1.text(v+400, i, f"{v:,}", va="center", fontsize=8.5, color="#555")
a1.set_xlabel("样本数"); a1.set_title(f"图5a  10类任务分布 (v1.1 共{len(rows):,}条 + 心理{psych}条强化)", fontsize=10.5)
a1.grid(axis="y", alpha=0)
SRCN={"med_zh_real":"真实问诊","Huatuo26M-Lite":"华佗百科","internal_seed_flywheel":"业务飞轮种子",
      "derived_from_seed":"种子派生","gen_minimax_m3":"生成(teacher)","DISC-Med-SFT":"DISC多轮",
      "Chinese-medical-dialogue":"中文医疗对话","shibing624-finetune-zh":"shibing624"}
sitems = src.most_common()
sl = [SRCN.get(k, k) for k, _ in sitems]; sv = [v for _, v in sitems]
a2.pie(sv, labels=sl, colors=[OI[i % len(OI)] for i in range(len(sv))], autopct="%1.0f%%",
       textprops={"fontsize": 8.5}, pctdistance=0.8, wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 1.5})
a2.set_title("图5b  训练数据来源构成", fontsize=10.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_data_distribution.png")); plt.close()

# ============ 图6: 评测对比 (CMB各阶段 + CMExam DPO vs GRPO) ============
def parse_acc(fn):
    p = os.path.join(EVAL, fn)
    if not os.path.exists(p): return None
    m = re.search(r"([0-9.]+)%", open(p, encoding="utf-8").read().split("=====")[-2] if "=====" in open(p,encoding='utf-8').read() else open(p,encoding='utf-8').read())
    return float(m.group(1)) if m else None
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
cmb_names = [("cmb_a_open.log","A纯开源"),("cmb_b_seed.log","B纯种子"),("cmb_c_mixed.log","C混合"),
             ("cmb_d_seed2x.log","D种子2x"),("cmb_r5.log","R5 Base"),("cmb_sft_final.log","正式版")]
cmb_v = []; cmb_l = []
for fn, lab in cmb_names:
    v = parse_acc(fn)
    if v: cmb_v.append(v); cmb_l.append(lab)
bars = a1.bar(cmb_l, cmb_v, color=[OI[1]]*4+[OI[4], OI[2]], width=0.62)
a1.axhline(73.8, color="#999", ls="--", lw=1, label="Base基线 73.8%")
for b, v in zip(bars, cmb_v): a1.text(b.get_x()+b.get_width()/2, v+0.4, f"{v:.1f}", ha="center", fontsize=8.5)
a1.set_ylabel("CMB 准确率 %"); a1.set_ylim(60, 78); a1.legend(frameon=False, fontsize=9)
a1.set_title("图6a  各阶段医学知识保持 (CMB 240题)", fontsize=10.5); a1.grid(axis="x", alpha=0)
plt.setp(a1.get_xticklabels(), rotation=20, ha="right", fontsize=8.5)
cmx = [("eval_dpo_cmexam.log","DPO"),("eval_grpo_cmexam.log","GRPO")]
cv=[]; cl=[]
for fn,lab in cmx:
    v=parse_acc(fn)
    if v: cv.append(v); cl.append(lab)
bars2 = a2.bar(cl, cv, color=[OI[5], OI[2]], width=0.5)
for b, v in zip(bars2, cv): a2.text(b.get_x()+b.get_width()/2, v+0.2, f"{v:.1f}%", ha="center", fontsize=10)
if len(cv)==2:
    a2.annotate(f"+{cv[1]-cv[0]:.1f}pts", xy=(1, cv[1]), xytext=(0.5, cv[1]+1.5),
                fontsize=11, color=OI[2], ha="center", fontweight="bold")
a2.set_ylabel("CMExam 准确率 %"); a2.set_ylim(58, 70)
a2.set_title("图6b  GRPO强化效果 (CMExam 1000题独立集)", fontsize=10.5); a2.grid(axis="x", alpha=0)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_eval_comparison.png")); plt.close()

print("完成! 生成图:", sorted(os.listdir(FIG)))
