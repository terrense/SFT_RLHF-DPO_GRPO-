#!/usr/bin/env python3
"""图7: DeepSpeed 调参单变量对照(峰值显存 + 吞吐)。Qwen2.5-0.5B 全参, 四卡, 60步, global batch=32 恒定。
配色=Okabe-Ito 色盲安全。输出 figures/fig7_deepspeed_tuning.png (300dpi)。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, "figures"); os.makedirs(FIG, exist_ok=True)
OI = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442", "#000000"]
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

# 单变量 sweep 实测(每次只从 ZeRO-2 基线改一个旋钮; global batch 恒为 32)
labels = ["base\n(micro8)", "micro=2\n(accum4)", "grad_ckpt\n(on)",
          "bucket\n20M", "bucket\n1000M", "overlap\noff"]
peak_gb = np.array([21575, 9547, 21575, 21295, 21875, 21435]) / 1024.0
sps     = np.array([41.7, 34.3, 43.6, 43.6, 41.7, 41.7])
# 突出 micro=2(唯一显著项)与基线
bar_c = [OI[7], OI[2], OI[1], OI[1], OI[1], OI[1]]  # base黑, micro2绿, 其余蓝

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
x = np.arange(len(labels))

b1 = ax1.bar(x, peak_gb, color=bar_c, width=0.62, edgecolor="white", linewidth=1.2)
ax1.set_ylabel("峰值显存 / 卡 (GB)")
ax1.set_title("① 峰值显存:micro_batch 是唯一有效杠杆", fontsize=11)
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylim(0, 24)
for r, v in zip(b1, peak_gb):
    ax1.text(r.get_x()+r.get_width()/2, v+0.3, f"{v:.1f}", ha="center", fontsize=9)
ax1.axhline(peak_gb[0], color=OI[4], ls="--", lw=0.9, alpha=0.7)
ax1.annotate("micro8→2:\n−56% 显存", xy=(1, peak_gb[1]), xytext=(1.55, 14),
             fontsize=9, color=OI[4],
             arrowprops=dict(arrowstyle="->", color=OI[4], lw=1.1))

b2 = ax2.bar(x, sps, color=bar_c, width=0.62, edgecolor="white", linewidth=1.2)
ax2.set_ylabel("吞吐 (样本/秒, 含初始化)")
ax2.set_title("② 吞吐:micro=2 慢 18%(4× 微步),其余持平", fontsize=11)
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylim(0, 50)
for r, v in zip(b2, sps):
    ax2.text(r.get_x()+r.get_width()/2, v+0.5, f"{v:.1f}", ha="center", fontsize=9)
ax2.axhline(sps[0], color=OI[4], ls="--", lw=0.9, alpha=0.7)

fig.suptitle("DeepSpeed 调参单变量对照 · Qwen2.5-0.5B 全参 · 四卡 · global batch=32 恒定",
             fontsize=12, y=1.02)
fig.tight_layout()
out = os.path.join(FIG, "fig7_deepspeed_tuning.png")
fig.savefig(out, bbox_inches="tight")
print("saved:", out, "| 中文字体:", _cn or "未找到")
