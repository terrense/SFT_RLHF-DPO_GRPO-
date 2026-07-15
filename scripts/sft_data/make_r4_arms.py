#!/usr/bin/env python3
"""R4 数据对照实验:从 v1.1 train_full 派生 4 个数据臂(sharegpt 格式)。
控制变量=数据构成,其余(rank64/lr2e-4/all-linear/seed42)全同。
ARM-A 纯开源(去种子+派生)  ARM-B 纯种子+派生  ARM-C 完整混合(基线)  ARM-D 混合+种子2x过采样
为可比,A/B/C 采样到同量级(取 min 有效量);D = C + 种子再抄一份。
需要 source 信息 → 从 05_final_v11/train.jsonl(带 source)对齐 id。
"""
import json, hashlib, collections, os

ROOT = "/data/shenxin/rlhf_lab/data"
SRC = f"{ROOT}/sft_pipeline/05_final_v11/train.jsonl"
DST = f"{ROOT}/sft_v11_arms"
os.makedirs(DST, exist_ok=True)
SEED_SRC = {"internal_seed_flywheel", "derived_from_seed"}

def h32(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)

def to_sharegpt(d):
    msgs = d["messages"]
    system = next((m["content"] for m in msgs if m["role"] == "system"), "")
    non_sys = [m for m in msgs if m["role"] != "system"]
    if non_sys and non_sys[0]["role"] == "assistant":
        system = (system + "\n你已经说过开场白:「" + non_sys[0]["content"] + "」").strip()
        non_sys = non_sys[1:]
    if not non_sys or non_sys[0]["role"] != "user" or non_sys[-1]["role"] != "assistant":
        return None
    for a, b in zip(non_sys, non_sys[1:]):
        if a["role"] == b["role"]:
            return None
    conv = [{"from": "human" if m["role"] == "user" else "gpt", "value": m["content"]}
            for m in non_sys]
    return {"conversations": conv, "system": system}

rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
seed = [d for d in rows if d.get("source") in SEED_SRC]
opensrc = [d for d in rows if d.get("source") not in SEED_SRC]
print(f"total={len(rows)} seed+derived={len(seed)} open={len(opensrc)}")

# 为公平对比,A/B 各取 N=min(len(seed),40000) 量级;C 用全量;D=C+seed
N = min(len(seed), 40000)
def sample(lst, n):
    lst = sorted(lst, key=lambda d: h32("s" + d["id"]))
    return lst[:n]

def dump(name, items):
    out = [s for s in (to_sharegpt(d) for d in items) if s]
    with open(f"{DST}/{name}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"{name}: {len(out)}")

dump("arm_a_open", sample(opensrc, N))            # 纯开源
dump("arm_b_seed", sample(seed, N))               # 纯种子+派生
dump("arm_c_mixed", sample(opensrc, N // 2) + sample(seed, N // 2))  # 混合等量
dump("arm_d_seed2x", sample(opensrc, N // 2) + sample(seed, N // 2) + sample(seed, N // 2))  # 种子2x
print("R4 arms ready in", DST)
