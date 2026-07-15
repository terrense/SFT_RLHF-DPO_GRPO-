#!/usr/bin/env python3
"""把心理危机样本(psych_support.jsonl,canonical messages格式)转 sharegpt +
过采样2x + 并入 sft_v11 训练集,产出 train_full_v12.json 供全量训练。"""
import json, hashlib, collections

import os, glob
LAB = "/root/autodl-tmp"
# 汇总所有心理样本来源:双API分片 + 旧文件
PSYCH_FILES = (glob.glob(f"{LAB}/data/sft_pipeline/generated/psych_minimax.jsonl") +
               glob.glob(f"{LAB}/data/sft_pipeline/generated/psych_deepseek.jsonl") +
               glob.glob(f"{LAB}/data/sft_pipeline/generated/psych_support.jsonl"))
BASE_TRAIN = f"{LAB}/data/sft_v11/train_full.json"
OUT = f"{LAB}/data/sft_v12/train_full.json"
os.makedirs(f"{LAB}/data/sft_v12", exist_ok=True)

def to_sharegpt(msgs):
    system = next((m["content"] for m in msgs if m["role"] == "system"), "")
    non_sys = [m for m in msgs if m["role"] != "system"]
    if not non_sys or non_sys[0]["role"] != "user" or non_sys[-1]["role"] != "assistant":
        return None
    for a, b in zip(non_sys, non_sys[1:]):
        if a["role"] == b["role"]:
            return None
    conv = [{"from": "human" if m["role"] == "user" else "gpt", "value": m["content"]}
            for m in non_sys]
    return {"conversations": conv, "system": system}

# 载入心理样本(多来源 + 跨源去重)
def dhash(msgs):
    t = "|".join(m["content"] for m in msgs if m["role"] != "system")
    return hashlib.md5(re.sub(r"\s+", "", t).lower().encode()).hexdigest()
import re
psych = []; seen = set()
for pf in PSYCH_FILES:
    if not os.path.exists(pf):
        continue
    for line in open(pf, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        h = dhash(d["messages"])
        if h in seen:
            continue
        seen.add(h)
        s = to_sharegpt(d["messages"])
        if s:
            psych.append(s)
print(f"心理危机样本 {len(psych)} 条(多源去重后,来自 {len(PSYCH_FILES)} 个文件)")

# 基础训练集
base = json.load(open(BASE_TRAIN, encoding="utf-8"))
print(f"基础训练集 {len(base)} 条")

# 过采样 2x(高危场景强化;心理样本占比仍很小,过采样安全)
merged = base + psych + psych
import random; random.seed(42); random.shuffle(merged)
json.dump(merged, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
print(f"合并后 {len(merged)} 条(心理样本×2过采样={len(psych)*2}),写入 {OUT}")

# 注册 dataset_info
info_path = f"{LAB}/data/dataset_info.json"
info = json.load(open(info_path, encoding="utf-8"))
info["sft_v12_full"] = {"file_name": "sft_v12/train_full.json", "formatting": "sharegpt",
                        "columns": {"messages": "conversations", "system": "system"}}
json.dump(info, open(info_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("已注册 sft_v12_full")
