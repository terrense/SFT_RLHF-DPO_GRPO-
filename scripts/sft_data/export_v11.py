#!/usr/bin/env python3
"""v1.1: 05_final_v11 → LLaMA-Factory sharegpt + assistant先手修复 + 子集 + 注册。
复用 v1 export 逻辑,仅换输入/输出目录与注册键(sft_v11_*)。"""
import json, hashlib, os, collections

ROOT = "/data/shenxin/rlhf_lab/data"
SRC = f"{ROOT}/sft_pipeline/05_final_v11"
DST = f"{ROOT}/sft_v11"
os.makedirs(DST, exist_ok=True)

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
    return {"conversations": conv, "system": system, "task_type": d["task_type"], "sid": d["id"]}

def load(split):
    out, bad = [], collections.Counter()
    with open(f"{SRC}/{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            s = to_sharegpt(d)
            if s is None:
                bad[d["task_type"]] += 1
            else:
                out.append(s)
    return out, bad

train, bad_tr = load("train")
dev, _ = load("dev")
print(f"train ok={len(train)} bad={dict(bad_tr)}; dev ok={len(dev)}")

def stratified(items, n_total):
    by_tt = collections.defaultdict(list)
    for s in items:
        by_tt[s["task_type"]].append(s)
    total = len(items); out = []
    for tt, lst in sorted(by_tt.items()):
        k = max(1, round(n_total * len(lst) / total))
        lst.sort(key=lambda s: h32("sub" + s["sid"]))
        out.extend(lst[:k])
    return out

sub40 = stratified(train, 40000)
dev1k = sorted(dev, key=lambda s: h32("d" + s["sid"]))[:1000]

def dump(name, items):
    for s in items:
        s.pop("task_type", None); s.pop("sid", None)
    with open(f"{DST}/{name}.json", "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
    print(f"{name}: {len(items)}")

dump("subset_40k", sub40)
dump("dev_1k", dev1k)
dump("train_full", train)
dump("dev", dev)

info_path = f"{ROOT}/dataset_info.json"
info = json.load(open(info_path, encoding="utf-8"))
for key, fn in [("sft_v11_40k", "sft_v11/subset_40k.json"),
                ("sft_v11_full", "sft_v11/train_full.json"),
                ("sft_v11_dev", "sft_v11/dev.json"),
                ("sft_v11_dev1k", "sft_v11/dev_1k.json")]:
    info[key] = {"file_name": fn, "formatting": "sharegpt",
                 "columns": {"messages": "conversations", "system": "system"}}
json.dump(info, open(info_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("registered:", "sft_v11_40k/full/dev/dev1k")
