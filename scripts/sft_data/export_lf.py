#!/usr/bin/env python3
"""导出 LLaMA-Factory sharegpt 格式 + 修 assistant 先手 + 分层子集 + 注册 dataset_info。
assistant 先手修复:预问诊开场白(固定UI话术)并入 system,对话从患者第一句开始。
产出: data/sft_v1/{train_full,dev,smoke_2k,subset_40k}.json + dataset_info.json 增4条目。
"""
import json, hashlib, os, collections

ROOT = "/data/shenxin/rlhf_lab/data"
SRC = f"{ROOT}/sft_pipeline/05_final"
DST = f"{ROOT}/sft_v1"
os.makedirs(DST, exist_ok=True)

def h32(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)

def to_sharegpt(d):
    msgs = d["messages"]
    system = next((m["content"] for m in msgs if m["role"] == "system"), "")
    non_sys = [m for m in msgs if m["role"] != "system"]
    if non_sys and non_sys[0]["role"] == "assistant":
        system = (system + "\n你已经说过开场白:「" + non_sys[0]["content"] + "」"
                  ).strip()
        non_sys = non_sys[1:]
    # 校验交替 human/gpt 且以 gpt 结束
    if not non_sys or non_sys[0]["role"] != "user" or non_sys[-1]["role"] != "assistant":
        return None
    for a, b in zip(non_sys, non_sys[1:]):
        if a["role"] == b["role"]:
            return None
    conv = [{"from": "human" if m["role"] == "user" else "gpt", "value": m["content"]}
            for m in non_sys]
    return {"conversations": conv, "system": system,
            "task_type": d["task_type"], "sid": d["id"]}

def load(split):
    out, bad = [], collections.Counter()
    with open(f"{SRC}/{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            s = to_sharegpt(d)
            if s is None:
                bad[d["task_type"]] += 1
                continue
            out.append(s)
    return out, bad

train, bad_tr = load("train")
dev, bad_dev = load("dev")
print(f"train ok={len(train)} bad={dict(bad_tr)}")
print(f"dev   ok={len(dev)} bad={dict(bad_dev)}")

# 分层子集:每 task_type 内按 md5(sid) 排序取前 N×比例
def stratified(items, n_total):
    by_tt = collections.defaultdict(list)
    for s in items:
        by_tt[s["task_type"]].append(s)
    total = len(items)
    out = []
    for tt, lst in sorted(by_tt.items()):
        k = max(1, round(n_total * len(lst) / total))
        lst.sort(key=lambda s: h32("sub" + s["sid"]))
        out.extend(lst[:k])
    return out

smoke = stratified(train, 2000)
sub40 = stratified(train, 40000)

def dump(name, items):
    for s in items:
        s.pop("task_type", None); s.pop("sid", None)
    with open(f"{DST}/{name}.json", "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
    print(f"{name}: {len(items)}")

# 注意顺序:先切子集再 dump(dump 会剥辅助字段)
dump("smoke_2k", smoke)
dump("subset_40k", sub40)
dump("train_full", train)
dump("dev", dev)

info_path = f"{ROOT}/dataset_info.json"
info = json.load(open(info_path, encoding="utf-8"))
for key, fn in [("sft_v1_smoke2k", "sft_v1/smoke_2k.json"),
                ("sft_v1_40k", "sft_v1/subset_40k.json"),
                ("sft_v1_full", "sft_v1/train_full.json"),
                ("sft_v1_dev", "sft_v1/dev.json")]:
    info[key] = {"file_name": fn, "formatting": "sharegpt",
                 "columns": {"messages": "conversations", "system": "system"}}
with open(info_path, "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)
print("dataset_info.json updated:", ", ".join(k for k in info))
