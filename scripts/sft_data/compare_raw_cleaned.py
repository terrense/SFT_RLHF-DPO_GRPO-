#!/usr/bin/env python3
"""对比原始 20k 与清洗后 15.5k:清洗到底洗掉了什么(scene/risk/triage/急诊覆盖)。"""
import json, collections

RAW = "/data/shenxin/rlhf_lab/data/medicine_dataset/pre_consultation_multiturn.jsonl"
CLEAN = "/data/shenxin/rlhf_lab/data/medicine_dataset/pre_consultation_multiturn.cleaned.jsonl"

def load_stats(path):
    sc, rl, tl, ids = collections.Counter(), collections.Counter(), collections.Counter(), set()
    turns = collections.Counter()
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                d = json.loads(line)
            except Exception:
                sc["<bad_json>"] += 1
                continue
            m = d.get("meta", {})
            sc[m.get("scene", "?")] += 1
            rl[d.get("risk_level", "?")] += 1
            tl[str(m.get("triage_level", "?"))] += 1
            turns[d.get("num_turns", m.get("num_turns", -1))] += 1
            ids.add(d.get("id"))
    return n, sc, rl, tl, turns, ids

n1, sc1, rl1, tl1, t1, ids1 = load_stats(RAW)
n2, sc2, rl2, tl2, t2, ids2 = load_stats(CLEAN)
removed = ids1 - ids2

print(f"原始 {n1} 条 / 清洗后 {n2} 条 / 被移除 {len(removed)} 条")
print("原始   scene :", dict(sc1.most_common()))
print("清洗后 scene :", dict(sc2.most_common()))
print("原始   risk  :", dict(rl1.most_common()))
print("清洗后 risk  :", dict(rl2.most_common()))
print("原始   triage:", dict(sorted(tl1.items())))
print("清洗后 triage:", dict(sorted(tl2.items())))
print("原始   turns :", dict(sorted(t1.items(), key=lambda x: str(x[0]))))
print("清洗后 turns :", dict(sorted(t2.items(), key=lambda x: str(x[0]))))

# 被移除样本的 scene/risk 画像
rm_sc, rm_rl, rm_tl = collections.Counter(), collections.Counter(), collections.Counter()
examples = []
with open(RAW, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("id") in removed:
            m = d.get("meta", {})
            rm_sc[m.get("scene", "?")] += 1
            rm_rl[d.get("risk_level", "?")] += 1
            rm_tl[str(m.get("triage_level", "?"))] += 1
            if m.get("scene") not in (None, "normal") and len(examples) < 3:
                examples.append({"id": d["id"], "scene": m.get("scene"),
                                 "persona": (m.get("persona") or "")[:80]})
print("\n被移除样本 scene :", dict(rm_sc.most_common()))
print("被移除样本 risk  :", dict(rm_rl.most_common()))
print("被移除样本 triage:", dict(sorted(rm_tl.items())))
print("非normal场景被移除示例:", json.dumps(examples, ensure_ascii=False))
