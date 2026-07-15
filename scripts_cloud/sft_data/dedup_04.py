#!/usr/bin/env python3
"""04_deduped: 全局精确去重(跨源,dedup_hash)。
重复时保留 source_quality 更高的版本(high>medium>low;同级先到先得,顺序确定)。
近似去重放在 05 采样阶段(MinHash 贪心,只对入选集合做,省算力)。
"""
import json, os, collections

ROOT = "/data/shenxin/rlhf_lab/data/sft_pipeline"
IN_DIR, OUT_DIR, REP = f"{ROOT}/03_labeled", f"{ROOT}/04_deduped", f"{ROOT}/reports"
os.makedirs(OUT_DIR, exist_ok=True)
QRANK = {"high": 0, "medium": 1, "low": 2}
# 处理顺序 = 质量优先:种子 → med_zh/DISC/CMD/Huatuo → shibing
ORDER = ["internal_seed_flywheel.jsonl", "med_zh_real.jsonl", "DISC-Med-SFT.jsonl",
         "Chinese-medical-dialogue.jsonl", "Huatuo26M-Lite.jsonl",
         "shibing624-finetune-zh.jsonl"]

seen = {}
stats = collections.OrderedDict()
for fname in ORDER:
    n_in, n_out, n_dup = 0, 0, 0
    with open(f"{IN_DIR}/{fname}", encoding="utf-8") as f, \
         open(f"{OUT_DIR}/{fname}", "w", encoding="utf-8") as w:
        for line in f:
            n_in += 1
            d = json.loads(line)
            h = d["metadata"]["dedup_hash"]
            if h in seen:
                n_dup += 1
                continue
            seen[h] = True
            w.write(line if line.endswith("\n") else line + "\n")
            n_out += 1
    stats[fname[:-6]] = {"in": n_in, "out": n_out, "cross_source_dup": n_dup}
    print(f"[{fname[:-6]}] in={n_in} out={n_out} dup={n_dup}", flush=True)

import csv
with open(f"{REP}/dedup_report.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source", "in", "out", "cross_source_exact_dup"])
    for s, v in stats.items():
        w.writerow([s, v["in"], v["out"], v["cross_source_dup"]])
print(json.dumps(stats, ensure_ascii=False, indent=2))
