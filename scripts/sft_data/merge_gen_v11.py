#!/usr/bin/env python3
"""合并 Codex 生成数据到主管线,生成 04_deduped 增量文件(gen_*.jsonl)。
generated 已是 canonical + 带 task_type,只需:补全 metadata 键 → 轻清洗(空/超短/ASCII泄漏)
→ 计算 dedup_hash → 与现有 04_deduped 全池跨源精确去重。近重留给 05 的 MinHash。
"""
import json, os, re, hashlib, collections

ROOT = "/data/shenxin/rlhf_lab/data/sft_pipeline"
GEN = f"{ROOT}/generated"
DEDUP = f"{ROOT}/04_deduped"
REP = f"{ROOT}/reports"
WS = re.compile(r"\s+")
ASCII4 = re.compile(r"[A-Za-z]{4,}")

SRCS = {
    "risk_redflag_safety_refusal": f"{GEN}/risk_redflag_safety_refusal.jsonl",
    "test_report_explanation": f"{GEN}/test_report_explanation.jsonl",
    "hospital_policy_rag_qa": f"{GEN}/hospital_policy_rag_qa.jsonl",
}

def norm(s):
    return WS.sub("", s).lower()

def dhash(msgs):
    t = "|".join(m["content"] for m in msgs if m["role"] != "system")
    return hashlib.md5(norm(t).encode()).hexdigest()

# 载入现有全池 hash(精确去重基准)
seen = set()
for fn in os.listdir(DEDUP):
    if fn.endswith(".jsonl"):
        with open(f"{DEDUP}/{fn}", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                seen.add(d["metadata"]["dedup_hash"])
print(f"现有池 hash 数: {len(seen)}")

DEFAULT_META = {"department": "unknown", "risk_level": "low", "red_flags": [],
                "evidence_required": False, "is_multiturn": False,
                "language": "zh", "source_quality": "high", "license": "internal"}

stats = collections.OrderedDict()
for tt, path in SRCS.items():
    n_in = n_out = n_drop_dup = n_drop_clean = 0
    out_path = f"{DEDUP}/gen_{tt}.jsonl"
    with open(path, encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as w:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            d = json.loads(line)
            msgs = d["messages"]
            non_sys = [m for m in msgs if m["role"] != "system"]
            ans = "".join(m["content"] for m in non_sys if m["role"] == "assistant")
            # 轻清洗
            if len(ans.strip()) < 12 or ASCII4.search(ans) or non_sys[-1]["role"] != "assistant":
                n_drop_clean += 1
                continue
            # 补全 metadata
            md = {**DEFAULT_META, **d.get("metadata", {})}
            md["dedup_hash"] = dhash(msgs)
            d["metadata"] = md
            d["task_type"] = tt
            d.setdefault("source", "gen_minimax_m3")
            if md["dedup_hash"] in seen:
                n_drop_dup += 1
                continue
            seen.add(md["dedup_hash"])
            w.write(json.dumps(d, ensure_ascii=False) + "\n")
            n_out += 1
    stats[tt] = {"in": n_in, "kept": n_out, "drop_dup": n_drop_dup, "drop_clean": n_drop_clean}
    print(f"[{tt}] in={n_in} kept={n_out} dup={n_drop_dup} clean={n_drop_clean}", flush=True)

with open(f"{REP}/merge_gen_v11_stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print(json.dumps(stats, ensure_ascii=False, indent=2))
