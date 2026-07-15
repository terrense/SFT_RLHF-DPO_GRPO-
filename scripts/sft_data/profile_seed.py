#!/usr/bin/env python3
"""Stage 1: 种子集画像 — pre_consultation_multiturn.cleaned.jsonl
全量校验格式 + 输出分布报告(JSON + markdown)。只读数据,报告写到 reports/ 下。
"""
import json, sys, hashlib, collections, statistics, os

IN = "/data/shenxin/rlhf_lab/data/medicine_dataset/pre_consultation_multiturn.cleaned.jsonl"
OUT_DIR = "/data/shenxin/rlhf_lab/data/sft_pipeline/reports"
os.makedirs(OUT_DIR, exist_ok=True)

REQUIRED = ["id", "task_type", "source", "messages", "risk_level", "dedup_hash"]

def pct(vals, p):
    if not vals:
        return 0
    vals = sorted(vals)
    k = (len(vals) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    return vals[f] + (vals[c] - vals[f]) * (k - f)

n_total = 0
bad_json, missing_field, empty_output = [], [], []
ids, hashes = set(), set()
dup_id, dup_hash = 0, 0
turns_dist = collections.Counter()
risk_dist = collections.Counter()
dept_dist = collections.Counter()
scene_dist = collections.Counter()
triage_dist = collections.Counter()
style_dist = collections.Counter()
role_order_bad = []
token_est, char_total, char_last_assistant, user_turn_chars = [], [], [], []
first_role = collections.Counter()
sub_task_hint = collections.Counter()  # 依据结构推断可派生的 sub_task

with open(IN, encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        n_total += 1
        try:
            d = json.loads(line)
        except Exception:
            bad_json.append(i)
            continue
        miss = [k for k in REQUIRED if k not in d]
        if miss:
            missing_field.append((i, miss))
            continue
        if d["id"] in ids:
            dup_id += 1
        ids.add(d["id"])
        if d["dedup_hash"] in hashes:
            dup_hash += 1
        hashes.add(d["dedup_hash"])

        msgs = d["messages"]
        non_sys = [m for m in msgs if m["role"] != "system"]
        if not non_sys or non_sys[-1]["role"] != "assistant" or not non_sys[-1]["content"].strip():
            empty_output.append(i)
        # 相邻同角色 = 结构异常
        for a, b in zip(non_sys, non_sys[1:]):
            if a["role"] == b["role"]:
                role_order_bad.append(i)
                break
        first_role[non_sys[0]["role"] if non_sys else "none"] += 1

        meta = d.get("meta", {})
        turns_dist[d.get("num_turns", meta.get("num_turns", -1))] += 1
        risk_dist[d.get("risk_level", "?")] += 1
        dept_dist[meta.get("target_department", "unknown")] += 1
        scene_dist[meta.get("scene", "unknown")] += 1
        triage_dist[meta.get("triage_level", "?")] += 1
        style_dist[(meta.get("style") or "unknown")[:20]] += 1

        te = d.get("token_estimate")
        if isinstance(te, (int, float)):
            token_est.append(te)
        total_c = sum(len(m["content"]) for m in msgs)
        char_total.append(total_c)
        if non_sys and non_sys[-1]["role"] == "assistant":
            char_last_assistant.append(len(non_sys[-1]["content"]))
        for m in non_sys:
            if m["role"] == "user":
                user_turn_chars.append(len(m["content"]))

        # 可派生 sub_task:中间 assistant 提问轮→next_question;末轮含科室推荐→recommendation
        assistant_mid = [m for m in non_sys[:-1] if m["role"] == "assistant"]
        if assistant_mid:
            sub_task_hint["next_question_generation(可派生)"] += len(assistant_mid)
        if non_sys and "就诊" in non_sys[-1]["content"]:
            sub_task_hint["recommendation_generation(可派生)"] += 1

report = {
    "file": IN,
    "n_total": n_total,
    "n_bad_json": len(bad_json),
    "n_missing_required_field": len(missing_field),
    "n_empty_or_bad_final_output": len(empty_output),
    "n_role_order_anomaly": len(role_order_bad),
    "n_duplicate_id": dup_id,
    "n_duplicate_dedup_hash": dup_hash,
    "first_role": dict(first_role),
    "num_turns_dist": dict(sorted(turns_dist.items(), key=lambda x: (isinstance(x[0], str), x[0]))),
    "risk_level_dist": dict(risk_dist.most_common()),
    "scene_dist": dict(scene_dist.most_common()),
    "triage_level_dist": dict(sorted(triage_dist.items(), key=str)),
    "department_top20": dict(dept_dist.most_common(20)),
    "n_departments": len(dept_dist),
    "style_top10": dict(style_dist.most_common(10)),
    "token_estimate": {p: round(pct(token_est, q), 1) for p, q in
                       [("p50", 50), ("p90", 90), ("p99", 99), ("max", 100)]},
    "chars_total_per_sample": {p: round(pct(char_total, q), 1) for p, q in
                               [("p50", 50), ("p90", 90), ("p99", 99), ("max", 100)]},
    "chars_final_assistant": {p: round(pct(char_last_assistant, q), 1) for p, q in
                              [("p50", 50), ("p90", 90), ("p99", 99)]},
    "chars_user_turn": {p: round(pct(user_turn_chars, q), 1) for p, q in
                        [("p50", 50), ("p90", 90), ("p99", 99)]},
    "sub_task_derivable": dict(sub_task_hint),
    "anomaly_line_samples": {"bad_json": bad_json[:5], "empty_output": empty_output[:5],
                             "role_order": role_order_bad[:5]},
}

out = os.path.join(OUT_DIR, "stage1_seed_profile.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(json.dumps(report, ensure_ascii=False, indent=2))
print(f"\n[saved] {out}", file=sys.stderr)
