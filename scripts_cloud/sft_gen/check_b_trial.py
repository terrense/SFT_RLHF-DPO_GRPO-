#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import collections
import hashlib
import json
import re
from pathlib import Path

OUT = Path("/data/shenxin/rlhf_lab/data/sft_pipeline/generated/test_report_explanation.jsonl")
STATS = Path("/data/shenxin/rlhf_lab/data/sft_pipeline/generated/test_report_explanation.stats.json")
REPORT = Path("/data/shenxin/rlhf_lab/data/sft_pipeline/generated/test_report_explanation.trial_report.md")
LOG = Path("/data/shenxin/rlhf_lab/docs/GEN_BACKLOG_LOG.md")

ASCII = re.compile(r"[A-Za-z]")
HIGH = re.compile(r"[\U00010000-\U0010ffff]")
BANNED = re.compile(r"(<think>|</think>|作为AI|作为 AI|我是AI|我是 AI)", re.I)
REAL_HOSP = re.compile(r"(协和医院|华西医院|湘雅医院|瑞金医院|同仁医院|中山医院|华山医院|301医院|解放军总医院)")


def dhash(messages):
    parts = []
    for msg in messages:
        if msg.get("role") == "system":
            continue
        parts.append(re.sub(r"\s+", "", msg.get("content", "")).lower())
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()


rows = [json.loads(line) for line in OUT.open(encoding="utf-8") if line.strip()]
stats = json.loads(STATS.read_text(encoding="utf-8")) if STATS.exists() else {}

bad = collections.Counter()
ids = [r.get("id") for r in rows]
hashes = [(r.get("metadata") or {}).get("dedup_hash") for r in rows]
types = collections.Counter()
risks = collections.Counter()
departments = collections.Counter()

for row in rows:
    meta = row.get("metadata") or {}
    messages = row.get("messages") or []
    types[meta.get("report_type", "unknown")] += 1
    risks[meta.get("risk_level", "unknown")] += 1
    departments[meta.get("department", "unknown")] += 1
    if row.get("schema_version") != "1.0" or row.get("source") != "gen_minimax_m3":
        bad["top_level"] += 1
    if row.get("task_type") != "test_report_explanation":
        bad["task_type"] += 1
    if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
        bad["message_shape"] += 1
    if meta.get("dedup_hash") != dhash(messages):
        bad["dedup_hash_mismatch"] += 1
    dialogue = "\n".join(m.get("content", "") for m in messages if m.get("role") != "system")
    assistant = messages[-1].get("content", "") if messages else ""
    if BANNED.search(dialogue):
        bad["banned_text"] += 1
    if ASCII.search(dialogue):
        bad["ascii_letter_in_dialogue"] += 1
    if HIGH.search(dialogue):
        bad["emoji_or_high_symbol"] += 1
    if REAL_HOSP.search(dialogue):
        bad["real_hospital"] += 1
    if "↑" in dialogue or "↓" in dialogue:
        bad["arrow_marker_leftover"] += 1
    if re.search(r"中性粒细胞比例:\s*[0-9]%", dialogue):
        bad["suspicious_single_digit_neutrophil_percent"] += 1
    if re.search(r"血红蛋白:\s*[0-9]\s*克每升", dialogue):
        bad["suspicious_single_digit_hemoglobin"] += 1
    if len(assistant) < 80 or len(assistant) > 600:
        bad["assistant_length"] += 1
    if not any(w in assistant for w in ("可能", "提示", "建议结合", "倾向", "方向")):
        bad["missing_uncertainty"] += 1
    if any(w in assistant for w in ("确诊为", "肯定是", "一定是", "就是癌")):
        bad["diagnostic_overclaim"] += 1
    if not any(w in assistant for w in ("复诊", "就诊", "门诊", "医生")):
        bad["missing_followup"] += 1
    if not any(w in assistant for w in ("不能替代医生面诊", "不替代医生面诊", "不能代替医生面诊", "不能代替面诊")):
        bad["missing_disclaimer"] += 1

print("CHECK_B_TRIAL")
print(f"jsonl_rows={len(rows)}")
print(f"first_id={ids[0] if ids else 'NA'}")
print(f"last_id={ids[-1] if ids else 'NA'}")
print(f"unique_ids={len(set(ids))}")
print(f"unique_dedup_hashes={len(set(hashes))}")
print(f"duplicate_hashes={len(hashes) - len(set(hashes))}")
print(f"report_types={dict(types)}")
expected_each = len(rows) // 10 if rows else 0
if len(rows) == 100 and any(types.get(name, 0) != 10 for name in [
    "血常规", "肝功能", "肾功能", "甲状腺功能", "血脂", "尿常规", "腹部超声", "胸部计算机断层扫描", "心电图", "肿瘤标志物"
]):
    bad["report_type_not_10_each"] += 1
print(f"risk_levels={dict(risks)}")
print(f"department_count={len(departments)}")
print(f"stats_count={stats.get('count')}")
print(f"stats_discard_reasons={stats.get('discard_reasons')}")
print(f"bad={dict(bad)}")
print(f"report_exists={REPORT.exists()} report_bytes={REPORT.stat().st_size if REPORT.exists() else 0}")
print(f"log_exists={LOG.exists()} log_bytes={LOG.stat().st_size if LOG.exists() else 0}")
