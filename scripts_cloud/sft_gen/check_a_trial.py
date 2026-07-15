#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import collections
import hashlib
import json
import re
from pathlib import Path

OUT = Path("/data/shenxin/rlhf_lab/data/sft_pipeline/generated/risk_redflag_safety_refusal.jsonl")
STATS = Path("/data/shenxin/rlhf_lab/data/sft_pipeline/generated/risk_redflag_safety_refusal.stats.json")
REPORT = Path("/data/shenxin/rlhf_lab/data/sft_pipeline/generated/risk_redflag_safety_refusal.trial_report.md")
LOG = Path("/data/shenxin/rlhf_lab/docs/GEN_BACKLOG_LOG.md")


def dedup_hash(messages):
    parts = []
    for msg in messages:
        if msg.get("role") == "system":
            continue
        content = re.sub(r"\s+", "", msg.get("content", "")).lower()
        parts.append(content)
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()


rows = [json.loads(line) for line in OUT.open(encoding="utf-8") if line.strip()]
stats = json.loads(STATS.read_text(encoding="utf-8"))

subtypes = collections.Counter((r["metadata"].get("generation_subtype") for r in rows))
voices = collections.Counter((r["metadata"].get("patient_voice") for r in rows))
departments = collections.Counter((r["metadata"].get("department") for r in rows))
risks = collections.Counter((r["metadata"].get("risk_level") for r in rows))
hashes = [r["metadata"].get("dedup_hash") for r in rows]
high_symbol = re.compile(r"[\U00010000-\U0010ffff]")
ascii_letter = re.compile(r"[A-Za-z]")

bad = collections.Counter()
for r in rows:
    messages = r.get("messages", [])
    if len(messages) < 3 or messages[0].get("role") != "system":
        bad["message_shape"] += 1
    if r.get("schema_version") != "1.0" or r.get("source") != "gen_minimax_m3":
        bad["top_level"] += 1
    if r.get("task_type") != "risk_redflag_safety_refusal":
        bad["task_type"] += 1
    if r["metadata"].get("dedup_hash") != dedup_hash(messages):
        bad["dedup_hash_mismatch"] += 1
    joined = json.dumps(r, ensure_ascii=False)
    if "<think>" in joined or "</think>" in joined:
        bad["think_tag"] += 1
    if "作为AI" in joined or "作为 AI" in joined:
        bad["ai_identity"] += 1
    dialogue_text = "\n".join(m.get("content", "") for m in messages if m.get("role") != "system")
    if high_symbol.search(dialogue_text):
        bad["emoji_or_high_symbol"] += 1
    if ascii_letter.search(dialogue_text):
        bad["ascii_letter_in_dialogue"] += 1
    assistants = [m.get("content", "") for m in messages if m.get("role") == "assistant"]
    if not assistants:
        bad["missing_assistant"] += 1
        continue
    final = assistants[-1]
    if len(final) > 600:
        bad["assistant_too_long"] += 1
    if r["metadata"].get("generation_subtype") == "A2_unsafe_request_refusal":
        if len(final) < 60:
            bad["assistant_too_short"] += 1
    elif len(final) < 80:
        bad["assistant_too_short"] += 1

print("CHECK_A_TRIAL")
print(f"jsonl_rows={len(rows)}")
print(f"first_id={rows[0]['id'] if rows else 'NA'}")
print(f"last_id={rows[-1]['id'] if rows else 'NA'}")
print(f"unique_dedup_hashes={len(set(hashes))}")
print(f"duplicate_hashes={len(hashes) - len(set(hashes))}")
print(f"subtypes={dict(subtypes)}")
print(f"risk_levels={dict(risks)}")
print(f"voices={dict(voices)}")
print(f"department_count={len(departments)}")
print(f"stats_count={stats.get('count')}")
print(f"stats_multiturn_count={stats.get('multiturn_count')}")
print(f"stats_discard_reasons={stats.get('discard_reasons')}")
print(f"bad={dict(bad)}")
print(f"report_exists={REPORT.exists()} report_bytes={REPORT.stat().st_size if REPORT.exists() else 0}")
print(f"log_exists={LOG.exists()} log_bytes={LOG.stat().st_size if LOG.exists() else 0}")
