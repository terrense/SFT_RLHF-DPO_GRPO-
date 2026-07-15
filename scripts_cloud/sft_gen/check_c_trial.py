#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import collections
import hashlib
import json
import re
from pathlib import Path

OUT = Path("/data/shenxin/rlhf_lab/data/sft_pipeline/generated/hospital_policy_rag_qa.jsonl")
KB = Path("/data/shenxin/rlhf_lab/data/sft_pipeline/generated/hospital_kb.jsonl")
STATS = Path("/data/shenxin/rlhf_lab/data/sft_pipeline/generated/hospital_policy_rag_qa.stats.json")
REPORT = Path("/data/shenxin/rlhf_lab/data/sft_pipeline/generated/hospital_policy_rag_qa.trial_report.md")
NO_ANSWER = "提供的资料里没有这个信息，建议咨询医院服务台。"
BANNED = re.compile(r"(<think>|</think>|作为AI|作为 AI|我是AI|我是 AI)", re.I)
HIGH = re.compile(r"[\U00010000-\U0010ffff]")
REAL_HOSP = re.compile(r"(协和医院|华西医院|湘雅医院|瑞金医院|同仁医院|中山医院|华山医院|301医院|解放军总医院)")


def dhash(messages):
    parts = []
    for msg in messages:
        if msg.get("role") == "system":
            continue
        parts.append(re.sub(r"\s+", "", msg.get("content", "")).lower())
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()


rows = [json.loads(line) for line in OUT.open(encoding="utf-8") if line.strip()]
kb = [json.loads(line) for line in KB.open(encoding="utf-8") if line.strip()]
stats = json.loads(STATS.read_text(encoding="utf-8")) if STATS.exists() else {}
bad = collections.Counter()
ids = [r.get("id") for r in rows]
hashes = [(r.get("metadata") or {}).get("dedup_hash") for r in rows]
no_answer = 0
answerable = 0
kb_ids = {r["kb_id"] for r in kb}
kb_by_id = {r["kb_id"]: r for r in kb}
kb_ref_counter = collections.Counter()

for row in rows:
    messages = row.get("messages") or []
    meta = row.get("metadata") or {}
    if row.get("schema_version") != "1.0" or row.get("source") != "gen_minimax_m3":
        bad["top_level"] += 1
    if row.get("task_type") != "hospital_policy_rag_qa":
        bad["task_type"] += 1
    if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
        bad["message_shape"] += 1
    if meta.get("dedup_hash") != dhash(messages):
        bad["dedup_hash_mismatch"] += 1
    if meta.get("evidence_required") is not True:
        bad["evidence_required_not_true"] += 1
    user = messages[1].get("content", "") if len(messages) > 1 else ""
    assistant = messages[2].get("content", "") if len(messages) > 2 else ""
    text = user + "\n" + assistant
    if "参考资料:" not in user:
        bad["missing_evidence_block"] += 1
    if BANNED.search(text) or HIGH.search(text) or REAL_HOSP.search(text):
        bad["bad_text"] += 1
    refs = meta.get("kb_ids") or []
    if not refs:
        bad["missing_kb_ids"] += 1
    for kid in refs:
        kb_ref_counter[kid] += 1
        if kid not in kb_ids:
            bad["unknown_kb_id"] += 1
    if meta.get("no_answer"):
        no_answer += 1
        if assistant.strip() != NO_ANSWER:
            bad["bad_no_answer_response"] += 1
    else:
        answerable += 1
        if "提供的资料里没有" in assistant:
            bad["false_no_answer"] += 1
        source_kb_id = meta.get("source_kb_id")
        question_title = meta.get("question_title")
        if not source_kb_id or source_kb_id not in user:
            bad["missing_primary_evidence"] += 1
        if question_title and question_title not in user:
            bad["question_title_not_in_evidence"] += 1
        if source_kb_id in kb_by_id and kb_by_id[source_kb_id]["content"] not in assistant:
            bad["primary_content_not_in_answer"] += 1
        if len(assistant) < 20 or len(assistant) > 600:
            bad["answer_length"] += 1

if len(kb) < 300 or len(kb) > 500:
    bad["kb_count_out_of_range"] += 1
if len(rows) == 100 and no_answer != 20:
    bad["no_answer_not_20_percent"] += 1

print("CHECK_C_TRIAL")
print(f"jsonl_rows={len(rows)}")
print(f"kb_rows={len(kb)}")
print(f"first_id={ids[0] if ids else 'NA'}")
print(f"last_id={ids[-1] if ids else 'NA'}")
print(f"unique_ids={len(set(ids))}")
print(f"unique_dedup_hashes={len(set(hashes))}")
print(f"duplicate_hashes={len(hashes) - len(set(hashes))}")
print(f"answerable={answerable}")
print(f"no_answer={no_answer}")
print(f"referenced_kb_count={len(kb_ref_counter)}")
print(f"stats_count={stats.get('count')}")
print(f"stats_no_answer_count={stats.get('no_answer_count')}")
print(f"stats_discard_reasons={stats.get('discard_reasons')}")
print(f"bad={dict(bad)}")
print(f"report_exists={REPORT.exists()} report_bytes={REPORT.stat().st_size if REPORT.exists() else 0}")
