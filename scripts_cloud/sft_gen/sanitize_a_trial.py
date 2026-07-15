#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean task A trial text to satisfy the all-Chinese content rule."""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path
import re
import statistics
import sys

SCRIPT_DIR = Path("/data/shenxin/rlhf_lab/scripts/sft_gen")
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sft_gap_generator import (  # noqa: E402
    GENERATED_DIR,
    LOG_PATH,
    TASK_A_OUT,
    TASK_A_STATS,
    append_log,
    dedup_hash,
    make_trial_report,
    now_iso,
)


LETTER_REPL = [
    (re.compile(r"PET\s*-\s*CT", re.I), "正电子发射断层显像"),
    (re.compile(r"\bCT\b", re.I), "影像检查"),
    (re.compile(r"\bCRP\b", re.I), "炎症指标"),
    (re.compile(r"\bMRI\b", re.I), "磁共振检查"),
]
HIGH_SYMBOL = re.compile(r"[\U00010000-\U0010ffff]")
VARIATION = re.compile(r"[\ufe0e\ufe0f]")
ASCII_LETTER = re.compile(r"[A-Za-z]+")


def clean_text(text: str) -> tuple[str, int]:
    before = text
    for pat, repl in LETTER_REPL:
        text = pat.sub(repl, text)
    text = HIGH_SYMBOL.sub("", text)
    text = VARIATION.sub("", text)
    text = ASCII_LETTER.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([，。；：？！、])", r"\1", text)
    return text.strip(), int(text != before)


def percentile(values: list[int], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    k = (len(values) - 1) * pct
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(values[lo])
    return float(values[lo] * (hi - k) + values[hi] * (k - lo))


def recompute_stats(rows, old_stats):
    assistant_lengths = []
    subtypes = collections.Counter()
    voices = collections.Counter()
    departments = collections.Counter()
    risk_levels = collections.Counter()
    red_flags = collections.Counter()
    multiturn = 0
    for row in rows:
        meta = row["metadata"]
        subtypes[meta.get("generation_subtype", "unknown")] += 1
        voices[meta.get("patient_voice", "unknown")] += 1
        departments[meta.get("department", "unknown")] += 1
        risk_levels[meta.get("risk_level", "unknown")] += 1
        if meta.get("is_multiturn"):
            multiturn += 1
        for flag in meta.get("red_flags") or []:
            red_flags[flag] += 1
        assistants = [m["content"] for m in row["messages"] if m["role"] == "assistant"]
        if assistants:
            assistant_lengths.append(len(assistants[-1]))
    return {
        "count": len(rows),
        "assistant_length": {
            "min": min(assistant_lengths) if assistant_lengths else None,
            "p25": percentile(assistant_lengths, 0.25),
            "p50": percentile(assistant_lengths, 0.50),
            "p75": percentile(assistant_lengths, 0.75),
            "p95": percentile(assistant_lengths, 0.95),
            "max": max(assistant_lengths) if assistant_lengths else None,
            "mean": round(statistics.mean(assistant_lengths), 2) if assistant_lengths else None,
        },
        "subtype_distribution": dict(subtypes),
        "risk_level_distribution": dict(risk_levels),
        "department_distribution": dict(departments),
        "patient_voice_distribution": dict(voices),
        "red_flag_distribution": dict(red_flags),
        "multiturn_count": multiturn,
        "discard_reasons": old_stats.get("discard_reasons", {}),
        "token_usage": old_stats.get("token_usage", {}),
        "updated_at": now_iso(),
        "postprocess": {
            "all_chinese_sanitize": True,
        },
    }


def main() -> None:
    rows = [json.loads(line) for line in TASK_A_OUT.open(encoding="utf-8") if line.strip()]
    old_stats = json.loads(TASK_A_STATS.read_text(encoding="utf-8"))
    changed_rows = 0
    changed_messages = 0
    for row in rows:
        row_changed = False
        for msg in row.get("messages", []):
            if msg.get("role") == "system":
                continue
            cleaned, changed = clean_text(msg.get("content", ""))
            if changed:
                msg["content"] = cleaned
                changed_messages += 1
                row_changed = True
        if row_changed:
            changed_rows += 1
            row["metadata"]["dedup_hash"] = dedup_hash(row["messages"])

    tmp = TASK_A_OUT.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(TASK_A_OUT)

    stats = recompute_stats(rows, old_stats)
    TASK_A_STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    report = make_trial_report(rows, stats, len(rows))
    report_path = GENERATED_DIR / "risk_redflag_safety_refusal.trial_report.md"
    report_path.write_text(report, encoding="utf-8")
    append_log(
        "任务 A 试产后清洗与复检",
        [
            "- 阶段: A 试产清洗",
            f"- 操作: 移除/替换 user 与 assistant 文本中的 emoji 和英文缩写；重算 dedup_hash、stats 和 trial_report。",
            f"- 变更: {changed_rows} 条样本、{changed_messages} 个消息文本发生清洗。",
            f"- 产出: {TASK_A_OUT} ({len(rows)} 条)",
            f"- 统计: {TASK_A_STATS}",
            f"- 自检报告: {report_path}",
            "- 自检摘要如下:",
            "",
            report.rstrip(),
            "",
            "- 状态: 清洗后仍停下，等待人工确认后再量产。",
        ],
    )
    print(f"SANITIZED rows_changed={changed_rows} messages_changed={changed_messages}")
    print(f"ROWS {len(rows)}")
    print(f"STATS {TASK_A_STATS}")
    print(f"REPORT {report_path}")


if __name__ == "__main__":
    main()
