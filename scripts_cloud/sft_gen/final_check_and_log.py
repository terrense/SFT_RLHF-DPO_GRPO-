#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import collections
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics


BASE = Path("/data/shenxin/rlhf_lab/data/sft_pipeline/generated")
LOG_PATH = Path("/data/shenxin/rlhf_lab/docs/GEN_BACKLOG_LOG.md")


def stable_hash(messages: list[dict[str, str]]) -> str:
    parts = []
    for msg in messages:
        if msg.get("role") != "system":
            parts.append(re.sub(r"\s+", "", msg.get("content", "")).lower())
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()


def pct(values: list[int], q: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(len(values) * q) - 1))
    return values[idx]


def check(stem: str, expected: int, task: str) -> dict[str, object]:
    path = BASE / f"{stem}.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    bad: collections.Counter[str] = collections.Counter()
    hashes: list[str | None] = []
    ids: list[str | None] = []
    lengths: list[int] = []
    subtypes: collections.Counter[str] = collections.Counter()
    risks: collections.Counter[str] = collections.Counter()
    departments: collections.Counter[str] = collections.Counter()

    for row in rows:
        ids.append(row.get("id"))
        if (
            row.get("schema_version") != "1.0"
            or row.get("source") != "gen_minimax_m3"
            or row.get("task_type") != task
        ):
            bad["top_level"] += 1
        messages = row.get("messages") or []
        roles = [m.get("role") for m in messages]
        valid_roles = (
            len(roles) >= 3
            and roles[0] == "system"
            and roles[-1] == "assistant"
            and all(role in {"user", "assistant"} for role in roles[1:])
            and all(roles[i] != roles[i - 1] for i in range(2, len(roles)))
            and roles[1] == "user"
        )
        if not valid_roles:
            bad["messages"] += 1
            continue
        text = "\n".join(m.get("content", "") for m in messages)
        if any(marker in text for marker in ["<think>", "</think>", "作为AI", "作为 AI"]):
            bad["bad_text"] += 1
        metadata = row.get("metadata") or {}
        if metadata.get("dedup_hash") != stable_hash(messages):
            bad["dedup_hash"] += 1
        hashes.append(metadata.get("dedup_hash"))
        answer = messages[-1].get("content", "")
        lengths.append(len(answer))
        subtype_value = row.get("sub_task_type") or metadata.get("sub_task_type") or metadata.get("subtype")
        if subtype_value:
            subtypes[f"subtype={subtype_value}"] += 1
        if metadata.get("report_type"):
            subtypes[f"report_type={metadata.get('report_type')}"] += 1
        if "no_answer" in metadata:
            subtypes[f"no_answer={metadata.get('no_answer')}"] += 1
        risks[str(metadata.get("risk_level"))] += 1
        departments[str(metadata.get("department"))] += 1

    stats_path = BASE / f"{stem}.stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    ok = (
        len(rows) == expected
        and not bad
        and len(set(ids)) == len(ids)
        and len(set(hashes)) == len(hashes)
        and stats.get("count") == expected
    )
    return {
        "stem": stem,
        "ok": ok,
        "rows": len(rows),
        "expected": expected,
        "unique_ids": len(set(ids)),
        "unique_hashes": len(set(hashes)),
        "duplicate_hashes": len(hashes) - len(set(hashes)),
        "bad": dict(bad),
        "assistant_length": {
            "min": min(lengths) if lengths else None,
            "p50": statistics.median(lengths) if lengths else None,
            "p95": pct(lengths, 0.95),
            "max": max(lengths) if lengths else None,
        },
        "risk_dist": dict(risks),
        "subtype_or_type_top": dict(subtypes.most_common(20)),
        "department_top": dict(departments.most_common(12)),
        "stats_updated_at": stats.get("updated_at"),
        "token_total": (stats.get("token_usage") or {}).get("total_tokens"),
        "discard_reasons": stats.get("discard_reasons"),
    }


def main() -> None:
    checks = [
        check("risk_redflag_safety_refusal", 10000, "risk_redflag_safety_refusal"),
        check("test_report_explanation", 8000, "test_report_explanation"),
        check("hospital_policy_rag_qa", 5000, "hospital_policy_rag_qa"),
    ]

    for stem in ("risk_redflag_safety_refusal", "test_report_explanation"):
        src = BASE / f"{stem}.trial_report.md"
        dst = BASE / f"{stem}.production_report.md"
        if src.exists():
            shutil.copyfile(src, dst)

    summary = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "total_rows": sum(int(item["rows"]) for item in checks),
    }
    (BASE / "sft_gap_generation_final_check.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "",
        f"## {summary['checked_at']} A/B/C 生成任务最终完成校验",
    ]
    for item in checks:
        lines.append(
            f"- {item['stem']}: {item['rows']}/{item['expected']} 条；"
            f"唯一 id={item['unique_ids']}，唯一 hash={item['unique_hashes']}，"
            f"重复 hash={item['duplicate_hashes']}，bad={item['bad']}，"
            f"最终 stats 时间={item['stats_updated_at']}，通过={item['ok']}。"
        )
    lines.extend(
        [
            "- A/B 历史脚本输出名仍为 trial_report.md；已在 generated/ 内另存 production_report.md 便于汇总。",
            f"- 总量: {summary['total_rows']} 条；所有产出均在 /data/shenxin/rlhf_lab/data/sft_pipeline/generated/。",
            f"- 最终校验摘要: {BASE / 'sft_gap_generation_final_check.json'}",
            "",
        ]
    )
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
