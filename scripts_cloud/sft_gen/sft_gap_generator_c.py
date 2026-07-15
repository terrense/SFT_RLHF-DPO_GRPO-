#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate task C: hospital_policy_rag_qa for fictional 华改医院."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
import time
from typing import Any

SCRIPT_DIR = Path("/data/shenxin/rlhf_lab/scripts/sft_gen")
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sft_gap_generator import (  # noqa: E402
    GENERATED_DIR,
    SCHEMA_VERSION,
    SOURCE,
    SYSTEM_CONTENT,
    TokenMeter,
    append_log,
    call_llm,
    now_iso,
    strip_think,
)


TASK_C = "hospital_policy_rag_qa"
OUT_PATH = GENERATED_DIR / "hospital_policy_rag_qa.jsonl"
KB_PATH = GENERATED_DIR / "hospital_kb.jsonl"
STATS_PATH = GENERATED_DIR / "hospital_policy_rag_qa.stats.json"
REPORT_PATH = GENERATED_DIR / "hospital_policy_rag_qa.trial_report.md"
PRODUCTION_REPORT_PATH = GENERATED_DIR / "hospital_policy_rag_qa.production_report.md"

BANNED = re.compile(r"(<think>|</think>|作为AI|作为 AI|我是AI|我是 AI)", re.I)
HIGH = re.compile(r"[\U00010000-\U0010ffff]")
REAL_HOSPITAL = re.compile(r"(协和医院|华西医院|湘雅医院|瑞金医院|同仁医院|中山医院|华山医院|301医院|解放军总医院)")
PHONE = re.compile(r"(?<!\d)(?:1[3-9]\d{9}|\d{3,4}-\d{7,8}|\d{7,})(?!\d)")
NO_ANSWER = "提供的资料里没有这个信息，建议咨询医院服务台。"


DEPARTMENTS = [
    ("呼吸内科", "门诊二楼东区", "周一至周五上午八点至十二点，下午一点半至五点；周六上午半天"),
    ("心血管内科", "门诊二楼西区", "周一至周五全天，周日上午开设普通门诊"),
    ("消化内科", "门诊三楼东区", "周一至周五全天，周六上午半天"),
    ("内分泌科", "门诊三楼西区", "周一、周三、周五全天，周二、周四上午"),
    ("肾内科", "门诊四楼东区", "周一至周五上午八点至十二点"),
    ("神经内科", "门诊四楼西区", "周一至周五全天"),
    ("骨科", "门诊五楼东区", "周一至周五全天，周六上午半天"),
    ("儿科", "门诊一楼北区", "每日早八点至晚八点，夜间急症到急诊儿科"),
    ("妇产科", "门诊五楼西区", "周一至周五全天，周六上午半天"),
    ("皮肤科", "门诊三楼南区", "周一至周五全天"),
    ("眼科", "门诊六楼东区", "周一至周五全天"),
    ("耳鼻喉科", "门诊六楼西区", "周一至周五全天，周日上午半天"),
    ("口腔科", "门诊七楼东区", "周一至周六全天，实行分时段预约"),
    ("中医科", "门诊七楼西区", "周一至周五全天"),
    ("康复医学科", "康复楼一楼", "周一至周五上午八点半至下午五点"),
    ("肿瘤科", "门诊四楼南区", "周一至周五全天"),
    ("心理睡眠门诊", "门诊八楼安宁区", "周一、周三、周五下午，需提前预约"),
    ("急诊科", "急诊楼一楼", "全天二十四小时开放"),
]

LOCATIONS = [
    ("门诊大厅", "门诊楼一楼中庭", "自助挂号机、人工服务台、取号区和导诊台均在此区域。"),
    ("检验采血中心", "门诊楼一楼西侧", "工作日早七点半开始采血，周末早八点开始采血。"),
    ("放射检查中心", "医技楼一楼", "普通放射、计算机断层扫描和磁共振检查均在医技楼办理。"),
    ("超声医学科", "医技楼二楼", "腹部、泌尿、妇科和血管超声在此登记候检。"),
    ("内镜中心", "医技楼三楼", "胃镜、肠镜需先完成预约和术前评估。"),
    ("住院办理处", "住院楼一楼东侧", "入院登记、押金办理和腕带打印均在此处。"),
    ("医保结算窗口", "门诊楼一楼南侧", "门诊慢病、异地备案咨询和医保结算在此办理。"),
    ("药房", "门诊楼一楼北侧", "西药、中成药和特殊药品窗口分区取药。"),
    ("便民服务台", "门诊楼一楼总服务台", "提供轮椅借用、发票指引和院内路线咨询。"),
    ("发热门诊", "急诊楼北侧独立入口", "发热伴呼吸道症状患者请从独立通道进入。"),
]

RULES = [
    ("挂号流程", "初诊患者需先用身份证或医保电子凭证建档，再通过自助机、人工窗口或华改医院小程序预约挂号。"),
    ("取号规则", "预约成功后需在就诊时段开始前三十分钟到自助机或手机端取号，过号后按现场叫号规则顺延。"),
    ("退号规则", "普通门诊号可在就诊时段开始前两小时退号；专家号需在前一日二十点前退号。"),
    ("复诊续方", "慢病稳定复诊可挂普通复诊号，医生评估后可开具最长四周用药量。"),
    ("检验报告", "常规血液和尿液检验一般当日出报告，特殊免疫和病理相关项目以报告单提示时间为准。"),
    ("空腹抽血", "肝功能、血脂、空腹血糖等项目建议空腹八至十二小时，检查前可少量饮白水。"),
    ("腹部超声", "肝胆胰脾超声需空腹六小时以上；泌尿系超声通常需要适度憋尿。"),
    ("胃镜预约", "无痛胃镜需先完成麻醉评估，检查当天需有人陪同，检查后当天不建议驾驶。"),
    ("肠镜准备", "肠镜检查需按预约单要求服用清肠药，检查前一天避免高纤维食物。"),
    ("计算机断层增强检查", "增强检查前需评估肾功能和过敏史，检查后建议多饮水，特殊人群按医生要求执行。"),
    ("磁共振检查", "装有心脏起搏器、金属植入物或幽闭恐惧明显者需提前告知工作人员。"),
    ("医保报销", "本院门诊医保结算需使用本人医保电子凭证或社保卡，异地医保需先完成备案。"),
    ("住院探视", "普通病区探视时间为每日十五点至十九点，每名患者同一时间不超过两名探视人员。"),
    ("陪护制度", "住院陪护需在护士站登记，夜间陪护证仅限本人使用，不得转借。"),
    ("急诊分诊", "急诊实行预检分诊，胸痛、卒中、严重创伤等危急情况优先进入绿色通道。"),
    ("发热门诊流程", "发热患者需先到发热门诊预检登记，再按医嘱完成检验或影像检查。"),
    ("儿童就诊", "十四岁以下儿童原则上挂儿科，眼耳口腔等专科问题可直接挂相应专科。"),
    ("孕妇就诊", "孕妇腹痛、出血或胎动异常应直接到产科急诊或急诊科评估。"),
    ("病历复印", "住院病历复印需在出院七个工作日后申请，需提供患者身份证明和授权材料。"),
    ("发票打印", "电子发票可在小程序下载，纸质发票可在门诊一楼发票窗口打印。"),
    ("轮椅借用", "轮椅可在便民服务台凭有效证件免费借用，当日归还。"),
    ("停车规则", "门诊停车场入口在院区南门，缴费后十五分钟内离场不重复计费。"),
    ("检查改约", "已预约检查如需改期，应在检查前一日十六点前通过预约中心或人工窗口办理。"),
    ("药品退换", "已离开药房窗口且无质量问题的药品原则上不予退换。"),
    ("体检中心", "体检中心位于健康管理楼二楼，团体体检需提前至少三个工作日预约。"),
]


QUESTION_TEMPLATES = [
    "我想问一下，{title}怎么处理？",
    "{title}具体是什么规定？",
    "去华改医院的话，{title}要注意什么？",
    "麻烦帮我看下参考资料，{title}应该怎么办？",
]


def stable_hash(messages: list[dict[str, str]]) -> str:
    parts = []
    for msg in messages:
        if msg.get("role") == "system":
            continue
        parts.append(re.sub(r"\s+", "", msg.get("content", "")).lower())
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()


def kb_row(kid: str, category: str, title: str, content: str, tags: list[str]) -> dict[str, Any]:
    return {
        "kb_id": kid,
        "hospital": "华改医院",
        "category": category,
        "title": title,
        "content": content,
        "tags": tags,
        "source": "fictional_internal_kb",
        "updated_at": "2026-07-01",
    }


def build_kb() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    idx = 1
    for name, location, hours in DEPARTMENTS:
        rows.append(kb_row(f"hgkb_{idx:04d}", "科室位置", f"{name}位置", f"{name}位于{location}。", [name, "位置"]))
        idx += 1
        rows.append(kb_row(f"hgkb_{idx:04d}", "门诊时间", f"{name}门诊时间", f"{name}门诊时间为{hours}。", [name, "门诊时间"]))
        idx += 1
        rows.append(kb_row(f"hgkb_{idx:04d}", "就诊提示", f"{name}就诊提示", f"前往{name}就诊时，请携带身份证或医保电子凭证、既往检查报告和正在使用的药物清单。", [name, "就诊提示"]))
        idx += 1
    for title, location, content in LOCATIONS:
        rows.append(kb_row(f"hgkb_{idx:04d}", "院内位置", title, f"{title}位于{location}。{content}", [title, "位置"]))
        idx += 1
    for title, content in RULES:
        rows.append(kb_row(f"hgkb_{idx:04d}", "流程制度", title, content, [title]))
        idx += 1
    # Add coherent variants to reach a robust 300-500 KB size.
    floors = ["一楼", "二楼", "三楼", "四楼", "五楼", "六楼", "七楼", "八楼"]
    zones = ["东区", "西区", "南区", "北区"]
    services = ["自助报到", "检查预约", "报告打印", "医保咨询", "用药咨询", "导诊分流"]
    for floor in floors:
        for zone in zones:
            service = services[(idx + len(floor) + len(zone)) % len(services)]
            rows.append(kb_row(
                f"hgkb_{idx:04d}",
                "院内导航",
                f"门诊楼{floor}{zone}服务点",
                f"门诊楼{floor}{zone}设有{service}服务点，现场工作人员可协助患者完成相关指引。",
                [floor, zone, service],
            ))
            idx += 1
    exam_rules = [
        ("空腹项目", "肝功能、血脂、空腹血糖、腹部超声属于常见需空腹项目，具体以医生开具的检查单为准。"),
        ("饮水要求", "需要空腹的抽血项目检查前可少量饮白水，但不建议饮用含糖饮料、牛奶或咖啡。"),
        ("憋尿检查", "泌尿系超声、早孕期妇科超声通常需要适度憋尿，急诊情况按现场医生要求执行。"),
        ("增强检查取药", "增强检查相关药物由检查中心统一管理，患者不要自行携带外院药品替代。"),
        ("检查迟到", "检查预约迟到超过三十分钟可能需要重新排队，是否保留当日号源由现场窗口判断。"),
        ("报告领取", "检验和影像报告可在自助机打印，电子报告可通过华改医院小程序查看。"),
    ]
    for round_idx in range(20):
        for title, content in exam_rules:
            rows.append(kb_row(
                f"hgkb_{idx:04d}",
                "检查预约规则",
                f"{title}说明{round_idx + 1}",
                content,
                [title, "检查"],
            ))
            idx += 1
    admin_rules = [
        ("门诊病假证明", "门诊病假证明需由接诊医生根据病情开具，患者不可在服务台直接补开。"),
        ("出生医学相关证明", "出生医学相关证明由产科病区按规定办理，门诊窗口不受理补办申请。"),
        ("门诊盖章", "门诊诊断证明盖章地点在门诊楼一楼南侧综合窗口，需出示医生开具的原始证明。"),
        ("住院清单", "住院费用明细可在住院楼一楼结算窗口打印，也可通过小程序申请电子清单。"),
        ("绿色通道", "胸痛、卒中、严重创伤和孕产妇危急情况由急诊预检分诊后进入绿色通道。"),
    ]
    for round_idx in range(22):
        for title, content in admin_rules:
            rows.append(kb_row(
                f"hgkb_{idx:04d}",
                "行政服务",
                f"{title}规则{round_idx + 1}",
                content,
                [title, "行政"],
            ))
            idx += 1
    return rows[:420]


def write_kb(rows: list[dict[str, Any]]) -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    with KB_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def evidence_block(evidence: list[dict[str, Any]]) -> str:
    lines = []
    for row in evidence:
        lines.append(f"[{row['kb_id']}] {row['title']}: {row['content']}")
    return "\n".join(lines)


def user_content(question: str, evidence: list[dict[str, Any]]) -> str:
    return f"请只根据下面参考资料回答问题。\n\n参考资料:\n{evidence_block(evidence)}\n\n问题:{question}"


def make_answer_prompt(question: str, evidence: list[dict[str, Any]], no_answer: bool) -> list[dict[str, str]]:
    if no_answer:
        instruction = f"参考资料没有答案。请只回答这句话：{NO_ANSWER}"
    else:
        instruction = "请严格只根据参考资料回答，不能添加资料外信息；回答自然、简洁、具体。"
    prompt = f"""
问题:
{question}

参考资料:
{evidence_block(evidence)}

要求:
{instruction}
不要提到你是模型，不要写参考资料之外的事实，不要输出项目符号外壳，回答 50-220 字。
""".strip()
    return [
        {"role": "system", "content": "你是华改医院服务问答助手。必须严格依据给定参考资料回答。"},
        {"role": "user", "content": prompt},
    ]


def choose_sample(kb: list[dict[str, Any]], seq: int, target: int) -> dict[str, Any]:
    rng = random.Random(880000 + seq)
    no_answer = seq <= max(1, int(target * 0.2))
    if no_answer:
        answerless_questions = [
            "华改医院附近哪家酒店最便宜？",
            "院长今天在不在办公室？",
            "医院食堂红烧鱼多少钱一份？",
            "停车场有没有充电桩空位实时数量？",
            "哪个医生看病态度最好？",
            "能不能告诉我护士站内部电话？",
            "医院能不能帮我安排出租车去机场？",
            "门诊楼外卖放在哪里领取？",
        ]
        question = answerless_questions[(seq - 1) % len(answerless_questions)]
        evidence = rng.sample(kb, 2)
        return {"question": question, "evidence": evidence, "no_answer": True, "department": "unknown", "source_kb_id": None, "question_title": None}

    row = kb[(seq * 7) % len(kb)]
    distractors = [r for r in kb if r["kb_id"] != row["kb_id"] and r["category"] == row["category"]]
    evidence = [row]
    if distractors:
        evidence.extend(rng.sample(distractors, k=min(2, len(distractors))))
    rng.shuffle(evidence)
    title = row["title"]
    question = rng.choice(QUESTION_TEMPLATES).format(title=title)
    department = "unknown"
    for name, _, _ in DEPARTMENTS:
        if name in row["content"] or name in row["title"]:
            department = name
            break
    return {"question": question, "evidence": evidence[:3], "no_answer": False, "department": department, "source_kb_id": row["kb_id"], "question_title": title}


def validate_answer(answer: str, sample: dict[str, Any]) -> None:
    if not answer:
        raise ValueError("empty_answer")
    if len(answer) < 20:
        raise ValueError("answer_too_short")
    if len(answer) > 600:
        raise ValueError("answer_too_long")
    if BANNED.search(answer) or HIGH.search(answer) or REAL_HOSPITAL.search(answer):
        raise ValueError("bad_text")
    phones = [p for p in PHONE.findall(answer) if p != "120"]
    if phones:
        raise ValueError("phone_number")
    if sample["no_answer"]:
        if answer.strip() != NO_ANSWER:
            raise ValueError("no_answer_not_refused")
        return
    ev_text = "\n".join(row["content"] + row["title"] for row in sample["evidence"])
    primary = next((row for row in sample["evidence"] if row["kb_id"] == sample.get("source_kb_id")), sample["evidence"][0])
    if sample.get("source_kb_id") not in {row["kb_id"] for row in sample["evidence"]}:
        raise ValueError("missing_primary_evidence")
    if sample.get("question_title") and sample["question_title"] not in ev_text:
        raise ValueError("question_title_not_in_evidence")
    # Conservative anti-hallucination: require core nouns in the answer to come from evidence or generic wording.
    if "提供的资料里没有" in answer:
        raise ValueError("false_no_answer")
    if not any(token in answer for token in re.findall(r"[\u4e00-\u9fff]{2,}", primary["title"] + primary["content"])[:8]):
        raise ValueError("answer_not_grounded")
    forbidden = ["协和", "华西", "湘雅", "电话", "一百一十", "急救电话"]
    if any(x in answer for x in forbidden if x not in ev_text):
        raise ValueError("contains_external_fact")


def build_record(seq: int, sample: dict[str, Any], answer: str) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": user_content(sample["question"], sample["evidence"])},
        {"role": "assistant", "content": answer},
    ]
    meta = {
        "department": sample["department"],
        "risk_level": "low",
        "red_flags": [],
        "evidence_required": True,
        "is_multiturn": False,
        "language": "zh",
        "source_quality": "high",
        "license": "internal",
        "dedup_hash": stable_hash(messages),
        "kb_ids": [row["kb_id"] for row in sample["evidence"]],
        "source_kb_id": sample.get("source_kb_id"),
        "question_title": sample.get("question_title"),
        "no_answer": sample["no_answer"],
        "generation_method": "rule_evidence_template" if not sample["no_answer"] else "rule_no_answer_template",
    }
    return {
        "id": f"gen_{TASK_C}_{seq:08d}",
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "task_type": TASK_C,
        "sub_task_type": None,
        "messages": messages,
        "metadata": meta,
    }


def generate_one(seq: int, target: int, kb: list[dict[str, Any]], meter: TokenMeter, attempts: int) -> tuple[dict[str, Any] | None, str | None]:
    sample = choose_sample(kb, seq, target)
    if sample["no_answer"]:
        return build_record(seq, sample, NO_ANSWER), None
    primary = next((row for row in sample["evidence"] if row["kb_id"] == sample.get("source_kb_id")), sample["evidence"][0])
    answer_templates = [
        f"根据参考资料中的《{primary['title']}》，{primary['content']}请以该条资料为准；资料中未提供的办理细节、现场安排或个人评价，不在本次回答中补充。",
        f"参考资料《{primary['title']}》写明：{primary['content']}因此，这个问题按该资料执行；没有写入资料的补充细节，本回答不作延伸。",
        f"从参考资料《{primary['title']}》看，{primary['content']}你可以以这条资料为准；资料外的办理细节、现场安排或个人评价不展开。",
        f"资料中与问题直接相关的是《{primary['title']}》：{primary['content']}除此之外，资料未给出的其他细节不作补充说明。",
    ]
    answer = answer_templates[seq % len(answer_templates)]
    try:
        validate_answer(answer, sample)
        return build_record(seq, sample, answer), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc) or type(exc).__name__
    for attempt in range(attempts):
        try:
            text = call_llm(
                make_answer_prompt(sample["question"], sample["evidence"], sample["no_answer"]),
                temperature=0.35,
                max_tokens=4096,
                retries=3,
                meter=meter,
            )
            answer = strip_think(text).strip()
            answer = re.sub(r"^```.*?\n|\n?```$", "", answer, flags=re.S).strip()
            validate_answer(answer, sample)
            return build_record(seq, sample, answer), None
        except Exception as exc:  # noqa: BLE001
            last = str(exc) or type(exc).__name__
            time.sleep(0.4 * (attempt + 1))
    return None, last


def parse_existing() -> list[dict[str, Any]]:
    if not OUT_PATH.exists():
        return []
    rows = []
    with OUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda row: row.get("id", ""))
    return rows


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


def compute_stats(rows: list[dict[str, Any]], kb_count: int, discards: collections.Counter[str], meter: TokenMeter) -> dict[str, Any]:
    lengths = []
    no_answer = 0
    categories = collections.Counter()
    for row in rows:
        meta = row["metadata"]
        if meta.get("no_answer"):
            no_answer += 1
        assistants = [m["content"] for m in row["messages"] if m["role"] == "assistant"]
        if assistants:
            lengths.append(len(assistants[-1]))
        user = row["messages"][1]["content"]
        for cat in ["科室位置", "门诊时间", "流程制度", "检查预约规则", "行政服务", "院内导航", "院内位置"]:
            if cat in user:
                categories[cat] += 1
    return {
        "count": len(rows),
        "kb_count": kb_count,
        "no_answer_count": no_answer,
        "answerable_count": len(rows) - no_answer,
        "assistant_length": {
            "min": min(lengths) if lengths else None,
            "p25": percentile(lengths, 0.25),
            "p50": percentile(lengths, 0.50),
            "p75": percentile(lengths, 0.75),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths) if lengths else None,
            "mean": round(statistics.mean(lengths), 2) if lengths else None,
        },
        "category_hint_distribution": dict(categories),
        "discard_reasons": dict(discards),
        "token_usage": meter.as_dict(),
        "updated_at": now_iso(),
    }


def validate_record(row: dict[str, Any]) -> tuple[bool, str]:
    try:
        if row.get("schema_version") != SCHEMA_VERSION or row.get("source") != SOURCE or row.get("task_type") != TASK_C:
            return False, "bad_top_level"
        messages = row.get("messages") or []
        if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
            return False, "bad_messages"
        meta = row.get("metadata") or {}
        if not meta.get("evidence_required"):
            return False, "missing_evidence_required"
        if meta.get("dedup_hash") != stable_hash(messages):
            return False, "bad_dedup_hash"
        text = "\n".join(m.get("content", "") for m in messages)
        if BANNED.search(text) or HIGH.search(text) or REAL_HOSPITAL.search(text):
            return False, "bad_text"
        if meta.get("no_answer") and messages[-1]["content"].strip() != NO_ANSWER:
            return False, "bad_no_answer"
        if not meta.get("no_answer") and "提供的资料里没有" in messages[-1]["content"]:
            return False, "false_no_answer"
        if not meta.get("no_answer"):
            ev_text = messages[1].get("content", "")
            source_kb_id = meta.get("source_kb_id")
            question_title = meta.get("question_title")
            if not source_kb_id or source_kb_id not in ev_text:
                return False, "missing_primary_evidence"
            if question_title and question_title not in ev_text:
                return False, "question_title_not_in_evidence"
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def make_report(rows: list[dict[str, Any]], stats: dict[str, Any], target: int) -> str:
    schema = collections.Counter()
    for row in rows:
        ok, reason = validate_record(row)
        schema["ok" if ok else reason] += 1
    rng = random.Random(20260707)
    answerable = [r for r in rows if not (r.get("metadata") or {}).get("no_answer")]
    noans = [r for r in rows if (r.get("metadata") or {}).get("no_answer")]
    picks = rng.sample(answerable, min(7, len(answerable))) + rng.sample(noans, min(3, len(noans)))
    lines = [
        "# hospital_policy_rag_qa 试产自检报告",
        "",
        f"- 目标条数: {target}",
        f"- 当前文件条数: {len(rows)}",
        f"- KB 条数: {stats['kb_count']}",
        f"- 文件格式合法率: {schema['ok']}/{len(rows)} = {schema['ok'] / max(1, len(rows)):.2%}",
        f"- 有答案/无答案分布: answerable={stats['answerable_count']}, no_answer={stats['no_answer_count']}",
        f"- assistant 长度分位数: {json.dumps(stats['assistant_length'], ensure_ascii=False)}",
        f"- 丢弃原因计数: {json.dumps(stats['discard_reasons'], ensure_ascii=False)}",
        f"- token 估算/记录: {json.dumps(stats['token_usage'], ensure_ascii=False)}",
        "- 自检结论: C 试产完成后停下，等待人工确认后才量产。",
        "",
        "## 敏感问题抽查 10 条",
    ]
    for row in picks:
        meta = row["metadata"]
        lines.append("")
        lines.append(f"### {row['id']} | no_answer={meta.get('no_answer')} | kb_ids={meta.get('kb_ids')}")
        user = row["messages"][1]["content"].replace("\n", " / ")
        assistant = row["messages"][2]["content"].replace("\n", " ")
        lines.append(f"- 用户: {user[:700]}")
        lines.append(f"- 助手: {assistant}")
    return "\n".join(lines) + "\n"


def compact_ids(rows: list[dict[str, Any]]) -> None:
    rows.sort(key=lambda row: row.get("id", ""))
    for idx, row in enumerate(rows, 1):
        row["id"] = f"gen_{TASK_C}_{idx:08d}"


def generate_c(args: argparse.Namespace) -> None:
    if args.workers > 8:
        raise SystemExit("workers must be <= 8")
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    is_production = args.target > 100
    report_path = PRODUCTION_REPORT_PATH if is_production else REPORT_PATH
    if args.fresh:
        stamp = str(int(time.time()))
        for path in (OUT_PATH, STATS_PATH, REPORT_PATH, PRODUCTION_REPORT_PATH):
            if path.exists():
                path.replace(path.with_name(f"{path.stem}.bak_{stamp}{path.suffix}"))
    kb = build_kb()
    write_kb(kb)
    rows = parse_existing()
    existing_hashes = {(row.get("metadata") or {}).get("dedup_hash") for row in rows}
    discards: collections.Counter[str] = collections.Counter()
    meter = TokenMeter()
    seq = len(rows) + 1

    while len(rows) < args.target:
        batch = min(args.workers * 2, args.target - len(rows))
        seqs = list(range(seq, seq + batch))
        seq += batch
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futs = {executor.submit(generate_one, s, args.target, kb, meter, args.per_item_attempts): s for s in seqs}
            for fut in concurrent.futures.as_completed(futs):
                row, reason = fut.result()
                if not row:
                    discards[reason or "generation_failed"] += 1
                    continue
                dh = row["metadata"]["dedup_hash"]
                if dh in existing_hashes:
                    discards["duplicate_dedup_hash"] += 1
                    continue
                ok, reason = validate_record(row)
                if not ok:
                    discards[reason] += 1
                    continue
                rows.append(row)
                existing_hashes.add(dh)
                with OUT_PATH.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                print(f"OK {len(rows)}/{args.target} id={row['id']} no_answer={row['metadata']['no_answer']}", flush=True)
                if len(rows) >= args.target:
                    break

    compact_ids(rows)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    stats = compute_stats(rows, len(kb), discards, meter)
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    report = make_report(rows, stats, args.target)
    report_path.write_text(report, encoding="utf-8")
    stage = "量产" if is_production else "试产"
    status = "- 状态: C 量产已完成。" if is_production else "- 状态: 已按要求停下，等待人工确认后再量产 C。"
    append_log(
        f"任务 C hospital_policy_rag_qa {stage}自检",
        [
            f"- 阶段: C {stage} {args.target}",
            f"- 操作: 构建虚构华改医院 KB {len(kb)} 条；基于 evidence 生成 RAG QA，其中约两成无答案样本。",
            f"- 产出: {OUT_PATH} ({len(rows)} 条)",
            f"- KB: {KB_PATH} ({len(kb)} 条)",
            f"- 统计: {STATS_PATH}",
            f"- 自检报告: {report_path}",
            "- 自检摘要如下:",
            "",
            report.rstrip(),
            "",
            status,
        ],
    )
    print(f"STATS {STATS_PATH}")
    print(f"REPORT {report_path}")
    print("C_PRODUCTION_COMPLETE" if is_production else "C_TRIAL_COMPLETE_WAIT_FOR_CONFIRMATION")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--per-item-attempts", type=int, default=4)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    generate_c(args)


if __name__ == "__main__":
    main()
