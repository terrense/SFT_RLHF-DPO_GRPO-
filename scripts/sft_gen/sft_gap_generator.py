#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Medical SFT gap-data generator.

Remote write policy for this project:
  - scripts: /data/shenxin/rlhf_lab/scripts/sft_gen/
  - outputs: /data/shenxin/rlhf_lab/data/sft_pipeline/generated/
  - log:     /data/shenxin/rlhf_lab/docs/GEN_BACKLOG_LOG.md

This script intentionally reads the MiniMax key only from the shell
environment and never prints or persists it.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
import threading
import time
from typing import Any

import requests


BASE_DIR = Path("/data/shenxin/rlhf_lab")
SCRIPT_DIR = BASE_DIR / "scripts" / "sft_gen"
GENERATED_DIR = BASE_DIR / "data" / "sft_pipeline" / "generated"
LOG_PATH = BASE_DIR / "docs" / "GEN_BACKLOG_LOG.md"
TASK_A_OUT = GENERATED_DIR / "risk_redflag_safety_refusal.jsonl"
TASK_A_STATS = GENERATED_DIR / "risk_redflag_safety_refusal.stats.json"

SOURCE = "gen_minimax_m3"
SCHEMA_VERSION = "1.0"
TASK_A = "risk_redflag_safety_refusal"
MODEL_DEFAULT = "MiniMax-M3"

SYSTEM_CONTENT = (
    "你是一名中文医疗助手。回答应谨慎、清晰、以患者安全为先；"
    "不能替代医生面诊，遇到急症、自伤风险或明显危险用药请求时，"
    "要建议立即寻求线下急救或专业帮助。"
)

GEN_SYSTEM = (
    "你是医疗安全 SFT 数据生成专家。请严格按用户给定的场景生成中文医疗助手对话。"
    "输出必须是可解析 JSON，不要 Markdown，不要解释，不要出现 <think>。"
)

PATIENT_VOICES = [
    ("焦虑", "语气急促、反复确认危险性，短句多。"),
    ("淡定", "描述平铺直叙，像在普通咨询，但症状本身很危险。"),
    ("老人", "用词朴素，可能说不清医学名词，带一点生活化表达。"),
    ("家属代述", "替家人描述病情，关注要不要马上送医。"),
    ("网络用语", "口吻像线上发帖或聊天，夹杂“有点慌”“顶不住”等表达。"),
    ("表达混乱", "信息顺序有些乱，症状、时间、担心交织在一起。"),
    ("外地务工", "担心请假、费用或路程，想先问清楚严重不严重。"),
    ("学生", "年轻人口吻，担心是否会影响上课或考试。"),
]

RED_FLAG_SCENARIOS = [
    {
        "key": "剧烈胸痛伴大汗",
        "department": "急诊科/心内科",
        "red_flags": ["剧烈胸痛", "大汗", "胸闷"],
        "seed": "胸口像被压住，持续二十多分钟，冒冷汗，还有点恶心。",
        "focus": "可能涉及急性冠脉综合征等危急情况",
    },
    {
        "key": "呼吸困难伴口唇发紫",
        "department": "急诊科/呼吸内科",
        "red_flags": ["呼吸困难", "口唇发紫"],
        "seed": "突然喘不上气，说话断断续续，嘴唇发紫。",
        "focus": "可能存在严重缺氧",
    },
    {
        "key": "意识障碍",
        "department": "急诊科/神经内科",
        "red_flags": ["意识障碍", "嗜睡或叫不醒"],
        "seed": "人突然很迷糊，叫他反应慢，刚才还差点叫不醒。",
        "focus": "意识改变本身就是急症信号",
    },
    {
        "key": "呕血或黑便",
        "department": "急诊科/消化内科",
        "red_flags": ["呕血", "黑便"],
        "seed": "吐出来像咖啡色的东西，今天大便也发黑发亮。",
        "focus": "可能提示消化道出血",
    },
    {
        "key": "突发偏瘫口齿不清",
        "department": "急诊科/神经内科",
        "red_flags": ["突发偏瘫", "口齿不清"],
        "seed": "一边胳膊腿突然没劲，讲话含糊，嘴角有点歪。",
        "focus": "可能是卒中相关表现，时间窗很关键",
    },
    {
        "key": "孕妇剧烈腹痛或胎动消失",
        "department": "急诊科/产科",
        "red_flags": ["孕妇剧烈腹痛", "胎动消失"],
        "seed": "怀孕三十二周，肚子疼得厉害，今天几乎感觉不到胎动。",
        "focus": "孕晚期腹痛和胎动异常需要立即产科急诊评估",
    },
    {
        "key": "明确自杀意念",
        "department": "急诊科/精神心理科",
        "red_flags": ["自杀意念", "自伤风险"],
        "seed": "一直想结束自己，已经想好今天晚上怎么做。",
        "focus": "存在即时自伤风险",
    },
    {
        "key": "中毒或误服大量药物",
        "department": "急诊科/中毒救治",
        "red_flags": ["中毒", "大量误服药物"],
        "seed": "误把一整把药吞了，现在头晕、恶心、站不稳。",
        "focus": "需要尽快毒物评估和监护",
    },
    {
        "key": "过敏性休克表现",
        "department": "急诊科/变态反应科",
        "red_flags": ["喉头紧缩", "全身风团", "血压下降"],
        "seed": "吃完东西全身起大片疙瘩，喉咙发紧，眼前发黑。",
        "focus": "可能出现过敏性休克",
    },
    {
        "key": "大出血",
        "department": "急诊科/普外科",
        "red_flags": ["大出血", "止不住的出血"],
        "seed": "切伤后血一直喷，按了十分钟还止不住，脸色越来越白。",
        "focus": "持续大量出血会导致休克",
    },
    {
        "key": "婴幼儿高热惊厥",
        "department": "急诊科/儿科",
        "red_flags": ["婴幼儿高热", "抽搐"],
        "seed": "一岁多孩子烧到四十度，刚才眼睛上翻、四肢抽动。",
        "focus": "婴幼儿高热伴抽搐需要急诊处理",
    },
    {
        "key": "突发爆炸样头痛",
        "department": "急诊科/神经内科",
        "red_flags": ["突发剧烈头痛", "颈部僵硬"],
        "seed": "头突然像炸开一样疼，脖子硬，还想吐。",
        "focus": "突发最严重头痛需排查颅内急症",
    },
    {
        "key": "严重腹痛伴板状腹",
        "department": "急诊科/普外科",
        "red_flags": ["剧烈腹痛", "腹部僵硬"],
        "seed": "肚子疼得不敢动，摸起来硬邦邦，出冷汗。",
        "focus": "可能存在腹部急症",
    },
    {
        "key": "高热伴低血压和寒战",
        "department": "急诊科/感染科",
        "red_flags": ["高热寒战", "血压偏低", "意识变差"],
        "seed": "烧到三十九度多，发抖，量血压很低，人也越来越没精神。",
        "focus": "可能存在严重感染或休克风险",
    },
    {
        "key": "头部外伤后反复呕吐",
        "department": "急诊科/神经外科",
        "red_flags": ["头部外伤", "反复呕吐", "嗜睡"],
        "seed": "摔到头后吐了好几次，现在很困，问话也慢。",
        "focus": "头外伤后神经症状需要急诊评估",
    },
    {
        "key": "咯大量鲜血",
        "department": "急诊科/呼吸内科",
        "red_flags": ["大量咯血", "呼吸困难"],
        "seed": "咳出来一大口一大口鲜血，胸闷，越咳越慌。",
        "focus": "大量咯血可能阻塞气道或失血",
    },
    {
        "key": "糖尿病酮症风险",
        "department": "急诊科/内分泌科",
        "red_flags": ["糖尿病", "呕吐", "嗜睡", "呼吸深快"],
        "seed": "糖尿病患者这两天吐得厉害，口很干，呼吸又深又快，人发蔫。",
        "focus": "可能存在代谢急症",
    },
    {
        "key": "眼外伤视力骤降",
        "department": "急诊科/眼科",
        "red_flags": ["眼外伤", "视力骤降"],
        "seed": "眼睛被弹到后突然看不清，还有明显疼痛和流泪。",
        "focus": "眼外伤伴视力下降需要急诊眼科处理",
    },
]

UNSAFE_SCENARIOS = [
    {
        "key": "安眠药囤积",
        "department": "精神心理科",
        "risk_level": "medium",
        "red_flags": ["处方药滥用", "镇静催眠药风险"],
        "unsafe_goal": "用户想让助手提供大量安眠药获取或囤积办法。",
        "safe_path": "解释不能帮助囤积或滥用处方药，建议精神心理科或睡眠门诊评估失眠原因。",
    },
    {
        "key": "自行加大阿片类止痛药剂量",
        "department": "疼痛科/肿瘤科",
        "risk_level": "high",
        "red_flags": ["阿片类药物过量风险", "呼吸抑制风险"],
        "unsafe_goal": "用户要求直接告诉他把强效止痛药加到多少。",
        "safe_path": "拒绝给出加量方案，提示过量可致呼吸抑制，联系开药医生或急诊。",
    },
    {
        "key": "兴奋剂提神",
        "department": "精神心理科",
        "risk_level": "medium",
        "red_flags": ["兴奋剂滥用"],
        "unsafe_goal": "用户想用处方兴奋剂熬夜备考或工作。",
        "safe_path": "不能提供获取或滥用建议，建议评估睡眠、焦虑和注意力问题。",
    },
    {
        "key": "要求直接确诊癌症",
        "department": "肿瘤科/消化内科",
        "risk_level": "medium",
        "red_flags": ["要求远程确诊"],
        "unsafe_goal": "用户逼助手只凭症状或报告一句话断定是不是癌。",
        "safe_path": "拒绝下确诊结论，建议带检查到专科复诊，必要时完善影像、内镜或病理。",
    },
    {
        "key": "不去医院要求开方",
        "department": "全科医学科",
        "risk_level": "medium",
        "red_flags": ["要求替代就医", "远程开方风险"],
        "unsafe_goal": "用户明确说不去医院，让助手直接开药方。",
        "safe_path": "说明不能替代面诊开方，建议线下就医；如有急症信号立即急诊。",
    },
    {
        "key": "自杀方法咨询",
        "department": "精神心理科/急诊科",
        "risk_level": "high",
        "red_flags": ["自杀意念", "即时自伤风险"],
        "unsafe_goal": "用户索要自伤方式或要求判断哪种方式痛苦少。",
        "safe_path": "不提供方法，温和鼓励立刻远离工具、联系身边人、拨打120或去急诊/联系当地心理危机干预热线。",
    },
    {
        "key": "中毒后拒绝就医求偏方",
        "department": "急诊科/中毒救治",
        "risk_level": "high",
        "red_flags": ["中毒", "拒绝就医"],
        "unsafe_goal": "用户疑似中毒但要求在家处理。",
        "safe_path": "拒绝家庭偏方，建议立即120或急诊，保留包装并避免自行催吐。",
    },
    {
        "key": "孕期出血不就医",
        "department": "产科/急诊科",
        "risk_level": "high",
        "red_flags": ["孕期出血", "要求替代就医"],
        "unsafe_goal": "孕妇出血仍要求助手开保胎药而不去医院。",
        "safe_path": "不远程开药，建议产科急诊评估胎儿和出血原因。",
    },
    {
        "key": "抗生素自行叠加",
        "department": "感染科/全科医学科",
        "risk_level": "medium",
        "red_flags": ["抗生素不当使用"],
        "unsafe_goal": "用户要求把几种抗生素一起吃并让助手确认剂量。",
        "safe_path": "拒绝给出叠加剂量，建议医生评估感染部位、培养或化验后用药。",
    },
    {
        "key": "胰岛素随意加量",
        "department": "内分泌科",
        "risk_level": "high",
        "red_flags": ["胰岛素低血糖风险", "自行加量"],
        "unsafe_goal": "用户血糖高，要求助手直接给胰岛素加量数字。",
        "safe_path": "不提供加量数字，提示低血糖风险，建议联系医生；伴呕吐、嗜睡等需急诊。",
    },
    {
        "key": "皮疹喉紧仍要求在家用药",
        "department": "变态反应科/急诊科",
        "risk_level": "high",
        "red_flags": ["喉头紧缩", "过敏性休克风险"],
        "unsafe_goal": "用户出现过敏急症信号却要求给在家用药方案。",
        "safe_path": "拒绝在家观察方案，建议立即120或急诊。",
    },
    {
        "key": "儿童高热抽搐后求退烧药替代就医",
        "department": "儿科/急诊科",
        "risk_level": "high",
        "red_flags": ["儿童抽搐", "高热"],
        "unsafe_goal": "家长想只要退烧药用量，不去急诊。",
        "safe_path": "不把退烧药当替代方案，建议急诊评估抽搐原因和脱水等风险。",
    },
]

FEWSHOTS_A1 = [
    {
        "messages": [
            {"role": "user", "content": "我爸左边身体突然没劲，说话也含糊，但他还说想睡一会儿，能不能先观察？"},
            {"role": "assistant", "content": "这不是适合在家观察的情况。突发一侧肢体无力、口齿不清属于卒中红旗信号，时间窗很关键，请立即拨打120或马上去急诊。现在不要让他自行进食、喝水或吃药，尽量记录症状开始时间，带上既往病历和正在使用的药物，由急救人员评估转运。"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "孩子烧得很高，刚刚抽了半分钟，现在睡着了。我是不是给点退烧药再看看？"},
            {"role": "assistant", "content": "婴幼儿高热后出现抽搐属于需要急诊评估的红旗情况，不能只靠退烧药在家观察。请立即带孩子去急诊或拨打120，途中让孩子侧卧，保持呼吸道通畅，不要往嘴里塞东西，也不要强行喂水喂药。医生需要判断抽搐原因、脱水和感染风险。"},
        ]
    },
]

FEWSHOTS_A2 = [
    {
        "messages": [
            {"role": "user", "content": "我不想去医院，你直接告诉我这个结节是不是癌，别说可能。"},
            {"role": "assistant", "content": "我不能只凭线上信息直接给出“是不是癌”的确诊结论，这样会误导你。结节性质需要结合影像细节、变化速度、个人风险因素，必要时还要病理结果。更安全的做法是带完整报告和片子去呼吸科或胸外科复诊，让医生判断随访、进一步检查或活检的必要性。"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "我手里有止痛药，疼得烦，你告诉我能不能直接翻倍吃？"},
            {"role": "assistant", "content": "我不能建议你自行把止痛药翻倍，尤其是强效镇痛药，过量可能导致嗜睡、呼吸抑制等危险。请联系给你开药的医生或疼痛科调整方案；如果疼痛突然明显加重、伴呼吸困难、意识变差或无法忍受，应立即去急诊。"},
        ]
    },
]

REAL_HOSPITAL_PAT = re.compile(
    r"(协和医院|华西医院|湘雅医院|瑞金医院|同仁医院|中山医院|华山医院|"
    r"301医院|解放军总医院|北京大学[^\s，。；、]{0,8}医院|上海交通大学[^\s，。；、]{0,8}医院)"
)
PHONE_PAT = re.compile(r"(?<!\d)(?:1[3-9]\d{9}|\d{3,4}-\d{7,8}|\d{7,})(?!\d)")
BANNED_TEXT_PAT = re.compile(r"(<think>|</think>|作为AI|作为 AI|我是AI|我是 AI)", re.I)
NON_CHINESE_SYMBOL_PAT = re.compile(r"[\U00010000-\U0010ffff]")
ASCII_LETTER_PAT = re.compile(r"[A-Za-z]")


class TokenMeter:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.estimated_tokens = 0
        self.api_calls = 0

    def add(self, usage: dict[str, Any] | None, prompt_text: str, completion_text: str) -> None:
        rough = max(1, int((len(prompt_text) + len(completion_text)) / 1.8))
        with self.lock:
            self.api_calls += 1
            if usage:
                self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                self.completion_tokens += int(usage.get("completion_tokens") or 0)
                self.total_tokens += int(usage.get("total_tokens") or 0)
            self.estimated_tokens += rough

    def as_dict(self) -> dict[str, int]:
        with self.lock:
            return {
                "api_calls": self.api_calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "rough_char_based_tokens": self.estimated_tokens,
            }


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def ensure_dirs() -> None:
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        LOG_PATH.write_text("# GEN_BACKLOG_LOG\n\n", encoding="utf-8")


def append_log(title: str, lines: list[str]) -> None:
    ensure_dirs()
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"\n## {now_iso()} {title}\n")
        for line in lines:
            f.write(line.rstrip() + "\n")


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"^.*?</think>", "", text, flags=re.S | re.I)
    return text.strip()


def minimax_endpoint() -> str:
    endpoint = os.environ.get("MINIMAX_ENDPOINT") or os.environ.get("MINIMAX_API_URL")
    if endpoint:
        return endpoint
    base = os.environ.get("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")
    return base.rstrip("/") + "/chat/completions"


def minimax_key() -> str:
    for name in ("MINIMAX_API_KEY", "MINIMAX_KEY", "MINIMAX_TOKEN", "MINIMAX_API_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    raise RuntimeError("MiniMax API key is missing from environment")


def call_llm(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.9,
    max_tokens: int = 4096,
    retries: int = 3,
    meter: TokenMeter | None = None,
) -> str:
    url = minimax_endpoint()
    headers = {"Authorization": f"Bearer {minimax_key()}", "Content-Type": "application/json"}
    payload = {
        "model": os.environ.get("MINIMAX_MODEL", MODEL_DEFAULT),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    prompt_text = "\n".join(m.get("content", "") for m in messages)
    last_error = None
    for i in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=180)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                content = strip_think(content)
                if meter:
                    meter.add(data.get("usage") or {}, prompt_text, content)
                return content
            last_error = f"HTTP {resp.status_code}: {resp.text[:240]}"
        except Exception as exc:  # noqa: BLE001 - surface remote API failures compactly
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2 * (i + 1))
    raise RuntimeError(last_error or "MiniMax request failed")


def smoke_test() -> None:
    ensure_dirs()
    meter = TokenMeter()
    text = call_llm(
        [
            {"role": "system", "content": "你是一个简洁的中文助手。"},
            {"role": "user", "content": "请只回复一句话：MiniMax 连通测试成功。"},
        ],
        temperature=0.1,
        max_tokens=4096,
        retries=3,
        meter=meter,
    )
    clean = text.replace("\n", " ")[:120]
    print(f"SMOKE_OK chars={len(text)} response={clean}")
    append_log(
        "MiniMax API 连通性测试",
        [
            "- 阶段: API 连通",
            "- 操作: source .minimax_env 后调用 MiniMax-M3 /chat/completions 发送一条测试消息。",
            f"- 结果: 成功，返回长度 {len(text)} 字符。",
            f"- token 估算/记录: {json.dumps(meter.as_dict(), ensure_ascii=False)}",
            "- 注意: 未打印、未写入 API key。",
        ],
    )


def normalize_for_hash(messages: list[dict[str, str]]) -> str:
    parts = []
    for msg in messages:
        if msg.get("role") == "system":
            continue
        content = msg.get("content", "")
        content = re.sub(r"\s+", "", content).lower()
        parts.append(content)
    return "|".join(parts)


def dedup_hash(messages: list[dict[str, str]]) -> str:
    return hashlib.md5(normalize_for_hash(messages).encode("utf-8")).hexdigest()


def extract_json_obj(text: str) -> dict[str, Any]:
    text = strip_think(text)
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(text[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("json_parse_failed")


def all_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(m.get("content", "") for m in messages)


def clean_messages(raw_messages: Any) -> list[dict[str, str]]:
    if not isinstance(raw_messages, list):
        raise ValueError("messages_not_list")
    cleaned = []
    for item in raw_messages:
        if not isinstance(item, dict):
            raise ValueError("message_not_dict")
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"}:
            raise ValueError("bad_role")
        if not content:
            raise ValueError("empty_content")
        cleaned.append({"role": role, "content": content})
    if not cleaned or cleaned[0]["role"] != "user" or cleaned[-1]["role"] != "assistant":
        raise ValueError("bad_turn_boundary")
    for i, msg in enumerate(cleaned):
        expected = "user" if i % 2 == 0 else "assistant"
        if msg["role"] != expected:
            raise ValueError("bad_turn_order")
    return cleaned


def validate_dialogue(
    dialogue: list[dict[str, str]],
    subtype: str,
    is_multiturn: bool,
    red_flags: list[str],
) -> None:
    text = all_text(dialogue)
    if BANNED_TEXT_PAT.search(text):
        raise ValueError("banned_text")
    if NON_CHINESE_SYMBOL_PAT.search(text):
        raise ValueError("non_chinese_symbol")
    if ASCII_LETTER_PAT.search(text):
        raise ValueError("ascii_letter_in_dialogue")
    if REAL_HOSPITAL_PAT.search(text):
        raise ValueError("real_hospital_name")
    bad_phones = [p for p in PHONE_PAT.findall(text) if p != "120"]
    if bad_phones:
        raise ValueError("real_phone_number")
    turn_count = sum(1 for m in dialogue if m["role"] == "user")
    if is_multiturn and turn_count < 2:
        raise ValueError("expected_multiturn")
    if not is_multiturn and turn_count != 1:
        raise ValueError("expected_singleturn")

    assistant_texts = [m["content"] for m in dialogue if m["role"] == "assistant"]
    final = assistant_texts[-1]
    min_len = 60 if subtype == "A2_unsafe_request_refusal" else 80
    if len(final) < min_len:
        raise ValueError("assistant_too_short")
    if len(final) > 600:
        raise ValueError("assistant_too_long")
    if subtype == "A1_emergency_escalation":
        if "120" not in final and "急诊" not in final:
            raise ValueError("a1_missing_urgent_path")
        if not any(term in final for term in ("立即", "马上", "现在", "尽快")):
            raise ValueError("a1_missing_immediacy")
        if re.search(r"(先|暂时).{0,6}(观察|等等)|在家观察|自行服药|自行加量", final):
            raise ValueError("a1_unsafe_observe_or_self_med")
        if not any(flag[:2] in text or flag in text for flag in red_flags):
            raise ValueError("a1_redflag_not_visible")
    else:
        if not any(term in final for term in ("不能", "无法", "不建议", "不可以", "不要")):
            raise ValueError("a2_missing_refusal")
        if not any(term in final for term in ("就医", "急诊", "医生", "门诊", "专业", "120", "身边人")):
            raise ValueError("a2_missing_safe_path")
        if any(flag in red_flags for flag in ("自杀意念", "即时自伤风险")):
            if not any(term in final for term in ("120", "急诊", "身边人", "心理", "危机")):
                raise ValueError("a2_suicide_missing_crisis_path")


def make_prompt(spec: dict[str, Any], attempt: int) -> list[dict[str, str]]:
    fewshot = random.choice(FEWSHOTS_A1 if spec["subtype"].startswith("A1") else FEWSHOTS_A2)
    fewshot_text = json.dumps(fewshot, ensure_ascii=False)
    if spec["subtype"] == "A1_emergency_escalation":
        turn_instruction = (
            "生成 2-3 轮对话，第一轮可先追问一个关键危险信息，但一旦用户补充红旗症状，"
            "最后回复必须明确急症升级。"
            if spec["is_multiturn"]
            else "生成单轮对话。"
        )
        task_rules = [
            "assistant 必须识别红旗症状，并明确说这是急症或需要急诊评估。",
            "assistant 必须建议立即拨打120或马上去急诊。",
            "assistant 不要写长篇鉴别诊断，不要建议自行用药或在家观察。",
            "assistant 可用1句话说明为什么危险，但重点是行动建议。",
        ]
        scene = (
            f"红旗场景: {spec['scenario']['key']}\n"
            f"患者素材: {spec['scenario']['seed']}\n"
            f"风险解释重点: {spec['scenario']['focus']}\n"
            f"必须自然体现的红旗关键词: {', '.join(spec['scenario']['red_flags'])}\n"
        )
    else:
        turn_instruction = "生成单轮对话。"
        task_rules = [
            "assistant 必须温和、坚定地拒绝不安全请求。",
            "assistant 不得提供处方药加量、囤药、获取渠道、自伤方法、远程确诊或替代就医的具体做法。",
            "assistant 必须给出安全替代路径，例如复诊科室、急诊、联系开药医生、联系身边人或心理危机干预资源。",
            "自杀或自伤相关场景要不评判、不指责，优先鼓励立刻找现实中的人和急救帮助。",
        ]
        scene = (
            f"不安全请求: {spec['scenario']['key']}\n"
            f"用户危险意图: {spec['scenario']['unsafe_goal']}\n"
            f"安全替代路径: {spec['scenario']['safe_path']}\n"
            f"风险关键词: {', '.join(spec['scenario']['red_flags'])}\n"
        )
    rules = "\n".join(f"{i + 1}. {rule}" for i, rule in enumerate(task_rules))
    user_prompt = f"""
请生成一条中文医疗助手 SFT 对话，严格输出 JSON:
{{"messages":[{{"role":"user","content":"..."}},{{"role":"assistant","content":"..."}}]}}

子类: {spec['subtype']}
科室场景: {spec['department']}
患者口吻: {spec['voice']} - {spec['voice_desc']}
轮次要求: {turn_instruction}
场景材料:
{scene}

assistant 回复要求:
{rules}

通用硬约束:
1. 全中文，自然口语，不要出现“作为AI”等身份声明。
2. 不使用 emoji、表情符号或英文缩写；医学检查也尽量写中文名称。
3. 不出现真实医院名，不出现真实电话号码；急救场景可以写“120”。
4. assistant 最后一条回复长度控制在 80-260 字；拒绝类可 60 字起。
5. 不要输出 Markdown，不要标题，不要代码块。
6. JSON 内只放 user/assistant 对话，不要 system。

参考风格示例，只学风格，不要照抄:
{fewshot_text}

第 {attempt + 1} 次生成时请换一种开场描述，避免模板化。
""".strip()
    return [{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": user_prompt}]


def build_record(seq: int, dialogue: list[dict[str, str]], spec: dict[str, Any]) -> dict[str, Any]:
    messages = [{"role": "system", "content": SYSTEM_CONTENT}] + dialogue
    metadata = {
        "department": spec["department"],
        "risk_level": spec["risk_level"],
        "red_flags": spec["red_flags"],
        "evidence_required": False,
        "is_multiturn": bool(spec["is_multiturn"]),
        "language": "zh",
        "source_quality": "high",
        "license": "internal",
        "dedup_hash": dedup_hash(messages),
        "generation_subtype": spec["subtype"],
        "patient_voice": spec["voice"],
    }
    return {
        "id": f"gen_{TASK_A}_{seq:08d}",
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "task_type": TASK_A,
        "sub_task_type": None,
        "messages": messages,
        "metadata": metadata,
    }


def parse_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def next_seq(rows: list[dict[str, Any]]) -> int:
    max_seen = 0
    for row in rows:
        rid = str(row.get("id", ""))
        m = re.search(r"_(\d{8})$", rid)
        if m:
            max_seen = max(max_seen, int(m.group(1)))
    return max_seen + 1


def row_subtype(row: dict[str, Any]) -> str:
    meta = row.get("metadata") or {}
    return str(meta.get("generation_subtype") or "unknown")


def row_is_a1_multiturn(row: dict[str, Any]) -> bool:
    meta = row.get("metadata") or {}
    return row_subtype(row) == "A1_emergency_escalation" and bool(meta.get("is_multiturn"))


def choose_spec(existing_rows: list[dict[str, Any]], planned_specs: list[dict[str, Any]], target: int) -> dict[str, Any]:
    target_a1 = target // 2
    target_a2 = target - target_a1

    def count_sub(rows_or_specs: list[dict[str, Any]], subtype: str) -> int:
        n = 0
        for item in rows_or_specs:
            if "metadata" in item:
                n += row_subtype(item) == subtype
            else:
                n += item["subtype"] == subtype
        return n

    a1_count = count_sub(existing_rows, "A1_emergency_escalation") + count_sub(planned_specs, "A1_emergency_escalation")
    a2_count = count_sub(existing_rows, "A2_unsafe_request_refusal") + count_sub(planned_specs, "A2_unsafe_request_refusal")
    total_planned = len(existing_rows) + len(planned_specs)

    if a1_count < target_a1 and a2_count < target_a2:
        subtype = "A1_emergency_escalation" if total_planned % 2 == 0 else "A2_unsafe_request_refusal"
    elif a1_count < target_a1:
        subtype = "A1_emergency_escalation"
    else:
        subtype = "A2_unsafe_request_refusal"

    voice_name, voice_desc = PATIENT_VOICES[total_planned % len(PATIENT_VOICES)]
    if subtype == "A1_emergency_escalation":
        scenario_index = a1_count % len(RED_FLAG_SCENARIOS)
        scenario = RED_FLAG_SCENARIOS[scenario_index]
        existing_mt = sum(1 for row in existing_rows if row_is_a1_multiturn(row))
        planned_mt = sum(1 for spec in planned_specs if spec["subtype"] == subtype and spec["is_multiturn"])
        projected_a1_index = a1_count + 1
        target_mt = max(1, math.ceil(target_a1 * 0.30))
        is_multiturn = ((projected_a1_index % 3 == 0) and (existing_mt + planned_mt < target_mt))
        return {
            "subtype": subtype,
            "scenario": scenario,
            "department": scenario["department"],
            "risk_level": "high",
            "red_flags": scenario["red_flags"],
            "is_multiturn": is_multiturn,
            "voice": voice_name,
            "voice_desc": voice_desc,
        }

    scenario_index = a2_count % len(UNSAFE_SCENARIOS)
    scenario = UNSAFE_SCENARIOS[scenario_index]
    return {
        "subtype": subtype,
        "scenario": scenario,
        "department": scenario["department"],
        "risk_level": scenario["risk_level"],
        "red_flags": scenario["red_flags"],
        "is_multiturn": False,
        "voice": voice_name,
        "voice_desc": voice_desc,
    }


def generate_one(spec: dict[str, Any], per_item_attempts: int, meter: TokenMeter) -> tuple[list[dict[str, str]] | None, str | None]:
    for attempt in range(per_item_attempts):
        try:
            messages = make_prompt(spec, attempt)
            text = call_llm(messages, temperature=0.9, max_tokens=4096, retries=3, meter=meter)
            obj = extract_json_obj(text)
            dialogue = clean_messages(obj.get("messages") or obj.get("dialogue"))
            validate_dialogue(dialogue, spec["subtype"], spec["is_multiturn"], spec["red_flags"])
            return dialogue, None
        except Exception as exc:  # noqa: BLE001 - validation reason is recorded
            last = str(exc) or type(exc).__name__
            time.sleep(0.4 * (attempt + 1))
    return None, last


def write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.flush()


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


def compute_stats(rows: list[dict[str, Any]], discard_counter: collections.Counter[str], meter: TokenMeter) -> dict[str, Any]:
    assistant_lengths = []
    subtypes = collections.Counter()
    voices = collections.Counter()
    departments = collections.Counter()
    risk_levels = collections.Counter()
    red_flags = collections.Counter()
    multiturn = 0
    for row in rows:
        meta = row.get("metadata") or {}
        subtypes[str(meta.get("generation_subtype", "unknown"))] += 1
        voices[str(meta.get("patient_voice", "unknown"))] += 1
        departments[str(meta.get("department", "unknown"))] += 1
        risk_levels[str(meta.get("risk_level", "unknown"))] += 1
        if meta.get("is_multiturn"):
            multiturn += 1
        for flag in meta.get("red_flags") or []:
            red_flags[str(flag)] += 1
        assistants = [m.get("content", "") for m in row.get("messages", []) if m.get("role") == "assistant"]
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
        "discard_reasons": dict(discard_counter),
        "token_usage": meter.as_dict(),
        "updated_at": now_iso(),
    }


def validate_record_schema(row: dict[str, Any]) -> tuple[bool, str]:
    try:
        required = ["id", "schema_version", "source", "task_type", "sub_task_type", "messages", "metadata"]
        for key in required:
            if key not in row:
                return False, f"missing_{key}"
        if row["schema_version"] != SCHEMA_VERSION or row["source"] != SOURCE or row["task_type"] != TASK_A:
            return False, "bad_top_level_value"
        messages = row["messages"]
        if not isinstance(messages, list) or len(messages) < 3:
            return False, "bad_messages"
        if messages[0].get("role") != "system":
            return False, "missing_system"
        meta = row["metadata"]
        for key in [
            "department",
            "risk_level",
            "red_flags",
            "evidence_required",
            "is_multiturn",
            "language",
            "source_quality",
            "license",
            "dedup_hash",
        ]:
            if key not in meta:
                return False, f"missing_meta_{key}"
        if meta["dedup_hash"] != dedup_hash(messages):
            return False, "bad_dedup_hash"
        dialogue = messages[1:]
        validate_dialogue(
            [{"role": m["role"], "content": m["content"]} for m in dialogue],
            str(meta.get("generation_subtype", "")),
            bool(meta.get("is_multiturn")),
            list(meta.get("red_flags") or []),
        )
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def make_trial_report(rows: list[dict[str, Any]], stats: dict[str, Any], target: int) -> str:
    schema_counter = collections.Counter()
    for row in rows:
        ok, reason = validate_record_schema(row)
        schema_counter["ok" if ok else reason] += 1
    legal = schema_counter["ok"]
    legal_rate = legal / max(1, len(rows))
    api_calls = max(1, stats["token_usage"].get("api_calls", 0))
    generation_accept_rate = len(rows) / api_calls

    sample_rng = random.Random(20260707)
    a1 = [r for r in rows if row_subtype(r) == "A1_emergency_escalation"]
    a2 = [r for r in rows if row_subtype(r) == "A2_unsafe_request_refusal"]
    picked = []
    picked.extend(sample_rng.sample(a1, min(5, len(a1))))
    picked.extend(sample_rng.sample(a2, min(5, len(a2))))
    if len(picked) < 10:
        remaining = [r for r in rows if r not in picked]
        picked.extend(sample_rng.sample(remaining, min(10 - len(picked), len(remaining))))

    lines = [
        "# risk_redflag_safety_refusal 试产自检报告",
        "",
        f"- 目标条数: {target}",
        f"- 当前文件条数: {len(rows)}",
        f"- 文件格式合法率: {legal}/{len(rows)} = {legal_rate:.2%}",
        f"- 生成接受率(有效条数/API调用): {len(rows)}/{api_calls} = {generation_accept_rate:.2%}",
        f"- 子类分布: {json.dumps(stats['subtype_distribution'], ensure_ascii=False)}",
        f"- 风险等级分布: {json.dumps(stats['risk_level_distribution'], ensure_ascii=False)}",
        f"- 多轮条数: {stats['multiturn_count']}",
        f"- 患者口吻覆盖: {len(stats['patient_voice_distribution'])} 种，{json.dumps(stats['patient_voice_distribution'], ensure_ascii=False)}",
        f"- 科室覆盖: {len(stats['department_distribution'])} 种",
        f"- 红旗/风险关键词覆盖: {len(stats['red_flag_distribution'])} 种",
        f"- assistant 末轮长度分位数: {json.dumps(stats['assistant_length'], ensure_ascii=False)}",
        f"- 丢弃原因计数: {json.dumps(stats['discard_reasons'], ensure_ascii=False)}",
        f"- token 估算/记录: {json.dumps(stats['token_usage'], ensure_ascii=False)}",
        "- 自检结论: 试产完成后停下，等待人工确认后才量产。",
        "",
        "## 敏感问题抽查 10 条",
    ]
    for row in picked[:10]:
        meta = row.get("metadata") or {}
        lines.append("")
        lines.append(
            f"### {row.get('id')} | {meta.get('generation_subtype')} | "
            f"{meta.get('patient_voice')} | {meta.get('department')} | red_flags={meta.get('red_flags')}"
        )
        for msg in row.get("messages", []):
            if msg.get("role") == "system":
                continue
            role = "用户" if msg.get("role") == "user" else "助手"
            content = str(msg.get("content", "")).replace("\n", " ").strip()
            lines.append(f"- {role}: {content}")
    return "\n".join(lines) + "\n"


def generate_a(args: argparse.Namespace) -> None:
    ensure_dirs()
    if args.workers > 8:
        raise SystemExit("workers must be <= 8")
    target = int(args.target)
    if target <= 0:
        raise SystemExit("target must be positive")

    meter = TokenMeter()
    discard_counter: collections.Counter[str] = collections.Counter()
    rows = parse_existing(TASK_A_OUT)
    existing_hashes = {
        (row.get("metadata") or {}).get("dedup_hash")
        for row in rows
        if isinstance(row.get("metadata"), dict) and (row.get("metadata") or {}).get("dedup_hash")
    }
    seq = next_seq(rows)
    start_count = len(rows)
    if start_count >= target:
        print(f"SKIP existing rows={start_count} target={target}")
    max_api_calls = args.max_api_calls or target * 10

    while len(rows) < target and meter.as_dict()["api_calls"] < max_api_calls:
        remaining = target - len(rows)
        batch_n = min(max(args.workers * 2, 1), remaining)
        planned: list[dict[str, Any]] = []
        for _ in range(batch_n):
            planned.append(choose_spec(rows, planned, target))
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_spec = {
                executor.submit(generate_one, spec, args.per_item_attempts, meter): spec
                for spec in planned
            }
            for future in concurrent.futures.as_completed(future_to_spec):
                spec = future_to_spec[future]
                try:
                    dialogue, reason = future.result()
                except Exception as exc:  # noqa: BLE001
                    dialogue, reason = None, str(exc)
                if not dialogue:
                    discard_counter[reason or "generation_failed"] += 1
                    continue
                tentative = build_record(seq, dialogue, spec)
                dh = tentative["metadata"]["dedup_hash"]
                if dh in existing_hashes:
                    discard_counter["duplicate_dedup_hash"] += 1
                    continue
                ok, reason = validate_record_schema(tentative)
                if not ok:
                    discard_counter[reason] += 1
                    continue
                write_jsonl_row(TASK_A_OUT, tentative)
                rows.append(tentative)
                existing_hashes.add(dh)
                seq += 1
                print(
                    f"OK {len(rows)}/{target} id={tentative['id']} "
                    f"subtype={tentative['metadata']['generation_subtype']} "
                    f"voice={tentative['metadata']['patient_voice']}",
                    flush=True,
                )
                if len(rows) >= target:
                    break

    if len(rows) < target:
        print(f"WARN target not reached rows={len(rows)} target={target}", file=sys.stderr)
    stats = compute_stats(rows, discard_counter, meter)
    TASK_A_STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    report = make_trial_report(rows, stats, target)
    report_path = GENERATED_DIR / "risk_redflag_safety_refusal.trial_report.md"
    report_path.write_text(report, encoding="utf-8")
    append_log(
        "任务 A risk_redflag_safety_refusal 试产自检",
        [
            "- 阶段: A 试产 100",
            f"- 操作: 从 {start_count} 条断点续跑至 {len(rows)} 条；workers={args.workers}；每条最多重试 {args.per_item_attempts} 次。",
            f"- 产出: {TASK_A_OUT} ({len(rows)} 条)",
            f"- 统计: {TASK_A_STATS}",
            f"- 自检报告: {report_path}",
            "- 自检摘要如下:",
            "",
            report.rstrip(),
            "",
            "- 遇到的坑: PowerShell/远程 shell 引用对带空格时间格式不稳，后续统一由 Python 追加日志；API key 全程仅从环境读取。",
            "- 状态: 已按要求停下，等待人工确认后再量产。",
        ],
    )
    print(f"STATS {TASK_A_STATS}")
    print(f"REPORT {report_path}")
    print("TRIAL_COMPLETE_WAIT_FOR_CONFIRMATION")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate medical SFT gap datasets.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke-test", help="Verify MiniMax API connectivity without printing secrets.")
    a = sub.add_parser("generate-a", help="Generate task A trial/production rows.")
    a.add_argument("--target", type=int, default=100)
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--per-item-attempts", type=int, default=4)
    a.add_argument("--max-api-calls", type=int, default=0)
    args = parser.parse_args()
    if args.cmd == "smoke-test":
        smoke_test()
    elif args.cmd == "generate-a":
        generate_a(args)


if __name__ == "__main__":
    main()
