#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate task B: test_report_explanation."""

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


TASK_B = "test_report_explanation"
OUT_PATH = GENERATED_DIR / "test_report_explanation.jsonl"
STATS_PATH = GENERATED_DIR / "test_report_explanation.stats.json"
REPORT_PATH = GENERATED_DIR / "test_report_explanation.trial_report.md"

BANNED = re.compile(r"(<think>|</think>|作为AI|作为 AI|我是AI|我是 AI)", re.I)
ASCII_LETTER = re.compile(r"[A-Za-z]")
HIGH_SYMBOL = re.compile(r"[\U00010000-\U0010ffff]")
REAL_HOSPITAL = re.compile(r"(协和医院|华西医院|湘雅医院|瑞金医院|同仁医院|中山医院|华山医院|301医院|解放军总医院)")
PHONE = re.compile(r"(?<!\d)(?:1[3-9]\d{9}|\d{3,4}-\d{7,8}|\d{7,})(?!\d)")


REPORT_TYPES = [
    "血常规",
    "肝功能",
    "肾功能",
    "甲状腺功能",
    "血脂",
    "尿常规",
    "腹部超声",
    "胸部计算机断层扫描",
    "心电图",
    "肿瘤标志物",
]

DEPARTMENT_BY_TYPE = {
    "血常规": "血液科/感染科",
    "肝功能": "消化内科/肝病门诊",
    "肾功能": "肾内科",
    "甲状腺功能": "内分泌科",
    "血脂": "心血管内科/内分泌科",
    "尿常规": "肾内科/泌尿外科",
    "腹部超声": "消化内科/肝胆外科",
    "胸部计算机断层扫描": "呼吸内科",
    "心电图": "心血管内科",
    "肿瘤标志物": "肿瘤科/相关专科",
}


def stable_hash(messages: list[dict[str, str]]) -> str:
    parts = []
    for msg in messages:
        if msg.get("role") == "system":
            continue
        parts.append(re.sub(r"\s+", "", msg.get("content", "")).lower())
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()


def fmt(value: float, digits: int = 1) -> str:
    if digits == 0:
        return str(int(round(value)))
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def value(rng: random.Random, low: float, high: float, digits: int = 1) -> str:
    return fmt(rng.uniform(low, high), digits)


def line(name: str, val: str, ref: str, unit: str, abnormal: bool = False) -> str:
    return f"{name}: {val}{unit}（参考范围: {ref}{unit}）"


def make_blood(idx: int, rng: random.Random) -> dict[str, Any]:
    pattern = idx % 5
    if pattern == 0:
        rows = [
            line("白细胞计数", value(rng, 12.0, 18.5), "3.5-9.5", " 十的九次方每升", True),
            line("中性粒细胞比例", value(rng, 78, 90, 0), "40-75", "%", True),
            line("血红蛋白", value(rng, 125, 150, 0), "115-150", " 克每升"),
            line("血小板计数", value(rng, 180, 320, 0), "125-350", " 十的九次方每升"),
        ]
        abnormal = ["白细胞计数升高", "中性粒细胞比例升高"]
        direction = "更偏向细菌感染或明显炎症反应，需结合发热、咳痰、腹痛等症状判断感染部位"
        risk = "medium"
        flags = ["白细胞升高", "中性粒细胞升高"]
    elif pattern == 1:
        rows = [
            line("白细胞计数", value(rng, 2.1, 3.2), "3.5-9.5", " 十的九次方每升", True),
            line("中性粒细胞比例", value(rng, 32, 45, 0), "40-75", "%", True),
            line("淋巴细胞比例", value(rng, 46, 58, 0), "20-50", "%", True),
            line("血小板计数", value(rng, 130, 260, 0), "125-350", " 十的九次方每升"),
        ]
        abnormal = ["白细胞计数降低", "中性粒细胞比例偏低", "淋巴细胞比例偏高"]
        direction = "可能见于病毒感染恢复期、药物影响或骨髓抑制等情况，需要结合用药史和复查变化"
        risk = "medium"
        flags = ["白细胞降低", "中性粒细胞偏低"]
    elif pattern == 2:
        rows = [
            line("红细胞计数", value(rng, 3.0, 3.6), "3.8-5.1", " 十的十二次方每升", True),
            line("血红蛋白", value(rng, 72, 98, 0), "115-150", " 克每升", True),
            line("平均红细胞体积", value(rng, 62, 76, 0), "82-100", " 飞升", True),
            line("血小板计数", value(rng, 250, 420, 0), "125-350", " 十的九次方每升", True),
        ]
        abnormal = ["血红蛋白降低", "平均红细胞体积偏小", "血小板计数偏高"]
        direction = "组合上较符合小细胞性贫血方向，常见原因包括缺铁，需结合铁蛋白、月经量或消化道出血风险"
        risk = "medium"
        flags = ["贫血", "小细胞性改变"]
    elif pattern == 3:
        rows = [
            line("白细胞计数", value(rng, 4.2, 7.8), "3.5-9.5", " 十的九次方每升"),
            line("血红蛋白", value(rng, 132, 155, 0), "115-150", " 克每升"),
            line("血小板计数", value(rng, 35, 68, 0), "125-350", " 十的九次方每升", True),
            line("平均血小板体积", value(rng, 11.5, 13.2), "7.4-10.4", " 飞升", True),
        ]
        abnormal = ["血小板计数明显降低", "平均血小板体积升高"]
        direction = "提示出血风险增加，可能与免疫性血小板减少、药物影响或感染后改变有关"
        risk = "high"
        flags = ["血小板明显降低", "出血风险"]
    else:
        rows = [
            line("白细胞计数", value(rng, 4.5, 8.6), "3.5-9.5", " 十的九次方每升"),
            line("中性粒细胞比例", value(rng, 45, 70, 0), "40-75", "%"),
            line("血红蛋白", value(rng, 116, 148, 0), "115-150", " 克每升"),
            line("血小板计数", value(rng, 140, 300, 0), "125-350", " 十的九次方每升"),
        ]
        abnormal = []
        direction = "本次血常规主要指标在参考范围内，如有症状仍需结合病情判断"
        risk = "low"
        flags = []
    return pack("血常规", rows, abnormal, direction, risk, flags)


def make_liver(idx: int, rng: random.Random) -> dict[str, Any]:
    pattern = idx % 4
    if pattern == 0:
        rows = [
            line("谷丙转氨酶", value(rng, 95, 240, 0), "9-50", " 单位每升", True),
            line("谷草转氨酶", value(rng, 80, 190, 0), "15-40", " 单位每升", True),
            line("总胆红素", value(rng, 10, 19), "5-21", " 微摩尔每升"),
            line("白蛋白", value(rng, 39, 45), "35-55", " 克每升"),
        ]
        abnormal = ["谷丙转氨酶升高", "谷草转氨酶升高"]
        direction = "提示肝细胞受损方向，常见于脂肪肝、病毒性肝炎、酒精或药物影响"
        risk = "medium"
        flags = ["转氨酶升高"]
    elif pattern == 1:
        rows = [
            line("总胆红素", value(rng, 42, 88), "5-21", " 微摩尔每升", True),
            line("直接胆红素", value(rng, 25, 60), "0-6.8", " 微摩尔每升", True),
            line("碱性磷酸酶", value(rng, 180, 360, 0), "45-125", " 单位每升", True),
            line("谷氨酰转肽酶", value(rng, 160, 420, 0), "10-60", " 单位每升", True),
        ]
        abnormal = ["总胆红素升高", "直接胆红素升高", "碱性磷酸酶升高", "谷氨酰转肽酶升高"]
        direction = "更像胆汁淤积或胆道梗阻方向，需要结合腹部超声或胆道影像"
        risk = "high"
        flags = ["胆红素升高", "胆汁淤积"]
    elif pattern == 2:
        rows = [
            line("谷丙转氨酶", value(rng, 45, 76, 0), "9-50", " 单位每升", True),
            line("谷草转氨酶", value(rng, 35, 62, 0), "15-40", " 单位每升", True),
            line("谷氨酰转肽酶", value(rng, 70, 135, 0), "10-60", " 单位每升", True),
            line("总胆红素", value(rng, 8, 18), "5-21", " 微摩尔每升"),
        ]
        abnormal = ["谷丙转氨酶轻度升高", "谷草转氨酶轻度升高", "谷氨酰转肽酶升高"]
        direction = "轻中度肝功能异常，可能与脂肪肝、饮酒、药物或近期感染有关"
        risk = "medium"
        flags = ["轻度肝功能异常"]
    else:
        rows = [
            line("谷丙转氨酶", value(rng, 14, 35, 0), "9-50", " 单位每升"),
            line("谷草转氨酶", value(rng, 18, 32, 0), "15-40", " 单位每升"),
            line("总胆红素", value(rng, 7, 16), "5-21", " 微摩尔每升"),
            line("白蛋白", value(rng, 40, 47), "35-55", " 克每升"),
        ]
        abnormal = []
        direction = "本次肝功能主要项目在参考范围内"
        risk = "low"
        flags = []
    return pack("肝功能", rows, abnormal, direction, risk, flags)


def make_kidney(idx: int, rng: random.Random) -> dict[str, Any]:
    pattern = idx % 4
    if pattern == 0:
        rows = [
            line("血肌酐", value(rng, 145, 260, 0), "57-111", " 微摩尔每升", True),
            line("尿素", value(rng, 9.0, 16.0), "3.1-8.0", " 毫摩尔每升", True),
            line("估算肾小球滤过率", value(rng, 28, 55, 0), "大于90", " 毫升每分钟", True),
            line("血钾", value(rng, 4.3, 5.1), "3.5-5.3", " 毫摩尔每升"),
        ]
        abnormal = ["血肌酐升高", "尿素升高", "估算肾小球滤过率降低"]
        direction = "提示肾功能下降方向，需要区分急性肾损伤和慢性肾病"
        risk = "high"
        flags = ["肾功能下降"]
    elif pattern == 1:
        rows = [
            line("血肌酐", value(rng, 190, 330, 0), "57-111", " 微摩尔每升", True),
            line("尿素", value(rng, 12.0, 22.0), "3.1-8.0", " 毫摩尔每升", True),
            line("估算肾小球滤过率", value(rng, 18, 38, 0), "大于90", " 毫升每分钟", True),
            line("血钾", value(rng, 5.6, 6.2), "3.5-5.3", " 毫摩尔每升", True),
        ]
        abnormal = ["血肌酐明显升高", "尿素升高", "估算肾小球滤过率明显降低", "血钾升高"]
        direction = "提示较重肾功能异常并伴高钾风险，需尽快线下评估"
        risk = "high"
        flags = ["血钾升高", "肾功能明显异常"]
    elif pattern == 2:
        rows = [
            line("血肌酐", value(rng, 48, 62, 0), "57-111", " 微摩尔每升"),
            line("尿素", value(rng, 2.0, 3.0), "3.1-8.0", " 毫摩尔每升", True),
            line("尿酸", value(rng, 460, 620, 0), "155-357", " 微摩尔每升", True),
            line("血钾", value(rng, 3.8, 4.8), "3.5-5.3", " 毫摩尔每升"),
        ]
        abnormal = ["尿素偏低", "尿酸升高"]
        direction = "尿酸升高更需要关注痛风和代谢风险，尿素偏低可与饮食蛋白摄入少等有关"
        risk = "medium"
        flags = ["尿酸升高"]
    else:
        rows = [
            line("血肌酐", value(rng, 62, 95, 0), "57-111", " 微摩尔每升"),
            line("尿素", value(rng, 3.8, 6.8), "3.1-8.0", " 毫摩尔每升"),
            line("估算肾小球滤过率", value(rng, 92, 125, 0), "大于90", " 毫升每分钟"),
            line("血钾", value(rng, 3.8, 4.9), "3.5-5.3", " 毫摩尔每升"),
        ]
        abnormal = []
        direction = "本次肾功能和血钾大致在参考范围内"
        risk = "low"
        flags = []
    return pack("肾功能", rows, abnormal, direction, risk, flags)


def make_thyroid(idx: int, rng: random.Random) -> dict[str, Any]:
    pattern = idx % 4
    if pattern == 0:
        rows = [
            line("促甲状腺激素", value(rng, 0.01, 0.12, 2), "0.27-4.2", " 微国际单位每毫升", True),
            line("游离甲状腺素", value(rng, 26, 45), "12-22", " 皮摩尔每升", True),
            line("游离三碘甲状腺原氨酸", value(rng, 8.0, 16.0), "3.1-6.8", " 皮摩尔每升", True),
        ]
        abnormal = ["促甲状腺激素降低", "游离甲状腺素升高", "游离三碘甲状腺原氨酸升高"]
        direction = "组合上提示甲状腺功能亢进方向，需结合心慌、手抖、消瘦等症状和抗体检查"
        risk = "medium"
        flags = ["甲状腺功能亢进方向"]
    elif pattern == 1:
        rows = [
            line("促甲状腺激素", value(rng, 8.5, 18.0), "0.27-4.2", " 微国际单位每毫升", True),
            line("游离甲状腺素", value(rng, 6.5, 10.8), "12-22", " 皮摩尔每升", True),
            line("甲状腺过氧化物酶抗体", value(rng, 180, 680, 0), "0-34", " 国际单位每毫升", True),
        ]
        abnormal = ["促甲状腺激素升高", "游离甲状腺素降低", "甲状腺过氧化物酶抗体升高"]
        direction = "提示甲状腺功能减退方向，自身免疫性甲状腺炎可能性需考虑"
        risk = "medium"
        flags = ["甲状腺功能减退方向"]
    elif pattern == 2:
        rows = [
            line("促甲状腺激素", value(rng, 4.8, 7.5), "0.27-4.2", " 微国际单位每毫升", True),
            line("游离甲状腺素", value(rng, 13, 18), "12-22", " 皮摩尔每升"),
            line("游离三碘甲状腺原氨酸", value(rng, 3.8, 5.8), "3.1-6.8", " 皮摩尔每升"),
        ]
        abnormal = ["促甲状腺激素轻度升高"]
        direction = "可见于亚临床甲状腺功能减退，需要复查确认并结合症状、抗体和备孕情况"
        risk = "low"
        flags = ["促甲状腺激素轻度升高"]
    else:
        rows = [
            line("促甲状腺激素", value(rng, 1.0, 3.5), "0.27-4.2", " 微国际单位每毫升"),
            line("游离甲状腺素", value(rng, 13, 20), "12-22", " 皮摩尔每升"),
            line("游离三碘甲状腺原氨酸", value(rng, 3.6, 5.8), "3.1-6.8", " 皮摩尔每升"),
        ]
        abnormal = []
        direction = "本次甲状腺功能主要指标在参考范围内"
        risk = "low"
        flags = []
    return pack("甲状腺功能", rows, abnormal, direction, risk, flags)


def make_lipid(idx: int, rng: random.Random) -> dict[str, Any]:
    pattern = idx % 4
    if pattern == 0:
        rows = [
            line("总胆固醇", value(rng, 5.8, 7.2), "小于5.2", " 毫摩尔每升", True),
            line("甘油三酯", value(rng, 1.8, 2.6), "小于1.7", " 毫摩尔每升", True),
            line("低密度脂蛋白胆固醇", value(rng, 3.8, 5.0), "小于3.4", " 毫摩尔每升", True),
            line("高密度脂蛋白胆固醇", value(rng, 1.0, 1.5), "大于1.0", " 毫摩尔每升"),
        ]
        abnormal = ["总胆固醇升高", "低密度脂蛋白胆固醇升高", "甘油三酯偏高"]
        direction = "提示动脉粥样硬化风险增加方向，需要结合血压、血糖和吸烟等因素分层"
        risk = "medium"
        flags = ["血脂异常"]
    elif pattern == 1:
        rows = [
            line("总胆固醇", value(rng, 4.6, 5.5), "小于5.2", " 毫摩尔每升", True),
            line("甘油三酯", value(rng, 5.8, 12.0), "小于1.7", " 毫摩尔每升", True),
            line("低密度脂蛋白胆固醇", value(rng, 2.2, 3.5), "小于3.4", " 毫摩尔每升"),
            line("高密度脂蛋白胆固醇", value(rng, 0.7, 1.0), "大于1.0", " 毫摩尔每升", True),
        ]
        abnormal = ["甘油三酯明显升高", "高密度脂蛋白胆固醇偏低"]
        direction = "明显高甘油三酯可增加胰腺炎和代谢综合征风险，需尽快复诊评估"
        risk = "high"
        flags = ["甘油三酯明显升高"]
    elif pattern == 2:
        rows = [
            line("总胆固醇", value(rng, 4.0, 5.0), "小于5.2", " 毫摩尔每升"),
            line("甘油三酯", value(rng, 1.8, 2.7), "小于1.7", " 毫摩尔每升", True),
            line("低密度脂蛋白胆固醇", value(rng, 2.2, 3.2), "小于3.4", " 毫摩尔每升"),
            line("高密度脂蛋白胆固醇", value(rng, 1.0, 1.4), "大于1.0", " 毫摩尔每升"),
        ]
        abnormal = ["甘油三酯轻度升高"]
        direction = "常与近期饮食、饮酒、体重和血糖控制有关，可先生活方式干预后复查"
        risk = "low"
        flags = ["甘油三酯轻度升高"]
    else:
        rows = [
            line("总胆固醇", value(rng, 3.8, 5.0), "小于5.2", " 毫摩尔每升"),
            line("甘油三酯", value(rng, 0.7, 1.5), "小于1.7", " 毫摩尔每升"),
            line("低密度脂蛋白胆固醇", value(rng, 1.8, 3.0), "小于3.4", " 毫摩尔每升"),
            line("高密度脂蛋白胆固醇", value(rng, 1.1, 1.8), "大于1.0", " 毫摩尔每升"),
        ]
        abnormal = []
        direction = "本次血脂主要项目在目标范围内"
        risk = "low"
        flags = []
    return pack("血脂", rows, abnormal, direction, risk, flags)


def make_urine(idx: int, rng: random.Random) -> dict[str, Any]:
    pattern = idx % 4
    if pattern == 0:
        rows = [
            "尿白细胞: 阳性二个加号（参考: 阴性）",
            "亚硝酸盐: 阳性（参考: 阴性）",
            "尿潜血: 阳性一个加号（参考: 阴性）",
            "尿蛋白: 阴性（参考: 阴性）",
        ]
        abnormal = ["尿白细胞阳性", "亚硝酸盐阳性", "尿潜血阳性"]
        direction = "更支持尿路感染方向，需结合尿频、尿痛、发热和尿培养"
        risk = "medium"
        flags = ["尿路感染方向"]
    elif pattern == 1:
        rows = [
            "尿蛋白: 阳性二个加号（参考: 阴性）",
            "尿潜血: 阳性二个加号（参考: 阴性）",
            "红细胞计数: 56 个每高倍视野（参考: 0-3 个每高倍视野）",
            "尿白细胞: 阴性（参考: 阴性）",
        ]
        abnormal = ["尿蛋白阳性", "尿潜血阳性", "尿红细胞升高"]
        direction = "提示肾小球或泌尿系统出血、蛋白尿方向，需要肾内科进一步评估"
        risk = "medium"
        flags = ["蛋白尿", "血尿"]
    elif pattern == 2:
        rows = [
            "尿糖: 阳性三个加号（参考: 阴性）",
            "尿酮体: 阳性二个加号（参考: 阴性）",
            "尿蛋白: 阳性一个加号（参考: 阴性）",
            "尿白细胞: 阴性（参考: 阴性）",
        ]
        abnormal = ["尿糖阳性", "尿酮体阳性", "尿蛋白阳性"]
        direction = "提示血糖控制异常或饥饿、呕吐等导致酮体升高，糖尿病患者需警惕代谢失衡"
        risk = "high"
        flags = ["尿糖阳性", "尿酮体阳性"]
    else:
        rows = [
            "尿蛋白: 阴性（参考: 阴性）",
            "尿潜血: 阴性（参考: 阴性）",
            "尿白细胞: 阴性（参考: 阴性）",
            "尿糖: 阴性（参考: 阴性）",
        ]
        abnormal = []
        direction = "本次尿常规主要项目未见明显异常"
        risk = "low"
        flags = []
    return pack("尿常规", rows, abnormal, direction, risk, flags)


def make_ultrasound(idx: int, rng: random.Random) -> dict[str, Any]:
    pattern = idx % 4
    if pattern == 0:
        rows = [
            "肝脏大小正常，实质回声弥漫性增强、细密，肝内管道显示欠清。",
            "胆囊壁不厚，胆囊腔内未见明显结石强回声。",
            "脾脏不大，腹腔未见游离液体。",
        ]
        abnormal = ["肝实质回声弥漫性增强"]
        direction = "提示脂肪肝方向，需结合肝功能、血脂、体重和饮酒史"
        risk = "low"
        flags = ["脂肪肝方向"]
    elif pattern == 1:
        rows = [
            "胆囊大小约八点二乘三点四厘米，壁稍厚。",
            "胆囊腔内见多个强回声团，较大约一点二厘米，后方伴声影，随体位改变移动。",
            "肝内外胆管未见明显扩张。",
        ]
        abnormal = ["胆囊多发强回声伴声影", "胆囊壁稍厚"]
        direction = "符合胆囊结石方向，若伴右上腹痛、发热需评估胆囊炎"
        risk = "medium"
        flags = ["胆囊结石方向"]
    elif pattern == 2:
        rows = [
            "右肾集合系统分离，前后径约一点八厘米。",
            "右侧输尿管上段轻度扩张，远端显示欠清。",
            "左肾大小形态未见明显异常，膀胱充盈可。",
        ]
        abnormal = ["右肾集合系统分离", "右侧输尿管轻度扩张"]
        direction = "提示右侧尿路梗阻或积水方向，常需排查输尿管结石等原因"
        risk = "medium"
        flags = ["肾积水方向"]
    else:
        rows = [
            "肝脏大小形态正常，实质回声均匀。",
            "胆囊大小正常，壁不厚，腔内未见明显异常回声。",
            "双肾大小形态正常，集合系统未见分离。",
        ]
        abnormal = []
        direction = "本次腹部超声描述未见明显异常"
        risk = "low"
        flags = []
    return pack("腹部超声", rows, abnormal, direction, risk, flags)


def make_chest_scan(idx: int, rng: random.Random) -> dict[str, Any]:
    pattern = idx % 4
    if pattern == 0:
        rows = [
            "双肺纹理增多，右下肺见片状磨玻璃影及少量实变影，边界欠清。",
            "气管及主支气管通畅，纵隔内未见明显肿大淋巴结。",
            "胸腔未见明显积液。",
        ]
        abnormal = ["右下肺磨玻璃影及实变影"]
        direction = "更偏向感染性炎症改变，需要结合发热、咳嗽、血常规和炎症指标"
        risk = "medium"
        flags = ["肺部炎症方向"]
    elif pattern == 1:
        size = value(rng, 5, 8, 0)
        rows = [
            f"左上肺见磨玻璃结节，大小约{size}毫米，边界尚清，未见明显毛刺。",
            "双肺未见大片实变影，胸腔未见积液。",
            "纵隔内未见明显肿大淋巴结。",
        ]
        abnormal = ["左上肺磨玻璃结节"]
        direction = "提示肺小结节方向，多数需要按大小和风险因素定期随访，不宜仅凭一次影像判断性质"
        risk = "medium"
        flags = ["肺结节"]
    elif pattern == 2:
        rows = [
            "右肺门旁见不规则软组织密度影，边缘欠清，邻近支气管受压变窄。",
            "纵隔内见多发稍大淋巴结，短径约一点一厘米。",
            "右侧胸腔见少量积液。",
        ]
        abnormal = ["右肺门旁不规则软组织密度影", "纵隔淋巴结稍大", "右侧少量胸腔积液"]
        direction = "需要警惕占位性病变方向，必须结合增强影像、支气管镜或病理等进一步明确"
        risk = "high"
        flags = ["占位性病变警示"]
    else:
        rows = [
            "双肺透亮度尚可，未见明确新发实变影或结节影。",
            "气管居中，纵隔未见明显肿大淋巴结。",
            "双侧胸腔未见积液。",
        ]
        abnormal = []
        direction = "本次胸部影像未描述明确异常，如仍有症状需结合临床随访"
        risk = "low"
        flags = []
    return pack("胸部计算机断层扫描", rows, abnormal, direction, risk, flags)


def make_ecg(idx: int, rng: random.Random) -> dict[str, Any]:
    pattern = idx % 4
    if pattern == 0:
        rows = [
            "心率: 每分钟一百二十八次（参考: 每分钟六十至一百次）",
            "节律: 窦性心律。",
            "结论: 窦性心动过速。",
        ]
        abnormal = ["心率增快", "窦性心动过速"]
        direction = "可见于发热、疼痛、焦虑、贫血、甲状腺功能亢进等情况，需要结合症状查原因"
        risk = "medium"
        flags = ["心动过速"]
    elif pattern == 1:
        rows = [
            "节律: 窦性心律。",
            "前壁相关导联可见缺血样改变，伴轻度压低样表现。",
            "结论: 心肌缺血样改变，请结合症状及心肌损伤标志物。",
        ]
        abnormal = ["心肌缺血样改变"]
        direction = "提示心肌供血不足方向，若有胸痛、胸闷、出汗需及时急诊排查"
        risk = "high"
        flags = ["心肌缺血样改变"]
    elif pattern == 2:
        rows = [
            "心率: 每分钟五十二次（参考: 每分钟六十至一百次）",
            "节律: 窦性心律。",
            "结论: 窦性心动过缓。",
        ]
        abnormal = ["心率偏慢", "窦性心动过缓"]
        direction = "可见于运动员、睡眠状态、药物影响，也可能与传导系统问题有关"
        risk = "medium"
        flags = ["心动过缓"]
    else:
        rows = [
            "心率: 每分钟七十六次（参考: 每分钟六十至一百次）。",
            "节律: 窦性心律。",
            "结论: 大致正常心电图。",
        ]
        abnormal = []
        direction = "本次心电图描述大致正常"
        risk = "low"
        flags = []
    return pack("心电图", rows, abnormal, direction, risk, flags)


def make_tumor_marker(idx: int, rng: random.Random) -> dict[str, Any]:
    pattern = idx % 4
    if pattern == 0:
        rows = [
            line("癌胚抗原", value(rng, 8, 16), "0-5", " 纳克每毫升", True),
            line("糖类抗原一九九", value(rng, 18, 32), "0-37", " 单位每毫升"),
            line("甲胎蛋白", value(rng, 2, 7), "0-10", " 纳克每毫升"),
        ]
        abnormal = ["癌胚抗原升高"]
        direction = "癌胚抗原轻中度升高没有特异性，可受吸烟、炎症、肠道疾病等影响，也需结合肿瘤筛查"
        risk = "medium"
        flags = ["肿瘤标志物升高"]
    elif pattern == 1:
        rows = [
            line("甲胎蛋白", value(rng, 120, 680, 0), "0-10", " 纳克每毫升", True),
            line("癌胚抗原", value(rng, 2, 5), "0-5", " 纳克每毫升"),
            line("糖类抗原一九九", value(rng, 15, 30), "0-37", " 单位每毫升"),
        ]
        abnormal = ["甲胎蛋白明显升高"]
        direction = "需警惕肝脏相关疾病方向，包括活动性肝病和肝脏肿瘤风险，必须结合肝脏影像和肝炎指标"
        risk = "high"
        flags = ["甲胎蛋白明显升高"]
    elif pattern == 2:
        rows = [
            line("糖类抗原一九九", value(rng, 95, 360, 0), "0-37", " 单位每毫升", True),
            line("癌胚抗原", value(rng, 4, 9), "0-5", " 纳克每毫升", True),
            line("甲胎蛋白", value(rng, 2, 8), "0-10", " 纳克每毫升"),
        ]
        abnormal = ["糖类抗原一九九明显升高", "癌胚抗原轻度升高"]
        direction = "可见于胆胰系统炎症、梗阻或肿瘤等方向，需要结合腹部影像和症状"
        risk = "high"
        flags = ["糖类抗原一九九明显升高"]
    else:
        rows = [
            line("癌胚抗原", value(rng, 1, 4), "0-5", " 纳克每毫升"),
            line("甲胎蛋白", value(rng, 2, 7), "0-10", " 纳克每毫升"),
            line("糖类抗原一九九", value(rng, 8, 28), "0-37", " 单位每毫升"),
        ]
        abnormal = []
        direction = "本次列出的肿瘤标志物在参考范围内，但不能单独排除肿瘤"
        risk = "low"
        flags = []
    return pack("肿瘤标志物", rows, abnormal, direction, risk, flags)


MAKERS = {
    "血常规": make_blood,
    "肝功能": make_liver,
    "肾功能": make_kidney,
    "甲状腺功能": make_thyroid,
    "血脂": make_lipid,
    "尿常规": make_urine,
    "腹部超声": make_ultrasound,
    "胸部计算机断层扫描": make_chest_scan,
    "心电图": make_ecg,
    "肿瘤标志物": make_tumor_marker,
}


def pack(report_type: str, rows: list[str], abnormal: list[str], direction: str, risk: str, flags: list[str]) -> dict[str, Any]:
    title = f"{report_type}报告"
    report_text = "\n".join([f"【{title}】", "标本或检查日期: 今日", *rows])
    return {
        "report_type": report_type,
        "department": DEPARTMENT_BY_TYPE[report_type],
        "report_text": report_text,
        "abnormal_items": abnormal,
        "direction": direction,
        "risk_level": risk,
        "red_flags": flags,
    }


def make_report(seq: int) -> dict[str, Any]:
    rng = random.Random(20260707 + seq * 17)
    report_type = REPORT_TYPES[(seq - 1) % len(REPORT_TYPES)]
    maker = MAKERS[report_type]
    variant = (seq - 1) // len(REPORT_TYPES)
    return maker(variant, rng)


def make_report_by_type(report_type: str, variant: int) -> dict[str, Any]:
    rng = random.Random(20260707 + (REPORT_TYPES.index(report_type) + 1) * 1009 + variant * 37)
    return MAKERS[report_type](variant, rng)


def user_content(spec: dict[str, Any]) -> str:
    return (
        "以下是我的检查报告，帮我看看这个报告：\n\n"
        f"{spec['report_text']}\n\n"
        "我主要想知道哪些地方异常、可能提示什么、接下来应该去哪个科室复诊。"
    )


def prompt_for_assistant(spec: dict[str, Any], attempt: int) -> list[dict[str, str]]:
    abnormal = "、".join(spec["abnormal_items"]) if spec["abnormal_items"] else "本报告未见明确异常项目"
    prompt = f"""
请根据下面这份仿真检查报告，写一段中文医疗助手回复给患者。只输出 assistant 正文，不要标题、不要列表外壳、不要代码块。

检查报告:
{spec['report_text']}

规则引擎给出的异常项目:
{abnormal}

规则引擎给出的提示方向:
{spec['direction']}

建议复诊科室:
{spec['department']}

硬性要求:
1. 必须逐项解释异常指标的含义；如果没有明显异常，要说明主要项目在参考范围内。
2. 必须使用“可能、提示、建议结合临床”等谨慎措辞，严禁下确诊结论。
3. 必须明确建议复诊科室和随访动作，例如复查、补充检查、带原始报告和片子就诊。
4. 结尾必须包含“不能替代医生面诊”或同义表述。
5. 全中文，不要出现英文缩写、英文字母、真实医院名、真实电话号码。
6. 长度 120-420 字。第 {attempt + 1} 次生成请换一种自然表达。
""".strip()
    return [
        {"role": "system", "content": "你是一名谨慎的中文医学报告解读助手，只能依据给定报告和规则引擎提示解释，不能诊断。"},
        {"role": "user", "content": prompt},
    ]


def validate_assistant(text: str, spec: dict[str, Any]) -> None:
    if not text:
        raise ValueError("empty_assistant")
    if len(text) < 80:
        raise ValueError("assistant_too_short")
    if len(text) > 600:
        raise ValueError("assistant_too_long")
    if BANNED.search(text):
        raise ValueError("banned_text")
    if HIGH_SYMBOL.search(text):
        raise ValueError("emoji_or_high_symbol")
    if ASCII_LETTER.search(text):
        raise ValueError("ascii_letter")
    if REAL_HOSPITAL.search(text):
        raise ValueError("real_hospital")
    if PHONE.search(text):
        raise ValueError("phone_number")
    if not any(word in text for word in ("可能", "提示", "建议结合", "倾向", "方向")):
        raise ValueError("missing_uncertainty")
    if any(word in text for word in ("确诊为", "就是癌", "肯定是", "一定是")):
        raise ValueError("diagnostic_overclaim")
    if not any(word in text for word in ("复诊", "就诊", "门诊", "医生")):
        raise ValueError("missing_followup")
    if not any(word in text for word in ("不能替代医生面诊", "不替代医生面诊", "不能代替医生面诊", "不能代替面诊")):
        raise ValueError("missing_face_to_face_disclaimer")
    for item in spec["abnormal_items"][:4]:
        key = item[:2]
        if key and key not in text:
            raise ValueError(f"missing_abnormal_item_{key}")


def parse_existing() -> list[dict[str, Any]]:
    if not OUT_PATH.exists():
        return []
    rows = []
    with OUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=id_number)
    return rows


def id_number(row: dict[str, Any]) -> int:
    match = re.search(r"_(\d{8})$", str(row.get("id", "")))
    return int(match.group(1)) if match else 10**12


def next_seq(rows: list[dict[str, Any]]) -> int:
    max_seen = 0
    for row in rows:
        match = re.search(r"_(\d{8})$", row.get("id", ""))
        if match:
            max_seen = max(max_seen, int(match.group(1)))
    return max_seen + 1


def build_record(seq: int, spec: dict[str, Any], assistant: str) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": user_content(spec)},
        {"role": "assistant", "content": assistant},
    ]
    meta = {
        "department": spec["department"],
        "risk_level": spec["risk_level"],
        "red_flags": spec["red_flags"],
        "evidence_required": False,
        "is_multiturn": False,
        "language": "zh",
        "source_quality": "high",
        "license": "internal",
        "dedup_hash": stable_hash(messages),
        "report_type": spec["report_type"],
        "abnormal_items": spec["abnormal_items"],
    }
    return {
        "id": f"gen_{TASK_B}_{seq:08d}",
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "task_type": TASK_B,
        "sub_task_type": None,
        "messages": messages,
        "metadata": meta,
    }


def generate_one(
    seq: int,
    meter: TokenMeter,
    attempts: int,
    report_type: str | None = None,
    variant: int = 0,
) -> tuple[dict[str, Any] | None, str | None]:
    spec = make_report_by_type(report_type, variant) if report_type else make_report(seq)
    for attempt in range(attempts):
        try:
            text = call_llm(
                prompt_for_assistant(spec, attempt),
                temperature=0.55,
                max_tokens=4096,
                retries=3,
                meter=meter,
            )
            text = strip_think(text).strip()
            text = re.sub(r"^```.*?\n|\n?```$", "", text, flags=re.S).strip()
            validate_assistant(text, spec)
            return build_record(seq, spec, text), None
        except Exception as exc:  # noqa: BLE001
            last = str(exc) or type(exc).__name__
            time.sleep(0.4 * (attempt + 1))
    return None, last


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


def compute_stats(rows: list[dict[str, Any]], discards: collections.Counter[str], meter: TokenMeter) -> dict[str, Any]:
    lengths = []
    report_types = collections.Counter()
    risks = collections.Counter()
    departments = collections.Counter()
    for row in rows:
        meta = row["metadata"]
        report_types[meta.get("report_type", "unknown")] += 1
        risks[meta.get("risk_level", "unknown")] += 1
        departments[meta.get("department", "unknown")] += 1
        assistants = [m["content"] for m in row["messages"] if m["role"] == "assistant"]
        if assistants:
            lengths.append(len(assistants[-1]))
    token_usage = meter.as_dict()
    if token_usage.get("api_calls", 0) == 0 and rows:
        chars = 0
        for row in rows:
            for msg in row.get("messages", []):
                if msg.get("role") != "system":
                    chars += len(msg.get("content", ""))
        token_usage["rough_existing_rows_tokens"] = max(1, int(chars / 1.8))
    return {
        "count": len(rows),
        "assistant_length": {
            "min": min(lengths) if lengths else None,
            "p25": percentile(lengths, 0.25),
            "p50": percentile(lengths, 0.50),
            "p75": percentile(lengths, 0.75),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths) if lengths else None,
            "mean": round(statistics.mean(lengths), 2) if lengths else None,
        },
        "report_type_distribution": dict(report_types),
        "risk_level_distribution": dict(risks),
        "department_distribution": dict(departments),
        "discard_reasons": dict(discards),
        "token_usage": token_usage,
        "updated_at": now_iso(),
    }


def validate_record(row: dict[str, Any]) -> tuple[bool, str]:
    try:
        if row.get("schema_version") != SCHEMA_VERSION or row.get("source") != SOURCE or row.get("task_type") != TASK_B:
            return False, "bad_top_level"
        messages = row.get("messages")
        if not isinstance(messages, list) or [m.get("role") for m in messages] != ["system", "user", "assistant"]:
            return False, "bad_messages"
        meta = row.get("metadata") or {}
        for key in ["department", "risk_level", "red_flags", "evidence_required", "is_multiturn", "language", "source_quality", "license", "dedup_hash"]:
            if key not in meta:
                return False, f"missing_{key}"
        if meta["dedup_hash"] != stable_hash(messages):
            return False, "bad_dedup_hash"
        joined = "\n".join(m.get("content", "") for m in messages)
        if BANNED.search(joined) or HIGH_SYMBOL.search(joined) or ASCII_LETTER.search(joined) or REAL_HOSPITAL.search(joined):
            return False, "bad_text"
        validate_assistant(messages[-1]["content"], meta)
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def desired_report_counts(target: int) -> dict[str, int]:
    base = target // len(REPORT_TYPES)
    extra = target % len(REPORT_TYPES)
    return {name: base + (1 if i < extra else 0) for i, name in enumerate(REPORT_TYPES)}


def compact_ids(rows: list[dict[str, Any]]) -> None:
    rows.sort(key=lambda row: (REPORT_TYPES.index((row.get("metadata") or {}).get("report_type", REPORT_TYPES[0])), id_number(row)))
    for idx, row in enumerate(rows, 1):
        row["id"] = f"gen_{TASK_B}_{idx:08d}"


def make_report_md(rows: list[dict[str, Any]], stats: dict[str, Any], target: int) -> str:
    schema = collections.Counter()
    for row in rows:
        ok, reason = validate_record(row)
        schema["ok" if ok else reason] += 1
    legal = schema["ok"]
    rng = random.Random(20260707)
    picks = rng.sample(rows, min(10, len(rows))) if rows else []
    lines = [
        "# test_report_explanation 试产自检报告",
        "",
        f"- 目标条数: {target}",
        f"- 当前文件条数: {len(rows)}",
        f"- 文件格式合法率: {legal}/{len(rows)} = {legal / max(1, len(rows)):.2%}",
        f"- 报告类型分布: {json.dumps(stats['report_type_distribution'], ensure_ascii=False)}",
        f"- 风险等级分布: {json.dumps(stats['risk_level_distribution'], ensure_ascii=False)}",
        f"- 科室覆盖: {len(stats['department_distribution'])} 种",
        f"- assistant 长度分位数: {json.dumps(stats['assistant_length'], ensure_ascii=False)}",
        f"- 丢弃原因计数: {json.dumps(stats['discard_reasons'], ensure_ascii=False)}",
        f"- token 估算/记录: {json.dumps(stats['token_usage'], ensure_ascii=False)}",
        "- 自检结论: B 试产完成后停下，等待人工确认后才量产。",
        "",
        "## 敏感问题抽查 10 条",
    ]
    for row in picks:
        meta = row["metadata"]
        lines.append("")
        lines.append(f"### {row['id']} | {meta.get('report_type')} | {meta.get('risk_level')} | {meta.get('department')}")
        user = row["messages"][1]["content"].replace("\n", " / ")
        assistant = row["messages"][2]["content"].replace("\n", " ")
        lines.append(f"- 用户: {user[:700]}")
        lines.append(f"- 助手: {assistant}")
    return "\n".join(lines) + "\n"


def generate_b(args: argparse.Namespace) -> None:
    if args.workers > 8:
        raise SystemExit("workers must be <= 8")
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        stamp = str(int(time.time()))
        for path in (OUT_PATH, STATS_PATH, REPORT_PATH):
            if path.exists():
                backup = path.with_name(f"{path.stem}.bak_{stamp}{path.suffix}")
                path.replace(backup)
    rows = parse_existing()
    existing_hashes = {(row.get("metadata") or {}).get("dedup_hash") for row in rows}
    seq = next_seq(rows)
    target = args.target
    start_count = len(rows)
    discards: collections.Counter[str] = collections.Counter()
    meter = TokenMeter()

    desired = desired_report_counts(target)
    while len(rows) < target:
        current = collections.Counter((row.get("metadata") or {}).get("report_type") for row in rows)
        remaining = target - len(rows)
        batch = min(args.workers * 2, remaining)
        jobs: list[tuple[int, str, int]] = []
        planned = collections.Counter()
        for _ in range(batch):
            deficits = [
                name for name in REPORT_TYPES
                if current[name] + planned[name] < desired[name]
            ]
            if not deficits:
                break
            report_type = max(deficits, key=lambda name: desired[name] - current[name] - planned[name])
            variant = current[report_type] + planned[report_type]
            jobs.append((seq, report_type, variant))
            planned[report_type] += 1
            seq += 1
        if not jobs:
            break
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futs = {
                executor.submit(generate_one, s, meter, args.per_item_attempts, report_type, variant): (s, report_type)
                for s, report_type, variant in jobs
            }
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
                with OUT_PATH.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                rows.append(row)
                existing_hashes.add(dh)
                print(f"OK {len(rows)}/{target} id={row['id']} type={row['metadata']['report_type']}", flush=True)
                if len(rows) >= target:
                    break

    compact_ids(rows)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    stats = compute_stats(rows, discards, meter)
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    report = make_report_md(rows, stats, target)
    REPORT_PATH.write_text(report, encoding="utf-8")
    append_log(
        "任务 B test_report_explanation 试产自检",
        [
            "- 阶段: B 试产 100",
            f"- 操作: 规则引擎生成可控检查报告数值，MiniMax-M3 只生成解释文本；从 {start_count} 条断点续跑至 {len(rows)} 条。",
            f"- 产出: {OUT_PATH} ({len(rows)} 条)",
            f"- 统计: {STATS_PATH}",
            f"- 自检报告: {REPORT_PATH}",
            "- 自检摘要如下:",
            "",
            report.rstrip(),
            "",
            "- 状态: 已按要求停下，等待人工确认后再量产 B。",
        ],
    )
    print(f"STATS {STATS_PATH}")
    print(f"REPORT {REPORT_PATH}")
    print("B_TRIAL_COMPLETE_WAIT_FOR_CONFIRMATION")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--per-item-attempts", type=int, default=4)
    parser.add_argument("--fresh", action="store_true", help="Backup existing B outputs and regenerate from scratch.")
    args = parser.parse_args()
    generate_b(args)


if __name__ == "__main__":
    main()
