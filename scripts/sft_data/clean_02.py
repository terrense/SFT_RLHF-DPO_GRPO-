#!/usr/bin/env python3
"""02_cleaned: 规则过滤 + 轻规范化。
- 种子集(internal_seed_flywheel)只标记 flags 不删除(业务种子不稀释)。
- 开源池硬过滤,所有删除记 removed_samples_reason.csv(原因计数+抽样示例)。
- 不改写内容(风格规范化留给后续 teacher pass),只做空白修剪。
规则:超短/超长、广告SEO营销、联系方式PII、无警示开药、无检查下定论、危险建议。
"""
import json, os, re, csv, collections, random

ROOT = "/data/shenxin/rlhf_lab/data/sft_pipeline"
IN_DIR, OUT_DIR = f"{ROOT}/01_raw_converted", f"{ROOT}/02_cleaned"
REP = f"{ROOT}/reports"
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(42)

# --- 广告/SEO/营销(问诊网站爬取数据的典型污染) ---
AD = re.compile(r"(点击(在线)?咨询|在线(专家|医生)?咨询|免费咨询|咨询热线|拨打(我院)?电话|"
                r"来(我|本)院(就诊|治疗|检查)|我院(专家|开展|引进|采用|拥有)|本院(专家|开展)|"
                r"(专家|咨询)QQ|加(我|微信)|微信号|公众号|扫码|挂号费|优惠活动|活动期间|"
                r"祝您?(早日)?康复.{0,6}(欢迎|点击)|healthcare\.com|www\.[a-z0-9]+\.(com|cn|net))")
# --- 联系方式 / PII ---
PII = re.compile(r"(1[3-9]\d{9}|(?:0\d{2,3}-)?\d{7,8}(?!\d)|[Qq]{2}[::]?\s*\d{6,11}|"
                 r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})")
# --- 具体剂量(开药) ---
DOSE = re.compile(r"(每[日天次][口服]?\d+(\.\d+)?\s*(mg|毫克|g|克|片|粒|袋|支|ml|毫升)|"
                  r"\d+(\.\d+)?\s*(mg|毫克|ml|毫升)\s*(bid|tid|qd|qid|每日|一日|每天)|"
                  r"(一|每)[日天]\s*[123一二三]\s*次.{0,8}(每次)?\s*\d+(\.\d+)?\s*(mg|毫克|g|克|片|粒))")
CAUTION = re.compile(r"(遵医嘱|医生指导|医师指导|就医|就诊|咨询医生|面诊|不能替代|仅供参考|专业医"
                     r"|说明书|药师)")
# --- 无检查下定论(assistant 口吻) ---
DEFINITIVE = re.compile(r"(?<![不没未必])(肯定|一定|绝对|百分之百|无疑)(是|得的?是|患的?是)")
HEDGE = re.compile(r"(可能|考虑|怀疑|倾向|建议.{0,10}(检查|就诊|就医)|需要.{0,6}(检查|确诊))")
# --- 危险建议(保守小黑名单) ---
DANGER = re.compile(r"(自行(加|减)量|擅自(停药|加量)没关系|不用去医院.{0,10}(没事|放心)|"
                    r"(大剂量|加倍)(服用|使用)|偏方(治愈|根治)|包治|根治率100)")
# 中医药经典描述误伤保护:含"辨证/中医认为"的不判危险
TCM_GUARD = re.compile(r"(辨证|中医认为|方剂)")

MIN_ANSWER, MAX_TOTAL = 12, 8000

def check(sample):
    """返回 flags 列表(命中的规则)。"""
    flags = []
    msgs = [m for m in sample["messages"] if m["role"] != "system"]
    ans_all = "".join(m["content"] for m in msgs if m["role"] == "assistant")
    final_ans = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant"), "")
    total = sum(len(m["content"]) for m in msgs)
    if len(final_ans.strip()) < MIN_ANSWER:
        flags.append("too_short_answer")
    if total > MAX_TOTAL:
        flags.append("too_long_total")
    if AD.search(ans_all):
        flags.append("ad_seo_marketing")
    if PII.search(ans_all):
        flags.append("pii_contact")
    if DOSE.search(ans_all) and not CAUTION.search(ans_all):
        flags.append("dose_without_caution")
    if DEFINITIVE.search(ans_all) and not HEDGE.search(ans_all):
        flags.append("definitive_diagnosis")
    if DANGER.search(ans_all) and not TCM_GUARD.search(ans_all):
        flags.append("dangerous_advice")
    return flags

reason_counter = collections.defaultdict(collections.Counter)
reason_examples = collections.defaultdict(list)
stats = collections.OrderedDict()

for fname in sorted(os.listdir(IN_DIR)):
    if not fname.endswith(".jsonl"):
        continue
    src = fname[:-6]
    protect = (src == "internal_seed_flywheel")
    n_in, n_out, n_drop, n_flag = 0, 0, 0, 0
    with open(f"{IN_DIR}/{fname}", encoding="utf-8") as f, \
         open(f"{OUT_DIR}/{fname}", "w", encoding="utf-8") as w:
        for line in f:
            n_in += 1
            d = json.loads(line)
            for m in d["messages"]:
                m["content"] = m["content"].strip()
            flags = check(d)
            if flags:
                for fl in flags:
                    reason_counter[src][fl] += 1
                if protect:
                    d["quality_flags"] = flags   # 种子:只标记
                    n_flag += 1
                else:
                    n_drop += 1
                    if len(reason_examples[flags[0]]) < 3 and random.random() < 0.3:
                        ans = next((m["content"] for m in reversed(d["messages"])
                                    if m["role"] == "assistant"), "")
                        reason_examples[flags[0]].append(
                            {"id": d["id"], "src": src, "ans": ans[:150]})
                    continue
            w.write(json.dumps(d, ensure_ascii=False) + "\n")
            n_out += 1
    stats[src] = {"in": n_in, "out": n_out, "dropped": n_drop,
                  "flagged_kept(seed)": n_flag}
    print(f"[{src}] in={n_in} out={n_out} drop={n_drop} flag={n_flag}", flush=True)

with open(f"{REP}/removed_samples_reason.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source", "reason", "count"])
    for src, cnt in reason_counter.items():
        for reason, c in cnt.most_common():
            w.writerow([src, reason, c])

report = {"stats": stats,
          "reasons_by_source": {s: dict(c.most_common()) for s, c in reason_counter.items()},
          "examples": dict(reason_examples)}
with open(f"{REP}/cleaning_stats.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(json.dumps({"stats": stats}, ensure_ascii=False, indent=2))
