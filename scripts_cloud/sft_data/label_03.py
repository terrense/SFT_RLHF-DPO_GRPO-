#!/usr/bin/env python3
"""03_labeled: 规则分类精修 task_type + 打 risk/department 标签。
挖掘目标(从开源池):triage_guidance / chronic_disease_management / medication_guidance_safe。
其余保持来源先验(symptom_consultation / health_encyclopedia_qa)。
种子集透传。输出 03_labeled/*.jsonl + task_type_distribution.csv 等报告。
"""
import json, os, re, csv, collections

ROOT = "/data/shenxin/rlhf_lab/data/sft_pipeline"
IN_DIR, OUT_DIR, REP = f"{ROOT}/02_cleaned", f"{ROOT}/03_labeled", f"{ROOT}/reports"
os.makedirs(OUT_DIR, exist_ok=True)

# --- triage:问句里问"挂哪个科/看什么科/去哪个科室" ---
TRIAGE_Q = re.compile(r"(挂(什么|哪个|哪)科|看(什么|哪个|哪)科|(去|该)(哪个|什么)科室|"
                      r"属于(什么|哪个)科|应该(去|看|挂).{0,4}(科|门诊)|哪个门诊)")
# --- 慢病管理:慢病名 + 管理语境(非急性发作问诊) ---
CHRONIC_D = re.compile(r"(高血压|糖尿病|高血脂|高脂血症|冠心病|慢性胃炎|哮喘|慢阻肺|COPD|"
                       r"过敏性鼻炎|慢性鼻炎|脂肪肝|痛风|甲减|甲亢|骨质疏松|慢性肾)")
CHRONIC_CTX = re.compile(r"(长期|控制|管理|饮食|运动|监测|随访|复查|注意(什么|事项|哪些)|"
                         r"日常|保养|调理|生活(方式|习惯)|平时)")
# --- 用药指导:用药问句 + 回答含警示语(安全前提) ---
MED_Q = re.compile(r"(吃什么药|用什么药|(这|该)药(怎么|如何)(吃|用|服)|服用方法|用法用量|"
                   r"副作用|不良反应|禁忌|忌口|(能|可以)一起(吃|服)|相互作用|(孕妇|哺乳|儿童|老人)(能|可以)吃)")
CAUTION = re.compile(r"(遵医嘱|医生指导|医师指导|就医|就诊|咨询医生|面诊|不能替代|仅供参考|说明书|药师)")
# --- 高危红旗(打 risk_level 用) ---
REDFLAG = re.compile(r"(剧烈?胸痛|压榨性|呼吸困难|喘不上气|意识(不清|丧失|模糊)|昏迷|抽搐|"
                     r"大出血|呕血|咯血|黑便|自杀|轻生|不想活|中毒|服毒|过量服|烈性|"
                     r"突然(晕倒|倒地|口齿不清)|偏瘫|口眼歪斜|胎动(消失|减少)|剧烈腹痛)")

RISK_HIGH_DEPT = re.compile(r"急诊")

def classify(d):
    msgs = [m for m in d["messages"] if m["role"] != "system"]
    q = "".join(m["content"] for m in msgs if m["role"] == "user")
    a = "".join(m["content"] for m in msgs if m["role"] == "assistant")
    tt = d["task_type"]
    if d["source"] == "internal_seed_flywheel":
        pass  # 种子透传
    elif TRIAGE_Q.search(q):
        tt = "triage_guidance"
    elif MED_Q.search(q) and CAUTION.search(a):
        tt = "medication_guidance_safe"
    elif CHRONIC_D.search(q) and CHRONIC_CTX.search(q + a[:200]) and not d["metadata"]["is_multiturn"]:
        tt = "chronic_disease_management"
    d["task_type"] = tt
    # risk_level
    rf = REDFLAG.findall(q)
    if rf:
        d["metadata"]["risk_level"] = "high"
        d["metadata"]["red_flags"] = sorted(set(x if isinstance(x, str) else x for x in
                                                [m for m in rf if isinstance(m, str)]))[:5]
    elif d["metadata"]["risk_level"] is None:
        d["metadata"]["risk_level"] = "medium" if tt in (
            "symptom_consultation", "triage_guidance") else "low"
    return d

tt_dist = collections.Counter()
tt_by_src = collections.defaultdict(collections.Counter)
risk_dist = collections.Counter()
dept_dist = collections.Counter()

for fname in sorted(os.listdir(IN_DIR)):
    if not fname.endswith(".jsonl"):
        continue
    with open(f"{IN_DIR}/{fname}", encoding="utf-8") as f, \
         open(f"{OUT_DIR}/{fname}", "w", encoding="utf-8") as w:
        for line in f:
            d = classify(json.loads(line))
            tt_dist[d["task_type"]] += 1
            tt_by_src[d["source"]][d["task_type"]] += 1
            risk_dist[d["metadata"]["risk_level"]] += 1
            dept_dist[d["metadata"]["department"]] += 1
            w.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"done {fname}", flush=True)

def wcsv(name, counter):
    with open(f"{REP}/{name}", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "count"])
        for k, c in counter.most_common():
            w.writerow([k, c])

wcsv("task_type_distribution.csv", tt_dist)
wcsv("risk_level_distribution.csv", risk_dist)
wcsv("department_distribution.csv", dept_dist)
with open(f"{REP}/task_type_by_source.json", "w", encoding="utf-8") as f:
    json.dump({s: dict(c.most_common()) for s, c in tt_by_src.items()},
              f, ensure_ascii=False, indent=2)
print(json.dumps({"task_type": dict(tt_dist.most_common()),
                  "risk": dict(risk_dist.most_common())}, ensure_ascii=False, indent=2))
