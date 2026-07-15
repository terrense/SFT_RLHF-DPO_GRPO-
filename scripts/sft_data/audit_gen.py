#!/usr/bin/env python3
"""独立审计 Codex 生成的三类数据:schema 一致性 + 内容抽样 + 红线规则复检。
不改数据,只读 + 报告。"""
import json, re, collections, random
random.seed(7)

GEN = "/data/shenxin/rlhf_lab/data/sft_pipeline/generated"
FILES = {
    "risk": f"{GEN}/risk_redflag_safety_refusal.jsonl",
    "report": f"{GEN}/test_report_explanation.jsonl",
    "rag": f"{GEN}/hospital_policy_rag_qa.jsonl",
}
CANON_KEYS = {"id", "schema_version", "source", "task_type", "messages", "metadata"}

def load(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def audit_schema(name, rows):
    key_ok = sum(1 for r in rows if CANON_KEYS <= set(r))
    has_msgs = sum(1 for r in rows if isinstance(r.get("messages"), list) and r["messages"])
    end_assistant = sum(1 for r in rows if r.get("messages") and r["messages"][-1].get("role") == "assistant")
    has_sys = sum(1 for r in rows if any(m["role"] == "system" for m in r.get("messages", [])))
    tt = collections.Counter(r.get("task_type") for r in rows)
    ascii_leak = sum(1 for r in rows if re.search(r"[A-Za-z]{4,}",
                     "".join(m["content"] for m in r["messages"] if m["role"] == "assistant")))
    print(f"\n===== [{name}] {len(rows)} 条 =====")
    print(f"canonical键全: {key_ok} | messages非空: {has_msgs} | 末轮assistant: {end_assistant} | 含system: {has_sys}")
    print(f"task_type: {dict(tt)} | assistant含≥4连续英文字母(疑漏译): {ascii_leak}")
    return end_assistant == len(rows)

def show_samples(name, rows, k=2):
    print(f"\n----- [{name}] 随机抽样 {k} 条 -----")
    for r in random.sample(rows, k):
        print(f"[id={r['id']}] risk={r['metadata'].get('risk_level')} dept={r['metadata'].get('department')}")
        for m in r["messages"]:
            c = m["content"].replace("\n", " ")
            print(f"  {m['role'][:4]}: {c[:180]}")
        print()

# ---- 规则复检 ----
ESC = re.compile(r"(120|急诊|立即就医|马上就医|尽快就医|立刻(去|到)医院|拨打)")
def check_risk(rows):
    high = [r for r in rows if r["metadata"].get("risk_level") == "high"]
    no_esc = [r for r in high if not ESC.search("".join(
        m["content"] for m in r["messages"] if m["role"] == "assistant"))]
    print(f"\n[risk复检] high风险 {len(high)} 条,其中末端未见急诊/120升级语: {len(no_esc)}")
    for r in no_esc[:2]:
        print("  疑似:", r["id"], "|", [m["content"][:80] for m in r["messages"] if m["role"]=="user"][:1])

def check_rag(rows):
    # 无答案样本:assistant 应表达"资料未提供/无法回答"
    NOANS = re.compile(r"(没有|未(提及|提供|包含)|无法(回答|确定)|建议(咨询|联系).{0,6}(前台|服务台|人工)|抱歉)")
    noans = [r for r in rows if r["metadata"].get("no_answer") is True
             or "no_answer=True" in json.dumps(r.get("metadata", {}), ensure_ascii=False)]
    # metadata 里可能没直接标,靠 evidence_required + 内容判
    flagged = [r for r in rows if NOANS.search(r["messages"][-1]["content"])]
    print(f"\n[rag复检] 命中'拒答/未提供'措辞的样本: {len(flagged)} (自检称无答案类=1000)")
    ev_req = sum(1 for r in rows if r["metadata"].get("evidence_required"))
    print(f"[rag复检] evidence_required=true: {ev_req}/{len(rows)}")

def check_report(rows):
    # 数值自洽:随机看是否 assistant 里解释了"异常项"且给了随访科室
    FOLLOW = re.compile(r"(复查|随访|就诊|进一步检查|建议.{0,8}(科|门诊|医生))")
    OVER = re.compile(r"(确诊为|一定是|就是癌|百分之百)")
    no_follow = sum(1 for r in rows if not FOLLOW.search(r["messages"][-1]["content"]))
    overclaim = sum(1 for r in rows if OVER.search(r["messages"][-1]["content"]))
    print(f"\n[report复检] 末端无随访建议: {no_follow} | 疑似过度确诊措辞: {overclaim}")

data = {k: load(v) for k, v in FILES.items()}
for name, rows in data.items():
    audit_schema(name, rows)
check_risk(data["risk"])
check_rag(data["rag"])
check_report(data["report"])
show_samples("risk", data["risk"], 3)
show_samples("rag", data["rag"], 3)
show_samples("report", data["report"], 2)
