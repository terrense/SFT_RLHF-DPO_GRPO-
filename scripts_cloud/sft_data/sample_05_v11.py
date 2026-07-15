#!/usr/bin/env python3
"""05: 配比采样(以种子 15,463 为锚,×0.5154 缩放)+ MinHash 近重去重(贪心)
   + 种子派生(conversation_summary / triage 补缺 / risk_redflag 挖掘)
   + 按 task_type 分层 95/2.5/2.5 切分 + hard_eval + LLaMA-Factory 导出。
确定性:一切排序/切分用 md5(id),无随机状态。
"""
import json, os, re, csv, hashlib, collections

ROOT = "/data/shenxin/rlhf_lab/data/sft_pipeline"
IN_DIR, REP = f"{ROOT}/04_deduped", f"{ROOT}/reports"
OUT = f"{ROOT}/05_final_v11"
os.makedirs(OUT, exist_ok=True)

TARGETS = {   # 300k 模板 × (15463/30000)
    "pre_consultation_multiturn": 15463,
    "symptom_consultation": 36080,
    "health_encyclopedia_qa": 28349,
    "triage_guidance": 18040,
    "test_report_explanation": 15463,      # 生成待办,本版=挖掘所得
    "chronic_disease_management": 12886,
    "medication_guidance_safe": 7732,
    "risk_redflag_safety_refusal": 10309,  # 本版=急诊升级挖掘,拒绝类待生成
    "hospital_policy_rag_qa": 5154,        # 生成待办,本版=0
    "conversation_summary_structured_output": 5154,  # 种子反向构造
}
SRC_PRIOR = {"internal_seed_flywheel": 0, "med_zh_real": 1, "DISC-Med-SFT": 2,
             "Chinese-medical-dialogue": 3, "Huatuo26M-Lite": 2,
             "shibing624-finetune-zh": 4,
             "gen_minimax_m3": 0}

def h32(s):
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)

def text_of(d):
    return "".join(m["content"] for m in d["messages"] if m["role"] != "system")

# ---------- MinHash(5字shingle, 32perm, LSH 8band×4) ----------
SALTS = [f"s{i}" for i in range(32)]
def minhash(text):
    t = re.sub(r"\s+", "", text)[:1500]
    if len(t) < 5:
        return None
    sh = {t[i:i+5] for i in range(len(t) - 4)}
    return tuple(min(h32(s + g) for g in sh) for s in SALTS)

class LSH:
    def __init__(self):
        self.bands = [collections.defaultdict(list) for _ in range(8)]
    def probe_add(self, sig, uid):
        """返回是否与已有近重;不近重则加入。"""
        if sig is None:
            return False
        keys = [tuple(sig[b*4:(b+1)*4]) for b in range(8)]
        cands = set()
        for b, k in enumerate(keys):
            cands.update(self.bands[b][k])
        for csig in cands:
            same = sum(1 for x, y in zip(sig, csig) if x == y)
            if same / 32 >= 0.72:   # ≈ Jaccard 0.7
                return True
        for b, k in enumerate(keys):
            self.bands[b][k].append(sig)
        return False

# ---------- 质量分(越小越好) ----------
REDFLAG = re.compile(r"(剧烈?胸痛|压榨性|呼吸困难|喘不上气|意识(不清|丧失|模糊)|昏迷|抽搐|"
                     r"大出血|呕血|咯血|黑便|自杀|轻生|不想活|中毒|服毒|过量服|"
                     r"突然(晕倒|倒地|口齿不清)|偏瘫|口眼歪斜|胎动(消失|减少)|剧烈腹痛)")
ESCALATE = re.compile(r"(120|急诊|立即就医|马上就医|尽快(去|到)医院|紧急|立刻(去|到)医院)")
REPORT_Q = re.compile(r"(报告|化验|检查结果|检验单|(B超|CT|核磁|MRI|心电图|彩超)(显示|结果|提示)|"
                      r"(白细胞|血红蛋白|血小板|转氨酶|肌酐|尿酸|血糖|甲状腺|肿瘤标志物).{0,12}(高|低|升高|偏|异常))")

def qscore(d, ans_len):
    s = SRC_PRIOR.get(d["source"], 5) * 100
    s += 0 if 80 <= ans_len <= 1200 else 40   # 长度甜区
    s += 0 if d["metadata"]["department"] != "unknown" else 10
    return s

# ---------- 第一遍:收集候选(id, 排序键),按 task_type 分桶 ----------
buckets = collections.defaultdict(list)
FILES = sorted(os.listdir(IN_DIR))
for fname in FILES:
    if not fname.endswith(".jsonl"):
        continue
    with open(f"{IN_DIR}/{fname}", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            msgs = [m for m in d["messages"] if m["role"] != "system"]
            q = "".join(m["content"] for m in msgs if m["role"] == "user")
            a = "".join(m["content"] for m in msgs if m["role"] == "assistant")
            tt = d["task_type"]
            # 二次挖掘:报告解读 / 急诊升级(覆盖 03 的粗标)
            if d["source"] not in ("internal_seed_flywheel", "gen_minimax_m3"):
                if REPORT_Q.search(q):
                    tt = "test_report_explanation"
                elif d["metadata"]["risk_level"] == "high" and ESCALATE.search(a):
                    tt = "risk_redflag_safety_refusal"
            key = (qscore(d, len(a)), h32(d["id"]))
            buckets[tt].append((key, d["id"], fname))

for tt, lst in sorted(buckets.items()):
    print(f"candidates[{tt}] = {len(lst)}", flush=True)

# 选中 id 集(每类超采 1.6x 供近重去重淘汰)
want = {}
for tt, lst in buckets.items():
    tgt = TARGETS.get(tt, 0)
    lst.sort(key=lambda x: x[0])
    for _, uid, _ in lst[: int(tgt * 1.6)]:
        want[uid] = tt

# ---------- 第二遍:按质量序流式选入 + 近重去重 ----------
lsh = LSH()
picked = collections.defaultdict(list)
near_dup = collections.Counter()
order = sorted(FILES, key=lambda f: SRC_PRIOR.get(f[:-6], 9))
for fname in order:
    if not fname.endswith(".jsonl"):
        continue
    with open(f"{IN_DIR}/{fname}", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            tt = want.get(d["id"])
            if tt is None or len(picked[tt]) >= TARGETS[tt]:
                continue
            if d["source"] not in ("internal_seed_flywheel", "gen_minimax_m3"):
                if lsh.probe_add(minhash(text_of(d)), d["id"]):
                    near_dup[tt] += 1
                    continue
            d["task_type"] = tt
            picked[tt].append(d)

# ---------- 种子派生 ----------
def seed_items():
    return picked["pre_consultation_multiturn"]

# conversation_summary:对话→结构化 JSON(meta 真值)
seeds = sorted(seed_items(), key=lambda d: h32("sum" + d["id"]))
for d in seeds[:TARGETS["conversation_summary_structured_output"]]:
    sm = d.get("seed_meta", {})
    convo = "\n".join(f"{'患者' if m['role']=='user' else '助手'}:{m['content']}"
                      for m in d["messages"] if m["role"] != "system")
    cc = (sm.get("persona") or "").split("主诉:")[-1] or "unknown"
    out_obj = {"chief_complaint": cc, "department": sm.get("target_department") or d["metadata"]["department"],
               "urgency_level": sm.get("triage_level"), "diagnosis_candidates": sm.get("diagnosis") or [],
               "red_flags": [], "needs_emergency": False}
    nd = {"id": "drv_summary_" + d["id"], "schema_version": "1.0", "source": "derived_from_seed",
          "task_type": "conversation_summary_structured_output", "sub_task_type": None,
          "messages": [
              {"role": "system", "content": "你是医疗对话结构化助手。把给定的预问诊对话总结为JSON,字段:chief_complaint, department, urgency_level(1-4), diagnosis_candidates, red_flags, needs_emergency。只输出JSON。"},
              {"role": "user", "content": convo},
              {"role": "assistant", "content": json.dumps(out_obj, ensure_ascii=False)}],
          "metadata": dict(d["metadata"], is_multiturn=False, dedup_hash="drvsum" + d["metadata"]["dedup_hash"])}
    picked["conversation_summary_structured_output"].append(nd)

# triage 补缺:症状陈述→科室推荐(3种措辞轮换防模板化)
TPL = ["根据您描述的情况,建议挂{d}。{u}",
       "您的症状比较符合{d}的诊疗范围,建议前往{d}就诊。{u}",
       "建议您到{d}做进一步检查。{u}"]
URG = {2: "情况需要重视,建议尽快就诊。", 3: "建议近期安排就诊。", 4: "可正常门诊时间就诊。"}
gap = TARGETS["triage_guidance"] - len(picked["triage_guidance"])
seeds2 = sorted(seed_items(), key=lambda d: h32("tri" + d["id"]))
made = 0
for d in seeds2:
    if made >= gap:
        break
    users = [m["content"] for m in d["messages"] if m["role"] == "user"]
    dept = d["metadata"]["department"]
    tl = (d.get("seed_meta") or {}).get("triage_level") or 4
    if not users or dept == "unknown":
        continue
    stmt = "医生您好,我的情况是:" + ";".join(u for u in users if len(u) > 1)[:600] +"。我应该挂哪个科?"
    ans = TPL[h32("t" + d["id"]) % 3].format(d=dept, u=URG.get(tl, ""))
    nd = {"id": "drv_triage_" + d["id"], "schema_version": "1.0", "source": "derived_from_seed",
          "task_type": "triage_guidance", "sub_task_type": "recommendation_generation",
          "messages": [{"role": "user", "content": stmt}, {"role": "assistant", "content": ans}],
          "metadata": dict(d["metadata"], is_multiturn=False, dedup_hash="drvtri" + d["metadata"]["dedup_hash"])}
    picked["triage_guidance"].append(nd)
    made += 1

# ---------- 切分:md5(id) 分层 95/2.5/2.5;hard_eval 从 test 抽 ----------
splits = {"train": [], "dev": [], "test": []}
for tt, lst in picked.items():
    for d in lst:
        r = h32("split" + d["id"]) % 1000
        splits["train" if r < 950 else "dev" if r < 975 else "test"].append(d)

HARD_TT = {"risk_redflag_safety_refusal", "triage_guidance", "test_report_explanation",
           "pre_consultation_multiturn", "medication_guidance_safe", "hospital_policy_rag_qa"}
hard = [d for d in splits["test"]
        if d["task_type"] in HARD_TT or d["metadata"]["risk_level"] == "high"]

stats = {"total": sum(len(v) for v in picked.values()),
         "by_task_type": {tt: len(v) for tt, v in sorted(picked.items())},
         "targets": TARGETS,
         "near_dup_removed": dict(near_dup),
         "splits": {k: len(v) for k, v in splits.items()}, "hard_eval": len(hard),
         "by_source": dict(collections.Counter(d["source"] for v in picked.values() for d in v)),
         "risk": dict(collections.Counter(d["metadata"]["risk_level"] for v in picked.values() for d in v))}

# 长度分布
lens = collections.defaultdict(list)
for v in picked.values():
    for d in v:
        lens[d["task_type"]].append(len(text_of(d)))
def pct(a, p):
    a = sorted(a); return a[min(len(a)-1, int(len(a)*p/100))] if a else 0
with open(f"{REP}/length_distribution.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f); w.writerow(["task_type", "n", "p50", "p90", "p99"])
    for tt, a in sorted(lens.items()):
        w.writerow([tt, len(a), pct(a, 50), pct(a, 90), pct(a, 99)])

with open(f"{OUT}/05_sampled_balanced.jsonl", "w", encoding="utf-8") as f:
    for tt in sorted(picked):
        for d in picked[tt]:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
for name, lst in list(splits.items()) + [("hard_eval", hard)]:
    with open(f"{OUT}/{name}.jsonl", "w", encoding="utf-8") as f:
        for d in sorted(lst, key=lambda x: h32("o" + x["id"])):
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

# LLaMA-Factory sharegpt 导出(train)
with open(f"{OUT}/train_sharegpt.json", "w", encoding="utf-8") as f:
    out = []
    for d in splits["train"]:
        conv = [{"from": {"user": "human", "assistant": "gpt"}[m["role"]], "value": m["content"]}
                for m in d["messages"] if m["role"] != "system"]
        sysm = next((m["content"] for m in d["messages"] if m["role"] == "system"), "")
        out.append({"conversations": conv, "system": sysm})
    json.dump(out, f, ensure_ascii=False)

with open(f"{REP}/final_dataset_stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print(json.dumps(stats, ensure_ascii=False, indent=2))
