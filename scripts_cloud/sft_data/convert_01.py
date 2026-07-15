#!/usr/bin/env python3
"""01_raw_converted: 各源 → canonical schema(messages 超集)。
canonical: {id, schema_version, source, task_type(来源先验,03步会精修), sub_task_type,
            messages[], metadata{department, risk_level, red_flags, evidence_required,
            is_multiturn, language, source_quality, license, dedup_hash}}
确定性:采样一律用 md5(id) 取模,seed 无关、可复现。
"""
import json, csv, hashlib, os, re, sys, glob, collections

ROOT = "/data/shenxin/rlhf_lab/data"
MD = f"{ROOT}/medicine_dataset"
OUT = f"{ROOT}/sft_pipeline/01_raw_converted"
os.makedirs(OUT, exist_ok=True)

# ===== 评测保留区(红线):这些数据集禁止进入 SFT 训练管线 =====
# CMB 73.8% / CMExam 82.6% 是本项目基线数字,混入训练则评测作废。
# CMExam 另作 GRPO-RLVR 题库(与评测子集 seed=42 切分不重叠)。
EVAL_RESERVED = ["CMB", "CMExam", "cmexam", "MLEC-QA-Benchmark", "CBLUE"]

WS = re.compile(r"\s+")

def norm_text(s):
    return WS.sub("", s).lower()

def dhash(*parts):
    return hashlib.md5(norm_text("|".join(parts)).encode("utf-8")).hexdigest()

def keep_frac(uid, frac):
    """确定性采样:md5(uid) 取模。"""
    h = int(hashlib.md5(uid.encode("utf-8")).hexdigest()[:8], 16)
    return (h % 10000) < frac * 10000

def rec(rid, source, task_type, messages, dept=None, lang="zh",
        quality="medium", license_="unknown", multiturn=None):
    if multiturn is None:
        multiturn = sum(1 for m in messages if m["role"] == "user") > 1
    text = "|".join(m["content"] for m in messages if m["role"] != "system")
    return {
        "id": rid, "schema_version": "1.0", "source": source,
        "task_type": task_type, "sub_task_type": None,
        "messages": messages,
        "metadata": {
            "department": dept or "unknown", "risk_level": None, "red_flags": [],
            "evidence_required": False, "is_multiturn": multiturn,
            "language": lang, "source_quality": quality, "license": license_,
            "dedup_hash": dhash(text),
        },
    }

def valid_msgs(msgs):
    if not msgs:
        return False
    non_sys = [m for m in msgs if m["role"] != "system"]
    if not non_sys or non_sys[-1]["role"] != "assistant":
        return False
    return all(isinstance(m.get("content"), str) and m["content"].strip() for m in non_sys)

stats = collections.OrderedDict()

def write_source(name, gen):
    path = f"{OUT}/{name}.jsonl"
    n_ok, n_skip = 0, 0
    with open(path, "w", encoding="utf-8") as w:
        for r in gen:
            if r is None or not valid_msgs(r["messages"]):
                n_skip += 1
                continue
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_ok += 1
    stats[name] = {"converted": n_ok, "skipped_invalid": n_skip}
    print(f"[{name}] ok={n_ok} skip={n_skip}", flush=True)

# ---------- 1. 种子集(透传,补齐 metadata 键) ----------
def gen_seed():
    with open(f"{MD}/pre_consultation_multiturn.cleaned.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            meta = d.get("meta", {})
            r = rec(d["id"], "internal_seed_flywheel", "pre_consultation_multiturn",
                    d["messages"], dept=meta.get("target_department"),
                    quality="high", license_="internal", multiturn=True)
            r["metadata"]["risk_level"] = d.get("risk_level")
            r["metadata"]["dedup_hash"] = d.get("dedup_hash") or r["metadata"]["dedup_hash"]
            r["seed_meta"] = {k: meta.get(k) for k in
                              ("scene", "triage_level", "persona", "style", "diagnosis")}
            r["num_turns"] = d.get("num_turns")
            yield r

# ---------- 2. DISC-Med-SFT(多轮 conversation) ----------
def gen_disc():
    with open(f"{MD}/DISC-Med-SFT/DISC-Med-SFT_released.jsonl", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                yield None
                continue
            msgs = [{"role": m["role"], "content": (m.get("content") or "").strip()}
                    for m in d.get("conversation", [])]
            yield rec(f"disc_{d.get('_id')}", "DISC-Med-SFT", "symptom_consultation",
                      msgs, license_="apache-2.0")

# ---------- 3. Chinese-medical-dialogue(GBK CSV,带科室) ----------
def gen_cmd():
    base = f"{MD}/Chinese-medical-dialogue-data/source_zip/Chinese-medical-dialogue-data-master/Data_数据"
    i = 0
    for path in sorted(glob.glob(f"{base}/*/*.csv")):
        with open(path, encoding="gb18030", errors="replace", newline="") as f:
            try:
                reader = csv.DictReader(f)
                for row in reader:
                    i += 1
                    dept = (row.get("department") or "").strip()
                    title = (row.get("title") or "").strip()
                    ask = (row.get("ask") or "").strip()
                    ans = (row.get("answer") or "").strip()
                    if not ans or (not ask and not title) or "�" in (title + ask + ans):
                        yield None
                        continue
                    q = ask if len(ask) >= len(title) else title
                    msgs = [{"role": "user", "content": q},
                            {"role": "assistant", "content": ans}]
                    yield rec(f"cmd_{i:08d}", "Chinese-medical-dialogue",
                              "symptom_consultation", msgs, dept=dept, license_="MIT")
            except Exception as e:
                print(f"[cmd] {path}: {e}", file=sys.stderr)

# ---------- 4. Huatuo26M-Lite(百科,带科室label) ----------
def gen_huatuo():
    with open(f"{MD}/Huatuo26M-Lite/format_data.jsonl", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                yield None
                continue
            q, a = (d.get("question") or "").strip(), (d.get("answer") or "").strip()
            if not q or not a or d.get("score", 5) < 4:
                yield None
                continue
            msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
            yield rec(f"huatuo_{d.get('id')}", "Huatuo26M-Lite", "health_encyclopedia_qa",
                      msgs, dept=(d.get("label") or "unknown"), license_="apache-2.0")

# ---------- 5. shibing624 finetune zh(195万,hash 采样封顶 ~30万) ----------
def gen_shibing():
    i = 0
    with open(f"{MD}/shibing624-medical/finetune/train_zh_0.json", encoding="utf-8") as f:
        for line in f:
            i += 1
            uid = f"shibing_ft_{i:08d}"
            if not keep_frac(uid, 0.154):  # 195万 * 0.154 ≈ 30万
                continue
            try:
                d = json.loads(line)
            except Exception:
                yield None
                continue
            q = ((d.get("instruction") or "") + "\n" + (d.get("input") or "")).strip()
            a = (d.get("output") or "").strip()
            if not q or not a:
                yield None
                continue
            msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
            yield rec(uid, "shibing624-finetune-zh", "health_encyclopedia_qa",
                      msgs, quality="low", license_="apache-2.0")

# ---------- 6. med_zh 真实问诊(已清洗 44.2万) ----------
def gen_medzh():
    i = 0
    with open(f"{ROOT}/med_zh/train_zh_clean.jsonl", encoding="utf-8") as f:
        for line in f:
            i += 1
            try:
                d = json.loads(line)
            except Exception:
                yield None
                continue
            q = ((d.get("instruction") or "") + "\n" + (d.get("input") or "")).strip()
            a = (d.get("output") or "").strip()
            if not q or not a:
                yield None
                continue
            msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
            yield rec(f"medzh_{i:08d}", "med_zh_real", "symptom_consultation",
                      msgs, license_="unknown")

write_source("internal_seed_flywheel", gen_seed())
write_source("DISC-Med-SFT", gen_disc())
write_source("Chinese-medical-dialogue", gen_cmd())
write_source("Huatuo26M-Lite", gen_huatuo())
write_source("shibing624-finetune-zh", gen_shibing())
write_source("med_zh_real", gen_medzh())

stats["_eval_reserved_blacklist"] = EVAL_RESERVED
with open(f"{ROOT}/sft_pipeline/reports/conversion_stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print(json.dumps(stats, ensure_ascii=False, indent=2))
