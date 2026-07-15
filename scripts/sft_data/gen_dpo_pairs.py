#!/usr/bin/env python3
"""DPO 偏好数据生成(R5 SFT 模型出来后跑)。
两阶段:①用 SFT 模型对业务 prompt 每题采样 K 个候选答案(多卡并行,本脚本单卡跑一个分片)
        ②MiniMax-M3 当裁判给候选排序 → 组 chosen/rejected 对。
长度偏置防护:同分时不偏向更长答案;裁判 prompt 明确"简洁准确优先,不因长度加分"。
用法(每卡一个分片): CUDA_VISIBLE_DEVICES=N python gen_dpo_pairs.py --shard N --nshards 4 --stage sample
        然后 --stage judge(调 MiniMax,不占卡)
"""
import json, os, argparse, hashlib, re, time, urllib.request

LAB = "/root/autodl-tmp"
SFT_ADAPTER = f"{LAB}/outputs/r5/sft_v11_final"   # R5 产物
BASE = f"{LAB}/models/Qwen3-8B-Base"
PROMPTS = f"{LAB}/data/sft_v11/train_full.json"    # 从训练集抽业务 prompt
OUT_DIR = f"{LAB}/data/rlhf/dpo"
os.makedirs(OUT_DIR, exist_ok=True)
K = 4                    # 每题候选数
N_PROMPTS = 8000         # 偏好对目标量级
MINIMAX_ENV = f"{LAB}/.minimax_env"   # 需从 5133 取 key
MINIMAX_URL = "https://api.minimaxi.com/v1/chat/completions"

def h32(s): return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)

def load_prompts():
    data = json.load(open(PROMPTS, encoding="utf-8"))
    # 只取单轮或多轮的首个 user 提问作为 DPO prompt(优先业务任务)
    out = []
    for d in data:
        conv = d.get("conversations", [])
        sysm = d.get("system", "")
        firstu = next((c["value"] for c in conv if c["from"] == "human"), None)
        if firstu:
            out.append({"system": sysm, "prompt": firstu,
                        "id": "dpo_%08x" % h32(sysm + firstu)})
    out.sort(key=lambda x: h32(x["id"]))
    # 去重
    seen, uniq = set(), []
    for o in out:
        if o["id"] in seen: continue
        seen.add(o["id"]); uniq.append(o)
    return uniq[:N_PROMPTS]

# ---------- 阶段①:采样候选(占卡) ----------
def stage_sample(shard, nshards):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                 device_map="cuda", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, SFT_ADAPTER); model.eval()
    prompts = [p for i, p in enumerate(load_prompts()) if i % nshards == shard]
    outf = open(f"{OUT_DIR}/cand_shard{shard}.jsonl", "w", encoding="utf-8")
    for j, p in enumerate(prompts):
        msgs = ([{"role": "system", "content": p["system"]}] if p["system"] else []) + \
               [{"role": "user", "content": p["prompt"]}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        cands = []
        with torch.no_grad():
            for _ in range(K):
                out = model.generate(**enc, max_new_tokens=384, do_sample=True,
                                     temperature=0.9, top_p=0.95,
                                     pad_token_id=tok.eos_token_id)
                cands.append(tok.decode(out[0][enc["input_ids"].shape[1]:],
                                        skip_special_tokens=True).strip())
        outf.write(json.dumps({**p, "candidates": cands}, ensure_ascii=False) + "\n")
        if (j + 1) % 50 == 0:
            print(f"[shard{shard}] {j+1}/{len(prompts)}", flush=True)
    outf.close()

# ---------- 阶段②:MiniMax 裁判排序(不占卡) ----------
def load_key():
    for line in open(MINIMAX_ENV):
        if "=" in line:
            k, v = line.strip().split("=", 1)
            if "KEY" in k.upper() or "TOKEN" in k.upper():
                return v.strip().strip('"')
    raise RuntimeError("no key in .minimax_env")

JUDGE_SYS = ("你是严格的医疗回答质量裁判。评判标准:准确、安全(高危症状要提示就医)、"
             "有条理、贴合问题。**简洁准确优先,不因回答更长而加分**。"
             "只输出最好和最差候选的编号,格式:best=N;worst=M")

def stage_judge(shard):
    key = load_key()
    inf = f"{OUT_DIR}/cand_shard{shard}.jsonl"
    outf = open(f"{OUT_DIR}/pairs_shard{shard}.jsonl", "w", encoding="utf-8")
    for line in open(inf, encoding="utf-8"):
        d = json.loads(line)
        cands = d["candidates"]
        if len(set(cands)) < 2:  # 候选全同,无偏好信号
            continue
        listing = "\n".join(f"[{i}] {c[:400]}" for i, c in enumerate(cands))
        body = json.dumps({"model": "MiniMax-M3", "max_tokens": 4096,
                           "messages": [{"role": "system", "content": JUDGE_SYS},
                                        {"role": "user", "content":
                                         f"问题:{d['prompt']}\n\n候选:\n{listing}"}]}).encode()
        req = urllib.request.Request(MINIMAX_URL, data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                txt = json.loads(r.read())["choices"][0]["message"]["content"]
            txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)  # M3 剥思考
            b = int(re.search(r"best=(\d+)", txt).group(1))
            w = int(re.search(r"worst=(\d+)", txt).group(1))
            if b == w or b >= len(cands) or w >= len(cands): continue
            # 长度偏置防护:若 chosen 比 rejected 长 >2.5x,跳过(疑似长度偏好污染)
            if len(cands[b]) > 2.5 * max(1, len(cands[w])): continue
            outf.write(json.dumps({"system": d["system"], "prompt": d["prompt"],
                                   "chosen": cands[b], "rejected": cands[w]},
                                  ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"judge err {d['id']}: {e}", flush=True)
        time.sleep(0.3)
    outf.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sample", "judge"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=4)
    a = ap.parse_args()
    if a.stage == "sample":
        stage_sample(a.shard, a.nshards)
    else:
        stage_judge(a.shard)
