#!/usr/bin/env python3
"""DPO阶段①:用正式SFT模型(Instruct+sft_final,nothink)对业务prompt采样K个候选。
四卡分片:--shard N --nshards 4。输出候选到 dpo/cand_shardN.jsonl。"""
import json, os, argparse, hashlib, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, required=True)
ap.add_argument("--nshards", type=int, default=4)
ap.add_argument("--n", type=int, default=2400)  # 总prompt数
ap.add_argument("--k", type=int, default=4)     # 每prompt候选数
A = ap.parse_args()

LAB = "/root/autodl-tmp"
BASE = f"{LAB}/models/Qwen3-8B-Instruct"
ADAPTER = f"{LAB}/outputs/sft_final"
PROMPTS_FILE = f"{LAB}/data/sft_v12/train_full.json"
OUT_DIR = f"{LAB}/data/rlhf/dpo"; os.makedirs(OUT_DIR, exist_ok=True)

def h32(s): return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                             device_map="cuda", trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER); model.eval()
IM_END = tok.convert_tokens_to_ids("<|im_end|>"); EOT = tok.eos_token_id

# 抽业务prompt(单轮/多轮首问),去重,分片
data = json.load(open(PROMPTS_FILE, encoding="utf-8"))
seen, prompts = set(), []
for d in data:
    conv = d.get("conversations", [])
    firstu = next((c["value"] for c in conv if c["from"] == "human"), None)
    if not firstu:
        continue
    pid = "%08x" % h32((d.get("system", "") + firstu)[:200])
    if pid in seen:
        continue
    seen.add(pid)
    prompts.append({"id": pid, "system": d.get("system", ""), "prompt": firstu})
prompts.sort(key=lambda x: h32(x["id"]))
prompts = prompts[:A.n]
mine = [p for i, p in enumerate(prompts) if i % A.nshards == A.shard]
print(f"[shard{A.shard}] 负责 {len(mine)} prompts", flush=True)

def gen(system, prompt, temp):
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True, enable_thinking=False)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=400, do_sample=(temp > 0),
                             temperature=temp if temp > 0 else None,
                             top_p=0.95 if temp > 0 else None,
                             eos_token_id=[IM_END, EOT], pad_token_id=EOT)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

outf = open(f"{OUT_DIR}/cand_shard{A.shard}.jsonl", "w", encoding="utf-8")
for j, p in enumerate(mine):
    cands = []
    for ki in range(A.k):
        temp = 0.0 if ki == 0 else 0.9   # 1个greedy + K-1个采样,保证多样
        try:
            cands.append(gen(p["system"], p["prompt"], temp))
        except Exception as e:
            print(f"gen err {p['id']}: {e}", flush=True)
    cands = [c for c in cands if c and len(c) > 3]
    if len(set(cands)) >= 2:
        outf.write(json.dumps({**p, "candidates": cands}, ensure_ascii=False) + "\n")
        outf.flush()
    if (j + 1) % 50 == 0:
        print(f"[shard{A.shard}] {j+1}/{len(mine)}", flush=True)
outf.close()
print(f"[shard{A.shard}] DONE", flush=True)
