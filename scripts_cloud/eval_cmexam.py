#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CMExam 医学单选准确率评测(选项logprob打分,对base公平)。
用法: python eval_cmexam.py --model <path> --data test.json --limit 500"""
import json, argparse, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load(mp):
    tok = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(mp, dtype=torch.bfloat16, device_map="cuda",
                                             trust_remote_code=True).eval()
    return tok, m

@torch.no_grad()
def predict(tok, m, q, options):
    letters = [o["key"] for o in options]
    opt_text = "\n".join(f'{o["key"]}. {o["value"]}' for o in options)
    prompt = f"以下是一道医学单项选择题，请选出正确答案。\n\n题目：{q}\n{opt_text}\n\n答案是："
    ids = tok(prompt, return_tensors="pt").to(m.device)
    lp = torch.log_softmax(m(**ids).logits[0, -1].float(), -1)
    best, bl = letters[0], -1e9
    for L in letters:
        t = tok.encode(L, add_special_tokens=False)
        if t and lp[t[0]].item() > bl:
            bl = lp[t[0]].item(); best = L
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=500); ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", default="base")
    a = ap.parse_args()
    allq = []
    for line in open(a.data, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        try: o = json.loads(line)
        except: continue
        if len(str(o.get("Answer", "")).strip()) == 1 and o.get("Options"):
            allq.append(o)
    random.seed(42); random.shuffle(allq)      # 随机抽样(可复现)
    data = allq[:a.limit]
    print(f"[CMExam] 单选题池={len(allq)} 随机抽{len(data)}道(seed42) | 模型 {a.tag}")
    tok, m = load(a.model)
    if a.adapter:
        from peft import PeftModel; m = PeftModel.from_pretrained(m, a.adapter)
    correct = 0
    for i, o in enumerate(data):
        if predict(tok, m, o["Question"], o["Options"]) == str(o["Answer"]).strip():
            correct += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(data)}  acc={correct/(i+1)*100:.1f}%", flush=True)
    print(f"\n===== [{a.tag}] CMExam {len(data)}题 准确率={correct/len(data)*100:.1f}% ({correct}/{len(data)}) =====")

if __name__ == "__main__":
    main()
