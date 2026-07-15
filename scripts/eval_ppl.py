#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基座在医疗对话数据上的困惑度(perplexity)评测 —— 衡量领域语言建模能力。
用法: python eval_ppl.py --model <path> --data disc500.jsonl --limit 500"""
import json, argparse, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--tag", default="base")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="cuda", trust_remote_code=True).eval()
    ROLE = {"user": "患者", "assistant": "医生"}
    tot_loss, tot_tok, n = 0.0, 0, 0
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            conv = o.get("conversation") or o.get("messages") or []
            text = "\n".join(f"{ROLE.get(m['role'], m['role'])}：{m['content']}" for m in conv)
            if not text:
                continue
            ids = tok(text, return_tensors="pt", truncation=True, max_length=args.max_len).to(model.device)
            ntok = ids["input_ids"].shape[1]
            if ntok < 2:
                continue
            with torch.no_grad():
                loss = model(**ids, labels=ids["input_ids"]).loss.item()
            tot_loss += loss * (ntok - 1)   # loss 是对 ntok-1 个位置的平均CE
            tot_tok += (ntok - 1)
            n += 1
            if n % 100 == 0:
                cur = tot_loss / tot_tok
                print(f"  {n}/{args.limit}  当前 ppl={math.exp(cur):.2f}", flush=True)
            if n >= args.limit:
                break
    avg = tot_loss / tot_tok
    print(f"\n===== [{args.tag}] DISC-Med-SFT {n}例 | 平均loss={avg:.4f} | perplexity={math.exp(avg):.2f} =====")

if __name__ == "__main__":
    main()
