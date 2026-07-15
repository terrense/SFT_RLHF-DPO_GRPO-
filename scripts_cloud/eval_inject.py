#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识注入率评测:模型生成回答 → 命中 keywords 任一即算对 → 准确率%
base 应≈0%(不可能知道虚构规范),CPT 注入后应大幅上升。
用法:
  python eval_inject.py --model models/Qwen3-8B-Base --tag base
  python eval_inject.py --model models/Qwen3-8B-Base --adapter outputs/cpt_synth_lora --tag cpt
"""
import json, argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROBE = "/data/shenxin/rlhf_lab/data/synthetic_kb/probe.jsonl"

def load(model_path, adapter=None):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                                                 device_map="cuda", trust_remote_code=True)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok, model

@torch.no_grad()
def answer(tok, model, q):
    prompt = f"问题：{q}\n回答："
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=80, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--probe", default=PROBE)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--show", type=int, default=0, help="打印前N条问答看效果")
    args = ap.parse_args()

    probes = [json.loads(l) for l in open(args.probe, encoding="utf-8") if l.strip()]
    print(f"[inject] 探针 {len(probes)} 道 | 模型 {args.tag}")
    tok, model = load(args.model, args.adapter)
    hit = 0
    for i, p in enumerate(probes):
        gen = answer(tok, model, p["question"])
        ok = any(k in gen for k in p["keywords"])
        hit += ok
        if args.show and i < args.show:
            print(f"  Q:{p['question']}\n  A:{gen.strip()[:80]}  -> {'✓' if ok else '✗'}(需含{p['keywords']})")
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(probes)}  命中率 {hit/(i+1)*100:.1f}%", flush=True)
    acc = hit / len(probes) * 100
    print(f"\n===== [{args.tag}] 知识注入率 = {acc:.1f}%  ({hit}/{len(probes)}) =====")

if __name__ == "__main__":
    main()
