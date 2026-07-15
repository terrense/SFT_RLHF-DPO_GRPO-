#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医学选择题准确率评测(贯穿全链路的"医学知识"指标)
==================================================
方法:选项字母的"对数概率打分"(option log-likelihood),取概率最高的字母为模型答案。
不依赖模型会不会听"只回字母"的指令 → 对 base / CPT / SFT / DPO 各阶段都公平可比。

用法:
  # base 基线
  python eval_mcq.py --model /data/shenxin/rlhf_lab/models/Qwen3-8B-Base
  # 评测带 LoRA 适配器的某阶段产物
  python eval_mcq.py --model /data/.../Qwen3-8B-Base --adapter /data/.../outputs/sft_lora_r16_e2
输出:准确率 %(可机判,可写进面试结论)
"""
import json, argparse, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = "/data/shenxin/rlhf_lab/data/CMB/CMB-Exam/CMB-val/CMB-val-merge.json"

def load_model(model_path, adapter=None):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model

def format_options(opt):
    """返回 (选项文本, 可选字母列表)。兼容 option 是 dict 或 str。"""
    if isinstance(opt, dict):
        items = [(k, v) for k, v in opt.items() if str(v).strip()]
        text = "\n".join(f"{k}. {v}" for k, v in items)
        return text, [k for k, _ in items]
    s = str(opt)
    letters = [L for L in "ABCDEF" if re.search(rf"(^|[\s\n]){L}[\.、:：]", s)]
    return s, (letters or list("ABCD"))

@torch.no_grad()
def predict(tok, model, question, opt):
    opt_text, letters = format_options(opt)
    prompt = (f"以下是一道医学单项选择题，请选出正确答案。\n\n"
              f"题目：{question}\n{opt_text}\n\n答案是：")
    ids = tok(prompt, return_tensors="pt").to(model.device)
    logits = model(**ids).logits[0, -1]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    best, best_lp = letters[0], -1e9
    for L in letters:
        tid = tok.encode(L, add_special_tokens=False)
        if not tid:
            continue
        lp = logprobs[tid[0]].item()
        if lp > best_lp:
            best_lp, best = lp, L
    return best, letters

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--limit", type=int, default=0, help="0=全部单选题")
    ap.add_argument("--tag", default="model")
    args = ap.parse_args()

    data = json.load(open(args.data, encoding="utf-8"))
    single = [d for d in data if len(str(d.get("answer", "")).strip()) == 1]  # 只取单选
    if args.limit:
        single = single[:args.limit]
    print(f"[eval] 单选题 {len(single)} 道  | 模型 {args.tag}")

    tok, model = load_model(args.model, args.adapter)
    correct = 0
    for i, d in enumerate(single):
        pred, _ = predict(tok, model, d["question"], d.get("option", ""))
        if pred == str(d["answer"]).strip():
            correct += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(single)}  当前准确率 {correct/(i+1)*100:.1f}%", flush=True)
    acc = correct / len(single) * 100
    print(f"\n===== [{args.tag}] 医学选择题准确率 = {acc:.1f}%  ({correct}/{len(single)}) =====")

if __name__ == "__main__":
    main()
