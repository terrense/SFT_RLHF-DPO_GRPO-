#!/usr/bin/env python3
"""GRPO验收:CMExam 1000题独立评测集,对比 DPO0.3 vs GRPO 的答对率(生成+抽字母)。
--stage dpo|grpo。GRPO=Instruct+merge(dpo)+grpo adapter。"""
import json, re, argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["dpo", "grpo"], required=True)
ap.add_argument("--n", type=int, default=1000)
A = ap.parse_args()

LAB = "/root/autodl-tmp"
BASE = f"{LAB}/models/Qwen3-8B-Instruct"
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                         device_map="cuda", trust_remote_code=True)
if A.stage == "dpo":
    model = PeftModel.from_pretrained(m, f"{LAB}/outputs/dpo_beta0.3")
else:
    m = PeftModel.from_pretrained(m, f"{LAB}/outputs/dpo_beta0.3").merge_and_unload()
    model = PeftModel.from_pretrained(m, f"{LAB}/outputs/grpo_final")
model.eval()
IM_END = tok.convert_tokens_to_ids("<|im_end|>"); EOT = tok.eos_token_id
SYS = "你是医学考试助手。仔细阅读题目和选项,只回答正确选项的字母(A/B/C/D/E),不要解释。"

rows = [json.loads(l) for l in open(f"{LAB}/data/rlhf/grpo_eval.jsonl", encoding="utf-8") if l.strip()][:A.n]
def ext(t):
    x = re.search(r"[ABCDE]", t.upper()); return x.group(0) if x else ""
correct = 0
for i, r in enumerate(rows):
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": f"题目:{r['question']}\n选项:\n{r['options']}\n请只回答正确选项的字母。"}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True, enable_thinking=False)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=8, do_sample=False,
                             eos_token_id=[IM_END, EOT], pad_token_id=EOT)
    ans = ext(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
    if ans == r["answer"].upper():
        correct += 1
    if (i + 1) % 200 == 0:
        print(f"{i+1}/{len(rows)} 当前{100*correct/(i+1):.1f}%", flush=True)
print(f"===== [{A.stage}] CMExam答对率 = {100*correct/len(rows):.1f}%  ({correct}/{len(rows)}) =====", flush=True)
