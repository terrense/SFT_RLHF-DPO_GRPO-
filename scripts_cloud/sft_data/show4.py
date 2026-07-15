#!/usr/bin/env python3
"""打印4条完整多轮预问诊,qwen3模板渲染后的原文(保留<|im_end|>等特殊标记)。"""
import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/Qwen3-8B-Base", trust_remote_code=True)

data = json.load(open("/root/autodl-tmp/data/sft_v11/train_full.json", encoding="utf-8"))
pc = [d for d in data if sum(1 for c in d["conversations"] if c["from"] == "gpt") >= 2]

for idx in range(4):
    d = pc[idx]
    msgs = ([{"role": "system", "content": d["system"]}] if d.get("system") else [])
    for c in d["conversations"]:
        msgs.append({"role": "user" if c["from"] == "human" else "assistant",
                     "content": c["value"]})
    # tokenize后再decode,保留特殊token(skip_special_tokens=False)
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False,
                                  return_dict=False)
    text = tok.decode(ids, skip_special_tokens=False)
    nturn = len(msgs)
    nimend = text.count("<|im_end|>")
    print("=" * 72)
    print(f"### 第{idx+1}条  |  消息数(system+user+assistant)={nturn}  |  <|im_end|>数量={nimend}")
    print("=" * 72)
    print(text)
    print()
