#!/usr/bin/env python3
"""诊断倒豆子:同一 prompt,对比 generate 用 <|endoftext|>(旧,错) vs <|im_end|>(新,对) 停止符。
用现成 R4 adapter,不重训。占卡 1-2 分钟。"""
import torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "/root/autodl-tmp/models/Qwen3-8B-Base"
ADAPTER = "/root/autodl-tmp/outputs/r4/r4_arm_c_mixed"   # 混合臂
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                             device_map="cuda", trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER); model.eval()

IM_END = tok.convert_tokens_to_ids("<|im_end|>")   # 151645
EOT = tok.eos_token_id                              # 151643

SYS = ("你是一名专业的智能分诊助手。你会通过逐步提问采集患者的症状信息,"
       "每次提出一个清晰的问题(必要时给出可选项),最终给出参考诊断、建议就诊科室与处理建议。")
tests = ["最近总是头晕,今天早上起床眼前一黑差点摔倒", "我这两天嗓子疼得厉害,咽口水都费劲"]

def run(prompt, eos_ids, tag):
    msgs = [{"role": "system", "content": SYS},
            {"role": "assistant", "content": "您好,我是导医美小护,请问您哪里不舒服?"},
            {"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=400, do_sample=False,
                             eos_token_id=eos_ids, pad_token_id=EOT)
    txt = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    ntok = out.shape[1] - enc["input_ids"].shape[1]
    print(f"\n########## {tag} | 生成{ntok}token ##########")
    print(txt[:500])

for p in tests:
    print("\n" + "=" * 70 + f"\n患者: {p}")
    run(p, EOT, "旧(eos=endoftext 错停止符)")
    run(p, [IM_END, EOT], "新(eos含im_end 正确停止符)")
