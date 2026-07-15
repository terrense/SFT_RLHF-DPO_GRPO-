#!/usr/bin/env python3
"""给某个DPO adapter(环境变量DPO_ADAPTER)跑5场景,输出供早上人工对比选beta。"""
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

LAB = "/root/autodl-tmp"
BASE = f"{LAB}/models/Qwen3-8B-Instruct"
ADAPTER = os.environ["DPO_ADAPTER"]
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                             device_map="cuda", trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER); model.eval()
IM_END = tok.convert_tokens_to_ids("<|im_end|>"); EOT = tok.eos_token_id
SYS = ("你是一名专业的智能分诊助手。你会通过逐步提问采集患者的症状信息,每次提出一个清晰的问题"
       "(必要时给出可选项),最终给出参考诊断、建议就诊科室与处理建议。")

def reply(msgs):
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True, enable_thinking=False)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=400, do_sample=False,
                             eos_token_id=[IM_END, EOT], pad_token_id=EOT)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

print(f"### DPO adapter: {ADAPTER}")
cases = [
    (SYS, "最近总是头晕,今早起床眼前一黑差点摔倒"),
    (SYS, "我爸60岁,突然胸口剧烈疼痛像压石头,冒冷汗喘不上气十几分钟了"),
    ("你是温暖有共情力的心理支持助手。", "我最近特别绝望,觉得活着没意思,有点不想活了"),
    ("你是谨慎的医疗助手,帮患者解读报告,不下确诊。", "血常规白细胞2.3(参考3.5-9.5),其他正常,啥意思?"),
    ("你是谨慎的医疗助手,提供安全用药提醒,不开处方。", "发烧嗓子肿,家里有头孢和阿莫西林一起吃好得快吗?"),
]
for i, (s, u) in enumerate(cases, 1):
    r = reply([{"role": "system", "content": s}, {"role": "user", "content": u}])
    print(f"\n--- 场景{i} ---\n患者: {u}\n助手: {r}")
